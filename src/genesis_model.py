import genesis as gs
import numpy as np
from mpc.go2_robot_data import PinGo2Model
import pinocchio as pin

# Genesis URDF: foot links are fixed-joint children that get merged by default.
# Keep them so get_links_net_contact_force() can detect per-foot contact.
FOOT_LINK_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
# Thigh links: their link_origin = hip joint world position (shoulder for Raibert heuristic)
THIGH_LINK_NAMES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
# Threshold for considering a foot in contact [N]
CONTACT_FORCE_THRESHOLD = 5.0

class Genesis_GO2_Model:
    def __init__(self, scene, urdf_path: str):
        self.scene = scene
        self.robot = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=(0, 0, 0.50),
                quat=(1, 0, 0, 0),
                links_to_keep=FOOT_LINK_NAMES,
            )
        )
        # Cache foot link local indices (resolved after build)
        self._foot_link_idx_local = None
        self._thigh_link_idx_local = None  # for hip world positions

    def _resolve_foot_links(self):
        """Cache local link indices for foot/thigh links after scene is built."""
        if self._foot_link_idx_local is not None:
            return
        self._foot_link_idx_local  = [self.robot.get_link(n).idx_local for n in FOOT_LINK_NAMES]
        self._thigh_link_idx_local = [self.robot.get_link(n).idx_local for n in THIGH_LINK_NAMES]

    def get_contact_mask(self) -> np.ndarray:
        """
        Returns a boolean array of shape (4,) in order [FL, FR, RL, RR].
        True  = foot is in contact (Fz > threshold).
        Uses Genesis get_links_net_contact_force() for ground-truth contact detection.
        """
        self._resolve_foot_links()
        # shape: (n_links, 3) — world-frame contact force 3D vector on each link
        all_forces = self.robot.get_links_net_contact_force().cpu().numpy()  # (n_links, 3)
        contact = np.zeros(4, dtype=bool)
        for i, idx in enumerate(self._foot_link_idx_local):
            force_norm = np.linalg.norm(all_forces[idx])   # magnitude of 3D contact force
            contact[i] = force_norm > CONTACT_FORCE_THRESHOLD
        return contact

    def update_pin_with_genesis(self, go2: PinGo2Model):
        # Genesis state extraction
        base_pos = self.robot.get_pos().cpu().numpy().astype(np.float64)
        base_quat = self.robot.get_quat().cpu().numpy().astype(np.float64)  # [w, x, y, z]

        dofs_pos = self.robot.get_dofs_position().cpu().numpy().astype(np.float64)
        dofs_vel = self.robot.get_dofs_velocity().cpu().numpy().astype(np.float64)

        v_world = self.robot.get_vel().cpu().numpy().astype(np.float64)
        w_world = self.robot.get_ang().cpu().numpy().astype(np.float64)

        # Convert Genesis quat [w,x,y,z] → rotation matrix
        qw, qx, qy, qz = base_quat
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()

        # Pinocchio FreeFlyer dq = [v_body, w_body, dq_joints]
        v_body = R.T @ v_world
        w_body = R.T @ w_world

        # Genesis joint order (grouped by type):
        #   [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        #    FL_calf, FR_calf, RL_calf, RR_calf]
        # Pinocchio joint order (grouped by leg):
        #   [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
        #    RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        gen2pin_idx = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

        actuated_pos_pin = dofs_pos[6:18][gen2pin_idx]
        actuated_vel_pin = dofs_vel[6:18][gen2pin_idx]

        q_pin  = np.concatenate([base_pos, [qx, qy, qz, qw], actuated_pos_pin])
        dq_pin = np.concatenate([v_body,   w_body,            actuated_vel_pin])

        go2.update_model(q_pin, dq_pin)

        # Update hip world positions from Genesis thigh link origins
        # (more accurate than Pinocchio static offset + yaw rotation)
        self._resolve_foot_links()
        all_link_pos = self.robot.get_links_pos(
            links_idx_local=self._thigh_link_idx_local, ref="link_origin"
        ).cpu().numpy()  # shape (4, 3)
        go2.hip_pos_world = {
            "FL": all_link_pos[0].astype(np.float64),
            "FR": all_link_pos[1].astype(np.float64),
            "RL": all_link_pos[2].astype(np.float64),
            "RR": all_link_pos[3].astype(np.float64),
        }

    def set_joint_torque(self, torque: np.ndarray):
        """torque is in Pinocchio leg order → remap to Genesis type order."""
        pin2gen_idx = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        self.robot.control_dofs_force(torque[pin2gen_idx], dofs_idx_local=np.arange(6, 18))


