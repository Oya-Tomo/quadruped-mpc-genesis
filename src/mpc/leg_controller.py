import numpy as np
from .go2_robot_data import PinGo2Model
from .gait import Gait
from dataclasses import dataclass

KP_SWING = np.diag([400, 400, 400])
KD_SWING = np.diag([75, 75, 75])

LEG_INDEX = {
    "FL": 0,
    "FR": 1,
    "RL": 2,
    "RR": 3,
}

JOINT_SLICES = {
    "FL": slice(6, 9),
    "FR": slice(9, 12),
    "RL": slice(12, 15),
    "RR": slice(15, 18),
}

@dataclass
class LegOutput:
    tau: np.ndarray       
    pos_des: np.ndarray   
    pos_now: np.ndarray   
    vel_des: np.ndarray   
    vel_now: np.ndarray   


class LegController:
    def __init__(self):
        self.last_gait_mask = np.array([2, 2, 2, 2])

    def compute_leg_torque(
        self,
        leg: str,
        go2: PinGo2Model,
        gait: Gait,
        contact_force: np.ndarray,
        current_time: float,
        real_contact: np.ndarray | None = None,
    ):
        leg_idx = LEG_INDEX[leg]
        joint_slice = JOINT_SLICES[leg]

        J_foot_world = go2.compute_3x3_foot_Jacobian_world(leg)      
        J_full_foot_world = go2.compute_full_foot_Jacobian_world(leg)  
        g, C_dq, M = go2.compute_dynamcis_terms()

        # Gait timing tells us when to plan swing trajectory
        gait_mask = gait.compute_current_mask(current_time)
        tau_cmd = np.zeros(3)

        foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)
        foot_pos_des = foot_pos_now.copy()
        foot_vel_des = foot_vel_now.copy()

        # Detect takeoff from gait schedule (plan swing trajectory early)
        if self.last_gait_mask[leg_idx] != gait_mask[leg_idx] and gait_mask[leg_idx] == 0:
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj_fn, td_pos = gait.compute_swing_traj_and_touchdown(go2, leg)
            setattr(self, f"{leg}_traj", traj_fn)
            setattr(self, f"{leg}_td_pos", td_pos)

        # Decide actual mode:
        # - Swing if gait says swing AND foot is not in contact
        # - Stance otherwise (gait says stance, OR foot is on ground)
        gait_swing = (gait_mask[leg_idx] == 0)
        if real_contact is not None:
            in_swing = gait_swing and not bool(real_contact[leg_idx])
        else:
            in_swing = gait_swing

        if in_swing:
            # Swing phase: track foot trajectory
            takeoff_time = getattr(self, f"{leg}_takeoff_time", current_time)
            traj_fn = getattr(self, f"{leg}_traj", None)

            if traj_fn is not None:
                time_since_takeoff = current_time - takeoff_time
                foot_pos_des, foot_vel_des, foot_acl_des = traj_fn(time_since_takeoff)
                foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

                pos_error = foot_pos_des - foot_pos_now
                vel_error = foot_vel_des - foot_vel_now

                Lambda = np.linalg.inv(
                    J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T
                )
                Jdot_dq = go2.compute_Jdot_dq_world(leg)
                f_ff = Lambda @ (foot_acl_des - Jdot_dq)

                force = KP_SWING @ pos_error + KD_SWING @ vel_error + f_ff
                tau_cmd = J_foot_world.T @ force + (C_dq + g)[joint_slice]
            else:
                tau_cmd = g[joint_slice]
        else:
            # Stance phase: apply MPC contact force
            tau_cmd = J_foot_world.T @ -contact_force

        self.last_gait_mask[leg_idx] = gait_mask[leg_idx]

        return LegOutput(
            tau=tau_cmd.reshape(3,),
            pos_des=foot_pos_des,
            pos_now=foot_pos_now,
            vel_des=foot_vel_des,
            vel_now=foot_vel_now,
        )
