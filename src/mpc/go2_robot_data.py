import pinocchio as pin
import numpy as np

class PinGo2Model:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        # Foot frame names (must match URDF)
        self.foot_names = {
            "FL": "FL_foot",
            "FR": "FR_foot",
            "RL": "RL_foot",
            "RR": "RR_foot"
        }
        
        self.foot_ids = {
            leg: self.model.getFrameId(name)
            for leg, name in self.foot_names.items()
        }
        
        # State
        self.q = pin.neutral(self.model)
        self.dq = np.zeros(self.model.nv)

        # COM state in world
        self.pos_com_world = np.zeros(3)
        self.vel_com_world = np.zeros(3)

        # Base orientation and rate
        self.R_world_to_body = np.eye(3)
        self.R_z = np.eye(3)
        self.yaw = 0.0

        # Trajectory targets
        self.x_vel_des_world = 0.0
        self.y_vel_des_world = 0.0
        self.x_pos_des_world = 0.0
        self.y_pos_des_world = 0.0
        self.yaw_rate_des_world = 0.0

        # Hip (shoulder) world positions — populated by Genesis_GO2_Model.update_pin_with_genesis()
        # using get_links_pos(thigh links, ref='link_origin') for Raibert heuristic
        self.hip_pos_world = {
            "FL": np.array([ 0.1934,  0.0465, 0.0]),
            "FR": np.array([ 0.1934, -0.0465, 0.0]),
            "RL": np.array([-0.1934,  0.0465, 0.0]),
            "RR": np.array([-0.1934, -0.0465, 0.0]),
        }

    def update_model(self, q: np.ndarray, dq: np.ndarray):
        """
        q: [px, py, pz, qx, qy, qz, qw, joint_angles...]
        dq: [vx_body, vy_body, vz_body, wx_body, wy_body, wz_body, joint_velocities...]
        """
        self.q[:] = q
        self.dq[:] = dq

        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.crba(self.model, self.data, self.q)
        pin.nonLinearEffects(self.model, self.data, self.q, self.dq)
        pin.computeAllTerms(self.model, self.data, self.q, self.dq)

        self._update_com_state()
        self._update_rotation()

    def update_model_simplified(self, q_base_pos, dq_base_vel):
        """Used in trajectory generation to approximate state"""
        self.q[0:3] = q_base_pos[0:3]
        self.q[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(q_base_pos[3:6])).coeffs()
        self.dq[0:6] = dq_base_vel[0:6]
        
        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        self.pos_com_world = pin.centerOfMass(self.model, self.data, self.q, self.dq)[0]
        
    def _update_com_state(self):
        pin.centerOfMass(self.model, self.data, self.q, self.dq)
        self.pos_com_world = self.data.com[0]
        self.vel_com_world = self.data.vcom[0]

    def _update_rotation(self):
        quat = pin.Quaternion(self.q[6], self.q[3], self.q[4], self.q[5]) # qw, qx, qy, qz
        self.R_world_to_body = quat.toRotationMatrix().T
        
        # Extract yaw for heading
        rpy = pin.rpy.matrixToRpy(quat.toRotationMatrix())
        self.yaw = rpy[2]
        self.R_z = pin.rpy.rpyToMatrix(0, 0, self.yaw)

    def compute_com_x_vec(self):
        """Returns the 12D centroidal state vector"""
        rpy = pin.rpy.matrixToRpy(self.R_world_to_body.T)
        # Assuming dq[0:3] is v_body, dq[3:6] is w_body
        v_world = self.R_world_to_body.T @ self.dq[0:3]
        w_world = self.R_world_to_body.T @ self.dq[3:6]
        return np.concatenate([
            self.pos_com_world,
            rpy,
            v_world,
            w_world
        ])

    def get_hip_offset(self, leg: str):
        # In Pinocchio, get hip joint frame placement relative to base
        # This is a bit ad-hoc, but usually it's the placement of the hip joint in the base frame.
        hip_joint_name = f"{leg}_hip_joint"
        hip_id = self.model.getJointId(hip_joint_name)
        if hip_id < len(self.model.joints):
            return self.model.jointPlacements[hip_id].translation
        # Fallback to rough offset
        return np.array([0.19, 0.046 if 'L' in leg else -0.046, 0])

    def get_single_foot_state_in_world(self, leg: str):
        foot_id = self.foot_ids[leg]
        placement = self.data.oMf[foot_id]
        pos = placement.translation
        vel = pin.getFrameVelocity(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        return pos, vel

    def get_foot_lever_world(self):
        levers = []
        for leg in ["FL", "FR", "RL", "RR"]:
            foot_id = self.foot_ids[leg]
            pos = self.data.oMf[foot_id].translation
            levers.append(pos - self.pos_com_world)
        return levers

    def compute_3x3_foot_Jacobian_world(self, leg: str):
        foot_id = self.foot_ids[leg]
        J = pin.computeFrameJacobian(self.model, self.data, self.q, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # J is 6xNV, we only need the 3x3 block for this leg's joints
        # Find which joints belong to this leg.
        # Typically joints are arranged as [base(6), FL(3), FR(3), RL(3), RR(3)]
        # We can map from leg name
        leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}[leg]
        start = 6 + leg_idx * 3
        return J[:3, start:start+3]

    def compute_full_foot_Jacobian_world(self, leg: str):
        foot_id = self.foot_ids[leg]
        J = pin.computeFrameJacobian(self.model, self.data, self.q, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J[:3, :]

    def compute_Jdot_dq_world(self, leg: str):
        foot_id = self.foot_ids[leg]
        acc = pin.getFrameClassicalAcceleration(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return acc.linear

    def compute_dynamcis_terms(self):
        # M is crba, C is nle minus g? Or nle is C*dq + g
        # pin.nle(model, data, q, dq) returns C*dq + g
        # pin.computeGeneralizedGravity(model, data, q) returns g
        M = self.data.M
        # Make M symmetric
        M = np.triu(M) + np.triu(M, 1).T
        g = pin.computeGeneralizedGravity(self.model, self.data, self.q)
        C_dq = pin.nonLinearEffects(self.model, self.data, self.q, self.dq) - g
        return g, C_dq, M

    @property
    def base_pos(self):
        return self.q[0:3]

    @property
    def base_vel(self):
        # world velocity
        return self.R_world_to_body.T @ self.dq[0:3]
