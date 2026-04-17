import numpy as np
import pinocchio as pin
from .go2_robot_data import PinGo2Model
from .gait import Gait

class ComTraj:
    def __init__(self, go2: PinGo2Model, urdf_path: str):
        self.dummy_go2 = PinGo2Model(urdf_path)
        x_vec = go2.compute_com_x_vec().reshape(-1)
        self.pos_des_world = x_vec[0:3].copy()
        
        self.initial_x_vec = None
        self.pos_traj_world = None
        self.rpy_traj_world = None
        self.vel_traj_world = None
        self.omega_traj_world = None
        
        self.m = 0.0
        self.I_com_world = np.zeros((3,3))
        self.N = 0
        self.time = None
        self.contact_table = None

    def compute_x_ref_vec(self):
        refs = [
            self.pos_traj_world,
            self.rpy_traj_world,
            self.vel_traj_world,
            self.omega_traj_world,
        ]
        N = min(r.shape[1] for r in refs)
        ref_traj = np.vstack([r[:, :N] for r in refs])
        return ref_traj      

    def generate_traj(self,
                      go2: PinGo2Model,
                      gait: Gait,
                      time_now: float,
                      x_vel_des_body: float,
                      y_vel_des_body: float,
                      z_pos_des_body: float,
                      yaw_rate_des_body: float,
                      time_step: float):
        
        self.initial_x_vec = go2.compute_com_x_vec()
        initial_pos = self.initial_x_vec[0:3]
        self.m = sum([i.mass for i in go2.model.inertias])
        # Simple diagonal approximation or actual COM inertia could be better,
        # but for MPC often the base link inertia + leg masses as points is used.
        # We will extract approximate composite inertia around CoM.
        import pinocchio as pin
        pin.ccrba(go2.model, go2.data, go2.q, go2.dq)
        self.I_com_world = go2.data.Ig.inertia
        
        x0, y0, z0 = initial_pos
        yaw = self.initial_x_vec[5]
        self.dummy_go2.yaw_rate_des_world = yaw_rate_des_body
        time_horizon = gait.gait_period

        max_pos_error = 0.1
        
        if self.pos_des_world[0] - x0 > max_pos_error:
            self.pos_des_world[0] = x0 + max_pos_error
        if x0 - self.pos_des_world[0] > max_pos_error:
            self.pos_des_world[0] = x0 - max_pos_error

        if self.pos_des_world[1] - y0 > max_pos_error:
            self.pos_des_world[1] = y0 + max_pos_error
        if y0 - self.pos_des_world[1] > max_pos_error:
            self.pos_des_world[1] = y0 - max_pos_error

        self.pos_des_world[2] = z_pos_des_body

        go2.x_pos_des_world = self.pos_des_world[0]
        go2.y_pos_des_world = self.pos_des_world[1]
        
        self.N = int(time_horizon / time_step)
        N = self.N
        t_vec = (np.arange(N) + 1) * time_step

        R_z = go2.R_z
        vel_desired_world = R_z @ np.array([x_vel_des_body, y_vel_des_body, 0.0])
        go2.x_vel_des_world = vel_desired_world[0]
        go2.y_vel_des_world = vel_desired_world[1]

        self.time = t_vec
        self.pos_traj_world   = np.zeros((3, N))
        self.vel_traj_world   = np.zeros((3, N))
        self.rpy_traj_world   = np.zeros((3, N))
        self.omega_traj_world = np.zeros((3, N))

        self.pos_traj_world[:, :] = (
            self.pos_des_world.reshape(3, 1) + (vel_desired_world.reshape(3, 1) * t_vec.reshape(1, N))
        )

        self.vel_traj_world[:, :] = vel_desired_world.reshape(3, 1)

        self.rpy_traj_world[0, :] = 0.0
        self.rpy_traj_world[1, :] = 0.0
        self.rpy_traj_world[2, :] = yaw + yaw_rate_des_body * t_vec

        self.omega_traj_world[0, :] = 0.0
        self.omega_traj_world[1, :] = 0.0
        self.omega_traj_world[2, :] = yaw_rate_des_body
        go2.yaw_rate_des_world = yaw_rate_des_body

        self.contact_table = gait.compute_contact_table(time_now, time_step, N)

        r_fl_traj_world = np.zeros((3,N))
        r_fr_traj_world = np.zeros((3,N))
        r_rl_traj_world = np.zeros((3,N))
        r_rr_traj_world = np.zeros((3,N))

        # Initialize with current foot positions (relative to CoM) as the starting lever arms
        current_levers = go2.get_foot_lever_world()
        r_next_td_world = list(current_levers)

        # Seed initial contact state from gait at t=time_now
        mask_previous = gait.compute_contact_table(time_now, 0, 1).flatten()

        q_sim = np.zeros(6)
        dq_sim = np.zeros(6)

        for i in range(N):
            current_mask = gait.compute_contact_table(time_now + i * time_step, 0, 1).flatten()

            q_sim[0:3] = self.pos_traj_world[:,i]
            q_sim[3:6] = self.rpy_traj_world[:,i]
            
            R = go2.R_world_to_body
            v_world = self.vel_traj_world[:,i]
            w_world = self.omega_traj_world[:,i]
            dq_sim[0:3] = R @ v_world
            dq_sim[3:6] = R @ w_world
            self.dummy_go2.update_model_simplified(q_sim, dq_sim)

            # Update hip_pos_world for dummy_go2 using predicted base pos + rotated static offset
            p_base_traj_world = self.dummy_go2.base_pos
            yaw_i = self.rpy_traj_world[2, i]
            R_z_i = pin.rpy.rpyToMatrix(0, 0, yaw_i)
            for leg_name in ["FL", "FR", "RL", "RR"]:
                hip_off = go2.get_hip_offset(leg_name)
                self.dummy_go2.hip_pos_world[leg_name] = (
                    np.array([p_base_traj_world[0], p_base_traj_world[1], hip_off[2]])
                    + R_z_i @ hip_off
                )

            p_base_traj_world = self.dummy_go2.base_pos

            for leg_idx, leg_name, r_traj in zip(
                range(4), 
                ["FL", "FR", "RL", "RR"], 
                [r_fl_traj_world, r_fr_traj_world, r_rl_traj_world, r_rr_traj_world],
            ):
                prev = mask_previous[leg_idx]
                curr = current_mask[leg_idx]
                if prev != curr and curr == 0:
                    # leg just took off: plan touchdown position
                    pos_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, leg_name)
                    r_next_td_world[leg_idx] = pos_next_td_world - p_base_traj_world
                    # During swing use zero (or previous stance pos)
                    r_traj[:,i] = r_traj[:,i-1] if i > 0 else current_levers[leg_idx]
                elif prev != curr and curr == 1:
                    # leg just landed: use predicted touchdown as stance point
                    r_traj[:,i] = r_next_td_world[leg_idx]
                else:
                    # no transition: carry forward
                    r_traj[:,i] = r_traj[:,i-1] if i > 0 else current_levers[leg_idx]

            mask_previous = current_mask

        self.r_fl_foot_world = r_fl_traj_world
        self.r_fr_foot_world = r_fr_traj_world
        self.r_rl_foot_world = r_rl_traj_world
        self.r_rr_foot_world = r_rr_traj_world

        self._discreteDynamics(time_step)

    def _skew(self, vector):
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _discreteDynamics(self, dt: float):
        N = self.N
        m = self.m
        I_inv = np.linalg.inv(self.I_com_world)

        yaw_avg = float(np.average(self.rpy_traj_world[2, :]))
        cy, sy = np.cos(yaw_avg), np.sin(yaw_avg)
        RzT = np.array([[cy, sy, 0.0],
                        [-sy, cy, 0.0],
                        [0.0, 0.0, 1.0]], dtype=float)

        # ---- Ad ----
        Ad = np.eye(12, dtype=float)
        Ad[0:3, 6:9] = dt * np.eye(3)
        Ad[3:6, 9:12] = dt * RzT
        self.Ad = Ad

        # ---- gd ----
        g = np.array([0.0, 0.0, -9.81], dtype=float)
        gd = np.zeros((12, 1), dtype=float)
        gd[0:3, 0] = 0.5 * g * dt * dt
        gd[6:9, 0] = g * dt
        self.gd = gd

        # ---- Bd ----
        Bd = np.zeros((N, 12, 12), dtype=float)

        Bp = (0.5 * dt * dt / m) * np.eye(3)
        Bv = (dt / m) * np.eye(3)

        for i in range(N):
            r1 = self.r_fl_foot_world[:, i]
            r2 = self.r_fr_foot_world[:, i]
            r3 = self.r_rl_foot_world[:, i]
            r4 = self.r_rr_foot_world[:, i]

            W1 = I_inv @ self._skew(r1)
            W2 = I_inv @ self._skew(r2)
            W3 = I_inv @ self._skew(r3)
            W4 = I_inv @ self._skew(r4)

            Bi = Bd[i]

            Bi[0:3, 0:3]   = Bp
            Bi[0:3, 3:6]   = Bp
            Bi[0:3, 6:9]   = Bp
            Bi[0:3, 9:12]  = Bp

            Bi[6:9, 0:3]   = Bv
            Bi[6:9, 3:6]   = Bv
            Bi[6:9, 6:9]   = Bv
            Bi[6:9, 9:12]  = Bv

            Bi[9:12, 0:3]  = dt * W1
            Bi[9:12, 3:6]  = dt * W2
            Bi[9:12, 6:9]  = dt * W3
            Bi[9:12, 9:12] = dt * W4

            Bi[3:6, 0:3]   = 0.5 * dt * dt * (RzT @ W1)
            Bi[3:6, 3:6]   = 0.5 * dt * dt * (RzT @ W2)
            Bi[3:6, 6:9]   = 0.5 * dt * dt * (RzT @ W3)
            Bi[3:6, 9:12]  = 0.5 * dt * dt * (RzT @ W4)

        self.Bd = Bd
