import numpy as np
from .go2_robot_data import PinGo2Model

# --------------------------------------------------------------------------------
# Gait Setting
# --------------------------------------------------------------------------------

PHASE_OFFSET = np.array([0.5, 0.0, 0.0, 0.5]).reshape(4)    # trotting gait
HEIGHT_SWING = 0.1 # Height of the swing leg trajectory apex

class Gait:
    def __init__(self, frequency_hz, duty):
        self.gait_duty = duty
        self.gait_hz = frequency_hz

        self.gait_period = 1.0 / frequency_hz
        self.stance_time = self.gait_duty * self.gait_period
        self.swing_time = (1 - self.gait_duty) * self.gait_period

    def compute_current_mask(self, time):
        mask = self.compute_contact_table(time, 0, 1)
        return mask.flatten()
    
    def compute_contact_table(self, t0: float, dt: float, N: int) -> np.ndarray:
        t = t0 + np.arange(N) * dt
        t = t + dt / 2

        phases = np.mod(PHASE_OFFSET[:, None] + t[None, :] / self.gait_period, 1.0)
        contact_table = (phases < self.gait_duty).astype(np.int32)
        return contact_table        

    def compute_touchdown_world_for_traj_purpose_only(self, go2: PinGo2Model, leg: str):
        base_pos = go2.base_pos
        base_vel = go2.base_vel
        yaw_rate = go2.yaw_rate_des_world

        # Use Genesis thigh link_origin for accurate shoulder world position
        hip_pos_world = go2.hip_pos_world[leg]

        t_swing = self.swing_time
        t_stance = self.stance_time

        T = t_swing + 0.5 * t_stance
        pred_time = T / 2.0

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]
        pos_drift_term = [base_vel[0] * pred_time, base_vel[1] * pred_time, 0]

        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
            -dtheta * r_xy[1],
             dtheta * r_xy[0],
             0.0
        ])
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(rotation_correction_term))

        return pos_touchdown_world

    def compute_swing_traj_and_touchdown(self, go2: PinGo2Model, leg: str):
        base_pos = go2.base_pos
        pos_com_world = go2.pos_com_world
        vel_com_world = go2.vel_com_world
        yaw_rate = go2.yaw_rate_des_world

        # Use Genesis thigh link_origin for accurate shoulder world position
        hip_pos_world = go2.hip_pos_world[leg]
        foot_pos, foot_vel = go2.get_single_foot_state_in_world(leg)

        x_vel_des = go2.x_vel_des_world
        y_vel_des = go2.y_vel_des_world
        x_pos_des = go2.x_pos_des_world
        y_pos_des = go2.y_pos_des_world

        t_swing = self.swing_time
        t_stance = self.stance_time

        T = t_swing + 0.5 * t_stance
        pred_time = T / 2.0

        k_v_x = 0.4 * T
        k_p_x = 0.1
        k_v_y = 0.2 * T
        k_p_y = 0.05

        pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]
        pos_drift_term = [x_vel_des * pred_time, y_vel_des * pred_time, 0]
        pos_correction_term = [k_p_x * (pos_com_world[0] - x_pos_des), k_p_y * (pos_com_world[1] - y_pos_des), 0]
        vel_correction_term = [k_v_x * (vel_com_world[0] - x_vel_des), k_v_y * (vel_com_world[1] - y_vel_des), 0]
        
        dtheta = yaw_rate * pred_time
        center_xy = np.array([base_pos[0], base_pos[1]])
        r_xy = np.array([pos_norminal_term[0], pos_norminal_term[1]]) - center_xy

        rotation_correction_term = np.array([
            -dtheta * r_xy[1],
             dtheta * r_xy[0],
             0.0
        ])
    
        pos_touchdown_world = (np.array(pos_norminal_term)
                                + np.array(pos_drift_term)
                                + np.array(pos_correction_term)
                                + np.array(vel_correction_term)
                                + np.array(rotation_correction_term))
        
        pos_foot_traj_eval = self.make_swing_trajectory(foot_pos, pos_touchdown_world, t_swing, h_sw=HEIGHT_SWING)
        return pos_foot_traj_eval, pos_touchdown_world

    def make_swing_trajectory(self, p0, pf, t_swing, h_sw):
        p0 = np.asarray(p0, dtype=float)
        pf = np.asarray(pf, dtype=float)
        T = float(t_swing)
        dp = pf - p0

        def eval_at(t):
            s = np.clip(t / T, 0.0, 1.0)
            mj   = 10*s**3 - 15*s**4 + 6*s**5
            dmj  = 30*s**2 - 60*s**3 + 30*s**4
            d2mj = 60*s    - 180*s**2 + 120*s**3

            p = p0 + dp * mj
            v = (dp * dmj) / T
            a = (dp * d2mj) / (T**2)

            if h_sw != 0.0:
                b    = 64 * s**3 * (1 - s)**3
                db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)
                d2b  = 192 * ( 2*s*(1 - s)**2*(1 - 2*s) - 2*s**2*(1 - s)*(1 - 2*s) - 2*s**2*(1 - s)**2 )

                p[2] += h_sw * b
                v[2] += h_sw * db / T
                a[2] += h_sw * d2b / (T**2)

            return p, v, a

        return eval_at
