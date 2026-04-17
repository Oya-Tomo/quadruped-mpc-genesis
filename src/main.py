import genesis as gs
import numpy as np
import time
from dataclasses import dataclass, field

from genesis_model import Genesis_GO2_Model
from mpc.go2_robot_data import PinGo2Model
from mpc.com_trajectory import ComTraj
from mpc.centroidal_mpc import CentroidalMPC
from mpc.leg_controller import LegController
from mpc.gait import Gait

RUN_SIM_LENGTH_S = 10.0
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

TAU_LIM = 0.9 * np.array([
    23.7, 23.7, 45.43,   # FL
    23.7, 23.7, 45.43,   # FR
    23.7, 23.7, 45.43,   # RL
    23.7, 23.7, 45.43,   # RR
])

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

def main():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=SIM_DT,
            substeps=1,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(2.0, -2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
    )

    scene.add_entity(gs.morphs.Plane())
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "../assets/URDF/go2_description/urdf/go2_description.urdf")
    urdf_path = os.path.normpath(urdf_path)
    genesis_go2 = Genesis_GO2_Model(scene, urdf_path)

    scene.build()

    go2 = PinGo2Model(urdf_path)
    leg_controller = LegController()
    traj = ComTraj(go2, urdf_path)
    gait = Gait(GAIT_HZ, GAIT_DUTY)

    # Allow robot to settle
    print("Settling robot...")
    init_q_pin = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5])
    pin2gen_idx = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    init_q_gen = init_q_pin[pin2gen_idx]

    genesis_go2.robot.control_dofs_position(
        init_q_gen,
        dofs_idx_local=np.arange(6, 18)
    )
    for _ in range(300):
        scene.step()

    # Switch to pure torque control, but keep a small damping for stability
    genesis_go2.robot.set_dofs_kp(np.zeros(12), np.arange(6, 18))
    genesis_go2.robot.set_dofs_kv(np.ones(12) * 0.5, np.arange(6, 18))

    genesis_go2.update_pin_with_genesis(go2)

    x_vel_des_body = 0.8
    y_vel_des_body = 0.0
    # Start z target from actual CoM height after settling, not hardcoded.
    # MPC clamps pos_des_world to within 0.1m of current position each call, so
    # this just avoids a large initial error causing the MPC to "pull" the robot down.
    z_pos_des_body = float(go2.pos_com_world[2])
    print(f"Initial CoM height: {z_pos_des_body:.3f} m")
    yaw_rate_des_body = 0.0

    traj.generate_traj(
        go2, gait, 0.0,
        x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body,
        time_step=MPC_DT
    )
    # After first traj generation, lock pos_des_world to actual CoM position so
    # subsequent calls will only drift 0.1m/step toward target
    traj.pos_des_world = go2.pos_com_world.copy()
    traj.pos_des_world[2] = z_pos_des_body
    mpc = CentroidalMPC(go2, traj)

    U_opt = np.zeros((12, traj.N), dtype=float)
    tau_cmd = np.zeros(12)
    mpc_force_world = np.zeros(12)

    ctrl_i = 0
    sim_steps = int(RUN_SIM_LENGTH_S * SIM_HZ)

    print("Starting MPC loop...")
    for k in range(sim_steps):
        time_now_s = k * SIM_DT

        if k % CTRL_DECIM == 0:
            genesis_go2.update_pin_with_genesis(go2)

            if ctrl_i % STEPS_PER_MPC == 0:
                traj.generate_traj(
                    go2, gait, time_now_s,
                    x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body,
                    time_step=MPC_DT
                )
                sol = mpc.solve_QP(go2, traj, False)
                w_opt = sol["x"].full().flatten()
                U_opt = w_opt[12 * traj.N :].reshape((12, traj.N), order="F")

            mpc_force_world = U_opt[:, 0]

            # Use real contact detection from Genesis instead of pure gait timing
            real_contact = genesis_go2.get_contact_mask()  # [FL, FR, RL, RR] bool

            for leg in ["FL", "FR", "RL", "RR"]:
                leg_out = leg_controller.compute_leg_torque(
                    leg, go2, gait, mpc_force_world[LEG_SLICE[leg]],
                    time_now_s, real_contact
                )
                tau_cmd[LEG_SLICE[leg]] = leg_out.tau

            tau_cmd = np.clip(tau_cmd, -TAU_LIM, TAU_LIM)
            ctrl_i += 1

        genesis_go2.set_joint_torque(tau_cmd)
        scene.step()
        
        if k % 100 == 0:
            pos = genesis_go2.robot.get_pos().cpu().numpy()
            print(f"Step {k}: z={pos[2]:.3f}, tau_FL={tau_cmd[0:3]}")

if __name__ == "__main__":
    main()
