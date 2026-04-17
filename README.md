# quadruped-mpc-genesis

This repository provides a standalone convex-optimization-based MPC implementation
for Go2 locomotion in Genesis.

The controller optimizes ground reaction forces using centroidal MPC and combines
them with Cartesian swing-leg control (PD + feedforward), enabling quadruped walking
on flat terrain.

## Features

- **Genesis-Native Contact Detection**: Uses `get_links_net_contact_force()` for robust ground-truth contact estimation.
- **Precision State Sync**: Seamless state synchronization between Genesis simulation and Pinocchio kinematics/dynamics.
- **Centroidal MPC**: Convex-optimization-based control using CasADi + OSQP to optimize ground reaction forces.
- **Swing-Leg Control**: Cartesian PD control with feedforward torques for precise foot placement.
- **Dynamic Touchdown**: Accurate Raibert heuristic calculations using real-time link positions from Genesis.

## Run

1. Install dependencies

```bash
uv sync
```

2. Run the simulation

```bash
uv run src/main.py
```

## Project Structure

- **[src/main.py](src/main.py)**: Main entry point. Handles the simulation loop, control rate scheduling, and high-level MPC update calls.
- **[src/genesis_model.py](src/genesis_model.py)**: Genesis-Pinocchio interface. Manages robot entity creation, state extraction, and torque application.
- **[src/mpc/](src/mpc/)**: Core MPC and controller implementation.
    - **[centroidal_mpc.py](src/mpc/centroidal_mpc.py)**: QP formulation and solver logic for ground reaction force optimization.
    - **[com_trajectory.py](src/mpc/com_trajectory.py)**: CoM trajectory generation and Raibert-based foot placement planning.
    - **[leg_controller.py](src/mpc/leg_controller.py)**: Cartesian leg controller (PD + Feedforward) for swing and stance.
    - **[gait.py](src/mpc/gait.py)**: Periodic gait schedule and phase management.
    - **[go2_robot_data.py](src/mpc/go2_robot_data.py)**: Pinocchio-based robot model data and kinematic utilities.
- **[assets/](assets/)**: Local MJCF/URDF models and meshes for the Unitree Go2 robot.

## Reference

This implementation is inspired by the theory and control flow of:

- https://github.com/elijah-waichong-chan/go2-convex-mpc

The method is restructured to run in Genesis and implemented so this repository can run independently.
The robot assets (URDF/MJCF and related meshes) were also referenced from the same repository and copied into this project for standalone use.
