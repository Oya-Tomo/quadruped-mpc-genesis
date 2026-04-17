# quadruped-mpc-genesis

This repository provides a standalone convex-optimization-based MPC implementation
for Go2 locomotion in Genesis.

The controller optimizes ground reaction forces using centroidal MPC and combines
them with Cartesian swing-leg control (PD + feedforward), enabling quadruped walking
on flat terrain.

## Features

- Integrated Genesis simulation and MPC walking control
- State, Jacobian, and dynamics-term computation with Pinocchio
- QP-based centroidal MPC using CasADi + OSQP
- Standalone project structure with local assets management (no runtime dependency on the reference repository)

## Run

1. Install dependencies

```bash
uv sync
```

2. Run the simulation

```bash
uv run main.py
```

## Project Structure

- [main.py](main.py): simulation loop, control-rate scheduling, MPC update calls
- [model.py](model.py): Genesis integration and Pinocchio robot model
- [mpc_core.py](mpc_core.py): gait logic, trajectory generation, centroidal MPC, leg torque controller
- [assets](assets): URDF/MJCF files and meshes

## Reference

This implementation is inspired by the theory and control flow of:

- https://github.com/elijah-waichong-chan/go2-convex-mpc

The method is restructured to run in Genesis and implemented so this repository can run independently.
The robot assets (URDF/MJCF and related meshes) were also referenced from the same repository and copied into this project for standalone use.
