# CleanSweep — The Autonomous Learner Bot

**Jeremy Galea** University of Malta — Faculty of ICT, Department of Artificial Intelligence

CleanSweep is an autonomous floor-coverage robot that learns to sweep an entire
room while avoiding obstacles it has never seen before. Coverage policies are
trained in a lightweight 2D grid simulator using two reinforcement-learning
paradigms — a tabular **Q-Learning** agent that plans at the level of room
*strips*, and a **PPO** agent that plans cell-by-cell over the occupancy grid —
and the same trained policies are then deployed unchanged in a **Gazebo**
simulation and on a **physical differential-drive robot** (Arduino + Raspberry
Pi + ROS 2). Localisation on hardware fuses wheel/IMU dead-reckoning with ArUco
visual fixes through an Extended Kalman Filter.

## Repository Structure

```
CleanSweep_JeremyGalea/
└── src/
    ├── cleansweep_firmware.ino              # Arduino firmware (motors, ultrasonics, IMU)
    │
    ├── clean_sweep_rl_v2/                   # RL training & evaluation
    │   ├── train.py                         # Q-Learning trainer (strip-level)
    │   ├── evaluate.py                      # Q-Learning evaluation + coverage maps
    │   ├── train_ppo.py                     # MaskablePPO trainer (cell-level, CNN)
    │   ├── evaluate_ppo.py                  # PPO evaluation + coverage maps
    │   ├── coverage_env/
    │   │   ├── room.py                      # Turns SDF world into occupancy grid loader
    │   │   ├── object_spawner.py            # random rectangular obstacle placement
    │   │   └── coverage_env.py              # strip/boustrophedon env (Q-Learning)
    │   ├── q_learning/
    │   │   └── q_agent.py                   # tabular ε-greedy Q-Learning agent
    │   ├── ppo_training/
    │   │   ├── cell_env.py                  # 4-channel grid Gymnasium env (PPO)
    │   │   └── feature_extractor.py         # GridCNN feature extractor
    │   ├── maps/                            # SDF rooms + cached wall_grid.npy
    │   ├── results_20cm_cells/              # trained Q-table + logs + curves (env 1)
    │   ├── ql_env2_corridor/                # trained Q-table (env 2)
    │   ├── results_ppo_3/                   # trained PPO model + logs + curves (env 1)
    │   └── ppo_env2_corridor_v3/            # trained PPO model (env 2)
    │
    ├── clean_sweep_description/             # ROS 2 robot model (URDF / Xacro)
    │   ├── urdf/                            # base, wheels, camera, ultrasonics, IMU
    │   ├── config/                          # ros2_control controllers + EKF config
    │   ├── rviz/                            # RViz display config
    │   └── launch/display.launch.xml
    │
    ├── clean_sweep_bringup/                 # ROS 2 launch + nodes (sim & hardware)
    │   ├── scripts/                         # serial bridge, odometry, ArUco, executors, visualisers
    │   ├── launch/                          # gazebo / hardware / coverage launch files
    │   ├── worlds/                          # Gazebo SDF worlds (bedroom / corridor)
    │   └── config/gazebo_bridge.yaml
    │
    └── results_tests/                       # recorded sim + real-robot run outputs
        ├── env1/  (Q_Learning, PPO)         # bedroom environment
        └── env2/  (Q_Learning, PPO)         # corridor environment
```

## Directory Overview

### Firmware

- **cleansweep_firmware.ino** — The only code that runs on the Arduino. It
  receives velocity commands from the Raspberry Pi over serial
  (`V<linear>,<angular>`), drives a dual-channel differential drive
  (Pulse-Width Modulation + direction per side with a left-side trim and a minimum
  PWM floor), and streams sensor state back at 20 Hz
  (`S<dist_front>,<dist_rear>,<ax,ay,az>,<gx,gy,gz>`) from two HC-SR04
  ultrasonic sensors and an MPU-6050 IMU. Includes a command-timeout safety stop.

### RL Training & Evaluation — `clean_sweep_rl_v2`

A self-contained 2D grid simulator and two RL agents. Both share a 0.20 m
occupancy grid built from a Gazebo SDF world, and both are trained against
**randomly placed, unknown obstacles** (14 rectangular objects of 20–60 cm per
episode) so the learned policies generalise to clutter that was not present at
design time.

- **coverage_env/room.py** — Parses an SDF world's static box models into a
  binary occupancy grid at a configurable cell size, prunes everything except
  the largest connected free region, and provides world-to-cell coordinate
  conversions used by both the simulator and the hardware executors.
- **coverage_env/object_spawner.py** — Places random rectangular obstacles by
  rejection sampling, enforcing a wall/object clearance margin and rejecting any
  placement that would disconnect the free space (connectivity check), so full
  coverage always remains physically possible.
- **coverage_env/coverage_env.py** — The **Q-Learning** environment. The room is
  divided into horizontal strips; each action selects the next *uncovered strip*
  to sweep, and the strip is then swept boustrophedon-style with BFS navigation
  between cells. State = `(current_strip, entry_side, visited-strip bitmask,
  front/left/right obstacle sensors)`. Reward = +1 per newly discovered cell,
  −0.05 per move, +50 on full coverage.
- **q_learning/q_agent.py** — Tabular ε-greedy Q-Learning agent backed by a
  sparse Q-table, with action masking over valid (uncovered) strips, linear
  ε-decay, and pickle save/load for checkpointing and deployment.
- **train.py / evaluate.py** — Train the Q-Learning agent under random obstacle
  layouts and evaluate the greedy policy, emitting per-episode CSV logs, training
  curves, and annotated coverage maps (path, visit-count heat, overlap stats).
- **ppo_training/cell_env.py** — The **PPO** environment (Gymnasium). The agent
  moves one cell at a time (UP/RIGHT/DOWN/LEFT) and observes a 4-channel grid:
  discovered obstacles, visited cells, robot position, and a BFS frontier-distance
  field. Action masking forbids moves into walls or known obstacles. Reward = +1
  per new cell, −0.3 per revisit/blocked move, −0.02 per step, +50 on full
  coverage.
- **ppo_training/feature_extractor.py** — `GridCNN`, a convolutional feature
  extractor over the grid observation feeding Stable-Baselines3 / sb3-contrib's
  `MaskablePPO`.
- **train_ppo.py / evaluate_ppo.py** — Train the cell-level PPO agent
  and evaluate the policy with a BFS loop-escape assist that takes
  over when the agent gets stuck. Outputs the same logs, curves, and coverage maps
  as the Q-Learning pipeline.
- **maps/** — SDF world files (bedroom and corridor environments) and a cached
  `wall_grid.npy` occupancy grid.
- **results_20cm_cells/, ql_env2_corridor/, results_ppo_3/, ppo_env2_corridor_v3/**
  — Trained artefacts per agent and environment: the Q-table
  (`q_table_final.pkl`) or PPO model (`ppo_model`, `policy_weights.pt`), the
  per-episode training log (`training_log.csv`), training curves
  (`training_curves.png`), and TensorBoard event files.

### Robot Description — `clean_sweep_description`

The ROS 2 robot model and its control/estimation configuration.

- **urdf/** — Xacro description of the differential-drive mobile base, wheels,
  forward-facing camera, front and rear ultrasonic sensors, and the MPU-6050 IMU,
  assembled in `clean_sweep.urdf.xacro`.
- **config/controllers.yaml** — `ros2_control` joint-state broadcaster and
  differential-drive controller parameters used in simulation.
- **config/ekf.yaml** — `robot_localization` EKF configuration. Fuses three
  sources into `/odometry/filtered`: dead-reckoning odometry (`/odom`, x/y as
  differential + yaw rate), the IMU (`/imu`, gyro yaw rate only), and absolute
  ArUco pose fixes (`/aruco/pose`, x/y/yaw).
- **rviz/**, **launch/display.launch.xml** — RViz configuration and a standalone
  model-display launch file.

### Bringup — `clean_sweep_bringup`

ROS 2 nodes and launch files that run the system in both Gazebo and on hardware.
The hardware localisation stack is layered: dead-reckoning odometry (Layer 1),
IMU heading, and ArUco visual fixes (Layer 3). They are fused by the EKF.

**Nodes (`scripts/`)**

- **serial_bridge_node.py** — Bridges the Arduino over USB serial: forwards
  `/cmd_vel` as `V` commands and parses the `S` sensor stream into ROS
  `sensor_msgs/Imu` and ultrasonic range messages.
- **cmd_vel_odom_node.py** — Layer 1 dead-reckoning odometry, integrating
  commanded `/cmd_vel` with the IMU yaw to publish `/odom`.
- **aruco_localizer_node.py** — Layer 3 localisation. Detects 4×4 ArUco markers
  in the camera image, solves the camera-to-marker transform, and publishes an
  absolute robot pose on `/aruco/pose` against a known marker map.
- **coverage_monitor_hw.py** — Tracks which grid cells have been visited from the
  fused pose, publishes a live coverage `OccupancyGrid`, and prints a run summary
  (coverage %, unique vs. revisited cells, overlap).
- **coverage_executor_gazebo.py** — Drives the **Q-Learning** policy: converts the
  strip plan into cell waypoints and follows them with `/cmd_vel`, using BFS
  routing and reactive obstacle handling from live sensor data.
- **coverage_executor_gazebo_ppo.py** — Drives the **PPO** policy cell-by-cell
  into `/cmd_vel`, with the same BFS loop-escape assist used in evaluation.
- **ground_truth_publisher.py** — In simulation, streams Gazebo's dynamic pose as
  `/ground_truth_pose` (and `/odometry/filtered`) so the simulation runs match the hardware
  topic layout.
- **coverage_map_visualiser.py / coverage_heatmap_visualiser.py** — Live
  Matplotlib coverage map and visit-count heatmap from odometry + range data.
- **ekf_particle_visualiser.py** — Visualises the fused EKF pose.

**Launch files (`launch/`)**

- **clean_sweep_gazebo.launch.xml** — Gazebo simulation with the diff-drive
  controllers, ROS-Gazebo bridge, and RViz.
- **clean_sweep_gazebo_with_ekf.launch.xml** — Simulation variant that also runs
  the ground-truth publisher.
- **clean_sweep_coverage.launch.xml / clean_sweep_coverage_ppo.launch.xml** — Run
  the Q-Learning or PPO coverage executor with a trained model.
- **clean_sweep_hw.launch.xml / clean_sweep_hw_ppo.launch.xml** — The full
  hardware stack: serial bridge, dead-reckoning odometry, USB camera, ArUco
  localiser, EKF, coverage monitor, and the Q-Learning or PPO coverage executor.

### Results — `results_tests`

Recorded outputs from simulation and real-robot test runs, organised by
environment (`env1` = bedroom, `env2` = corridor) and agent (`Q_Learning`,
`PPO`). Each test folder contains coverage-map and heatmap renders for the
simulator (`sim.png`, `sim_hm.png`) and, where applicable, the real robot
(`real.png`, `real_hm.png`, `particle.png`), together with a `results.txt` run
summary reporting coverage %, unique vs. revisited cells, overlap, and elapsed
time. The `2d env/` subfolders hold per-episode evaluation renders from the grid
simulator.
