# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for the TITA quadruped robot using Isaac Gym simulation. The project implements NP3O (Near Proximal Policy Optimization with Constraints) algorithm with Barlow Twins self-supervised learning for locomotion control. The trained policies can be deployed through sim2sim (Webots) and sim2real pipelines.

## Development Environment Setup

**Conda Environment:**
```bash
conda activate tita2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your_anaconda_path>/anaconda3/envs/tita2/lib
```

**Required Dependencies:**
- Python 3.8
- PyTorch 1.10.0+cu113
- Isaac Gym Preview 4
- TensorRT 8.6.0 (for deployment)
- CUDA 12.5

## Common Commands

### Training
```bash
# Navigate to training directory
cd ~/桌面/tita/tita_rl

# Train with GUI (requires high VRAM)
python train.py --task=tita_constraint

# Train headless (recommended for RTX 3060)
python train.py --task=tita_constraint --headless
```

### Model Export and Testing
```bash
# Test trained model in Isaac Gym
python simple_play.py --task=tita_constraint

# Export PyTorch model to ONNX
python export_direct.py \
    --pt_path model_11700.pt \
    --actor_class ActorCriticBarlowTwins \
    --obs_size 586 \
    --priv_obs_size 67 \
    --action_size 8 \
    --num_priv_latent 36 \
    --num_hist 10 \
    --num_prop 33 \
    --num_scan 187 \
    --activation elu

# Convert ONNX to TensorRT engine (inside Docker)
/usr/src/tensorrt/bin/trtexec --onnx=policy.onnx --saveEngine=model_gn.engine
```

### Deployment (Webots Sim2Sim)
```bash
# Start Docker container with proper mounts
sudo docker run -v ~/桌面/tita:/mnt/dev -w /mnt/dev --rm --gpus all --net=host --privileged \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -it registry.cn-guangzhou.aliyuncs.com/ddt_robot/ubuntu:webot2023b-v1

# Allow X11 access (run outside container)
xhost +local:root

# Launch Webots simulation (inside container)
cd tita_rl_sim2sim2real
sudo mkdir -p /usr/share/robot_description
sudo cp -r src/tita_locomotion/tita_description/tita /usr/share/robot_description/
source /opt/ros/humble/setup.bash && source install/setup.bash && \
  ros2 launch locomotion_bringup sim_bringup.launch.py

# Control with keyboard (new terminal, inside container)
docker exec -it amazing_booth /bin/bash
cd tita_rl_sim2sim2real
source /opt/ros/humble/setup.bash && source install/setup.bash && \
  ros2 run keyboard_controller keyboard_controller_node --ros-args -r __ns:=/tita
```

## Code Architecture

### Directory Structure
```
tita_rl/
├── configs/              # Configuration files
│   ├── base_config.py           # Base configuration class with recursive initialization
│   ├── legged_robot_config.py   # Generic legged robot configuration
│   └── tita_constraint_config.py # TITA-specific config (num_envs=2048, 8 joints)
├── envs/                 # Environment implementations
│   ├── base_task.py             # Base task interface
│   ├── legged_robot.py          # Main environment (Isaac Gym integration)
│   └── vec_env.py               # Vectorized environment wrapper
├── algorithm/            # RL algorithms
│   ├── ppo.py                   # Standard PPO implementation
│   └── np3o.py                  # NP3O with constraint handling and imitation loss
├── modules/              # Neural network modules
│   ├── actor_critic.py          # Base actor-critic architecture
│   └── depth_backbone.py        # Depth encoder (if using camera)
├── runner/               # Training orchestration
│   ├── on_constraint_policy_runner.py # Main training loop for NP3O
│   └── rollout_storage.py       # Experience buffer with cost tracking
├── utils/                # Utility functions
│   ├── terrain.py               # Terrain generation (trimesh/heightfield)
│   ├── helpers.py               # Helper functions
│   └── task_registry.py         # Task registration system
└── resources/tita/urdf/  # Robot URDF files
    └── tita_description.urdf    # TITA robot model (8 DOF bipedal)
```

### Key Components

**1. Configuration System (`configs/`)**
- Uses recursive class instantiation via `BaseConfig`
- `TitaConstraintRoughCfg`: Defines state space (673D: 33 proprio + 187 scan + 330 history + 35 priv), action space (8D joints)
- Key parameters: `num_envs=2048`, `num_observations=673`, `num_actions=8`, `num_costs=6`
- Domain randomization: friction [0.2, 2.75], mass [-1, 3], motor strength [0.8, 1.2]

**2. Environment (`envs/legged_robot.py`)**
- Inherits from `BaseTask`, manages Isaac Gym simulation
- **State Space**:
  - Proprioception (33D): base angular velocity, projected gravity, commands, joint positions/velocities, actions
  - Lidar scans (187D): terrain height measurements
  - History (330D): 10 steps × 33 proprio observations
  - Privileged info (35D): base linear velocity, contact states, randomization parameters (mass, friction, motor strength, lag)
- **Action Space**: 8D joint angle targets with PD control (`stiffness=40`, `damping=1.0`, `action_scale=0.5`)
- **Rewards**: Tracks linear/angular velocity, penalizes torques, collision, base height deviation
- **Costs** (6 types): position limits, torque limits, velocity limits, acceleration smoothness, foot contact forces, stumble detection
- Implements terrain curriculum, command curriculum, domain randomization (lag timesteps, motor/KP/KD randomization)

**3. Algorithm (`algorithm/np3o.py`)**
- **NP3O**: Extends PPO with constraint handling via penalty method
- Loss components:
  - Surrogate loss (clipped PPO objective)
  - Cost violation loss: `Σ k_i * ReLU(cost_advantage + cost_violation - d_i)`
  - Value loss (reward + cost value functions)
  - Imitation loss (if `imi_flag=True`, uses `ActorCriticBarlowTwins`)
  - Entropy regularization
- Adaptive learning rate based on KL divergence (`desired_kl=0.01`)
- Gradient clipping (`max_grad_norm=0.01`)
- k-value scheduling: `k *= 1.0004^i` to progressively enforce constraints

**4. Training Loop (`runner/on_constraint_policy_runner.py`)**
- **Rollout Phase**: Collects `num_steps_per_env=24` steps across 2048 envs (batch size 49152)
- **Compute Phase**: Calculates returns, advantages (GAE with `γ=0.99`, `λ=0.95`), cost violations
- **Update Phase**: 5 epochs × 4 mini-batches, updates actor-critic with combined loss
- **Checkpointing**: Saves every 100 iterations, resumes from `tita_example_10000.pt`
- **Imitation Learning**: If resuming, applies decaying imitation weight (linear decay over half of training)

**5. Actor-Critic (`modules/actor_critic.py`)**
- **ActorCriticBarlowTwins**: Integrates Barlow Twins self-supervised loss for robust feature learning
- Network architecture:
  - Scan encoder: [128, 64, 32]
  - Actor: [512, 256, 128] with ELU activation
  - Critic (reward): [512, 256, 128]
  - Cost critic: 6 separate heads for each cost type
- Outputs: action mean/std (Gaussian policy), value, cost values, log probabilities

### Critical Implementation Details

**Joint Reindexing:**
- `reindex(tensor)`: Maps actions from [0,1,2,3,4,5,6,7] to [4,5,6,7,0,1,2,3] for sim2real compatibility (tita_rl/envs/legged_robot.py:290, 304)
- Always use `reindex()` for actions and joint states in observations

**PD Control:**
- Special handling for joints 3 and 7 (likely passive wheels): `torques[:,[3, 7]] = kp * 10 * target - 0.5 * kd * vel` (line 693)
- Standard joints: `torques = kp * kp_factor * (target - pos) - kd * kd_factor * vel`

**Lag Timesteps:**
- When `randomize_lag_timesteps=True`, actions are delayed by 0-2 steps (`lag_timesteps=3`) using circular buffer (line 672-676)
- Randomized per environment for sim2real robustness

**Cost Constraint Enforcement:**
- 6 cost types map to `d_values` (all 0.0, meaning zero violation tolerance)
- Penalty coefficients: `cost_value_loss_coef=0.1`, `cost_viol_loss_coef=0.1`
- Monitor `mean_viol_loss` during training - if high, increase `cost_viol_loss_coef`

**Observation Normalization:**
- Scaling factors: `lin_vel`, `ang_vel`, `dof_pos`, `dof_vel`, `height_measurements`
- Noise injection during training (disabled via `cfg.noise.add_noise`)
- Clipping: `clip_observations` and `clip_actions` prevent extreme values

## Training Considerations

**Performance:**
- RTX 3060 (12GB VRAM): Use `--headless`, expect ~1000 steps/s with 2048 envs
- Training duration: ~30000 iterations (~15-20 hours on RTX 3060)
- Memory-constrained: Reduce `num_envs` to 1024 or 512 in `tita_constraint_config.py:36`

**Convergence:**
- Monitor `mean_reward`, `mean_viol_loss`, `mean_imitation_loss` in logs
- Successful training: `mean_reward > 5`, `mean_viol_loss < 1`
- If constraints not satisfied: increase `cost_viol_loss_coef` or `k_value` initial value

**Debugging:**
- Use `simple_play.py` to visualize policy behavior
- Check `logs/<task>/` for TensorBoard files
- If robot falls immediately: verify URDF path, check `default_joint_angles` in config

## Sim2Real Pipeline

**Workflow:**
1. Train in Isaac Gym (`tita_rl/`)
2. Export to ONNX (`export_direct.py`)
3. Convert to TensorRT engine (inside Docker)
4. Test in Webots sim2sim (`tita_rl_sim2sim2real/`)
5. Deploy to real TITA robot (requires ROS2 Humble, TensorRT on Jetson Orin NX)

**Key Files for Deployment:**
- `model_gn.engine`: TensorRT inference engine
- URDF must match: `/usr/share/robot_description/tita/` (copied in launch script)
- Update engine path in `tita_rl_sim2sim2real` launch files before running

## Important Notes

- **Do not modify** `reindex()` logic without understanding sim2real joint mapping
- **Always use headless mode** on low VRAM GPUs to avoid crashes
- **Check Isaac Gym installation** if `ImportError: libpython3.8.so.1.0` occurs (set `LD_LIBRARY_PATH`)
- **URDF path** uses `{ROOT_DIR}` placeholder, resolved by `global_config.py`
- **Task registration** must happen before `train()` call (see `train.py:35`)
- Training checkpoints include optimizer state - do not mix checkpoints from different runs

## Reference

Based on [N3PO Locomotion](https://github.com/zeonsunlightyu/LocomotionWithNP3O.git) by Nikita Rudin (ETH Zurich).
