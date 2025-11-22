# TITA RL 仿真与训练环境架构说明及深度相机跑酷开发指南

## 📋 目录
- [1. 项目整体架构](#1-项目整体架构)
- [2. 仿真环境搭建详解](#2-仿真环境搭建详解)
- [3. 训练环境架构](#3-训练环境架构)
- [4. 深度相机系统详解](#4-深度相机系统详解)
- [5. 添加跑酷功能实施指南](#5-添加跑酷功能实施指南)
- [6. 具体实现步骤](#6-具体实现步骤)
- [7. 调试与优化建议](#7-调试与优化建议)

---

## 1. 项目整体架构

### 1.1 目录结构说明

```
tita_rl/
├── configs/                    # 配置文件目录
│   ├── base_config.py         # 基础配置类
│   ├── legged_robot_config.py # 腿式机器人基础配置
│   └── tita_constraint_config.py # TITA 特定配置
├── envs/                       # 环境定义
│   ├── base_task.py           # 基础任务类
│   ├── legged_robot.py        # 主要环境实现
│   └── vec_env.py             # 向量化环境
├── modules/                    # 神经网络模块
│   ├── actor_critic.py        # Actor-Critic 网络
│   ├── depth_backbone.py      # 深度图像处理网络
│   └── common_modules.py      # 通用模块
├── algorithm/                  # 强化学习算法
│   ├── ppo.py                 # PPO 算法
│   └── np3o.py                # NP3O 算法（带约束）
├── runner/                     # 训练运行器
├── utils/                      # 工具函数
│   ├── terrain.py             # 地形生成
│   └── helpers.py             # 辅助函数
└── train.py                   # 训练入口
```

### 1.2 核心技术栈

- **仿真引擎**: NVIDIA Isaac Gym (基于 PhysX)
- **深度学习**: PyTorch
- **强化学习**: PPO / NP3O (带约束的策略优化)
- **视觉处理**: TorchVision, OpenCV
- **机器人模型**: URDF 格式

---

## 2. 仿真环境搭建详解

### 2.1 Isaac Gym 环境初始化流程

仿真环境的创建在 `legged_robot.py` 的 `LeggedRobot` 类中实现：

```python
# 初始化流程
1. 解析配置 (_parse_cfg)
2. 创建仿真实例 (create_sim)
3. 生成地形 (_create_terrain)
4. 创建环境实例 (_create_envs)
5. 初始化缓冲区 (_init_buffers)
```

### 2.2 关键组件说明

#### 2.2.1 地形系统
```python
class terrain:
    mesh_type = 'trimesh'           # 地形类型：trimesh/heightfield/plane
    horizontal_scale = 0.1          # 水平分辨率 [m]
    vertical_scale = 0.005          # 垂直分辨率 [m]
    curriculum = True               # 课程学习开关
    
    # 地形类型比例
    terrain_proportions = [
        0.1,  # 平滑斜坡
        0.1,  # 粗糙斜坡
        0.35, # 上楼梯
        0.25, # 下楼梯
        0.2   # 离散障碍
    ]
```

#### 2.2.2 物理引擎配置
```python
class physx:
    num_threads = 10
    solver_type = 1                 # 1: TGS (推荐), 0: PGS
    num_position_iterations = 4     # 位置迭代次数
    num_velocity_iterations = 0     # 速度迭代次数
    contact_offset = 0.01           # 接触偏移 [m]
    max_gpu_contact_pairs = 2**23   # GPU 最大接触对数
```

### 2.3 机器人模型加载

```python
# 在 _create_envs 方法中
robot_asset = self.gym.load_asset(
    self.sim,
    asset_root,
    asset_file,
    asset_options
)

# TITA 机器人默认关节角度（站立姿态）
default_joint_angles = {
    'joint_left_leg_1': 0,
    'joint_right_leg_1': 0,
    'joint_left_leg_2': 0.8,      # 髋关节
    'joint_right_leg_2': 0.8,
    'joint_left_leg_3': -1.5,     # 膝关节
    'joint_right_leg_3': -1.5,
    'joint_left_leg_4': 0,        # 踝关节
    'joint_right_leg_4': 0,
}
```

---

## 3. 训练环境架构

### 3.1 强化学习核心循环

```
训练循环 (train.py)
    ↓
环境交互 (LeggedRobot.step)
    ↓
├─ 执行动作
├─ 物理仿真 (Isaac Gym)
├─ 计算观测
├─ 计算奖励
└─ 判断终止
    ↓
策略网络更新 (PPO/NP3O)
    ↓
Actor-Critic 网络
```

### 3.2 观测空间设计

当前观测包含（`compute_observations` 方法）：

```python
obs_buf = torch.cat((
    self.base_ang_vel * self.obs_scales.ang_vel,        # 基座角速度 [3]
    self.projected_gravity,                             # 投影重力 [3]
    self.commands[:, :3] * self.commands_scale,         # 速度指令 [3]
    self.dof_pos * self.obs_scales.dof_pos,            # 关节位置 [8]
    self.dof_vel * self.obs_scales.dof_vel,            # 关节速度 [8]
    self.action_history_buf[:,-1]                       # 历史动作 [8]
), dim=-1)

# 总维度: 3 + 3 + 3 + 8 + 8 + 8 = 33 (n_proprio)
```

### 3.3 奖励函数设计

在 `configs/tita_constraint_config.py` 中定义：

```python
class rewards.scales:
    tracking_lin_vel = 1.0      # 跟踪线速度（主要任务）
    tracking_ang_vel = 0.5      # 跟踪角速度
    lin_vel_z = -0.0            # 惩罚垂直速度
    ang_vel_xy = -0.05          # 惩罚横滚/俯仰角速度
    orientation = -1.0          # 惩罚姿态偏差
    torques = 0.0               # 惩罚力矩
    powers = -2e-5              # 惩罚功率消耗
    dof_acc = -2.5e-7           # 惩罚关节加速度
    base_height = -1.0          # 惩罚高度偏差
    action_rate = -0.01         # 惩罚动作变化率
    collision = -1.0            # 惩罚碰撞
    termination = -200          # 终止惩罚
```

### 3.4 域随机化 (Domain Randomization)

用于提高 sim-to-real 迁移能力：

```python
class domain_rand:
    randomize_friction = True           # 摩擦系数随机化
    friction_range = [0.2, 2.75]
    
    randomize_base_mass = True          # 质量随机化
    added_mass_range = [-1., 3.]
    
    randomize_base_com = True           # 质心随机化
    added_com_range = [-0.1, 0.1]
    
    randomize_motor = True              # 电机强度随机化
    motor_strength_range = [0.8, 1.2]
    
    randomize_lag_timesteps = True      # 延迟随机化
    lag_timesteps = 3
    
    push_robots = True                  # 随机推力
    push_interval_s = 15
    max_push_vel_xy = 1
```

---

## 4. 深度相机系统详解

### 4.1 深度相机配置

在 `legged_robot_config.py` 中的 `depth` 类：

```python
class depth:
    use_camera = False              # 是否启用相机
    camera_num_envs = 192          # 使用相机的环境数量
    
    # 相机安装位置（相对机器人基座）
    position = [0.27, 0, 0.03]     # [前, 左, 上] 米
    angle = [-5, 5]                # 俯仰角范围 [度]
    
    # 图像参数
    original = (106, 60)           # 原始分辨率 (width, height)
    resized = (87, 58)             # 缩放后分辨率
    horizontal_fov = 87            # 水平视场角 [度]
    
    # 深度范围
    near_clip = 0                  # 近裁剪面 [米]
    far_clip = 2                   # 远裁剪面 [米]
    dis_noise = 0.0                # 深度噪声
    
    # 更新频率
    update_interval = 5            # 每 N 步更新一次
    buffer_len = 2                 # 历史帧缓冲长度
```

### 4.2 相机创建与附着

在 `attach_camera` 方法中：

```python
def attach_camera(self, i, env_handle, actor_handle):
    if self.cfg.depth.use_camera:
        # 1. 创建相机属性
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.depth.original[0]
        camera_props.height = self.cfg.depth.original[1]
        camera_props.enable_tensors = True
        camera_props.horizontal_fov = self.cfg.depth.horizontal_fov
        
        # 2. 创建相机传感器
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        
        # 3. 设置相机位姿（相对机器人基座）
        local_transform = gymapi.Transform()
        camera_position = np.copy(config.position)
        camera_angle = np.random.uniform(config.angle[0], config.angle[1])
        
        local_transform.p = gymapi.Vec3(*camera_position)
        local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
        
        # 4. 附着到机器人基座
        root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
        self.gym.attach_camera_to_body(
            camera_handle, env_handle, root_handle, 
            local_transform, gymapi.FOLLOW_TRANSFORM
        )
```

### 4.3 深度图像处理流程

```python
def update_depth_buffer(self):
    """更新所有环境的深度缓冲区"""
    if not self.cfg.depth.use_camera:
        return
    
    # 每 N 步更新一次
    if self.global_counter % self.cfg.depth.update_interval != 0:
        return
    
    # 渲染所有相机
    self.gym.step_graphics(self.sim)
    self.gym.render_all_camera_sensors(self.sim)
    self.gym.start_access_image_tensors(self.sim)
    
    for i in range(self.num_envs):
        # 获取深度图像
        depth_image_ = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[i], self.cam_handles[i],
            gymapi.IMAGE_DEPTH
        )
        depth_image = gymtorch.wrap_tensor(depth_image_)
        
        # 处理深度图像
        depth_image = self.process_depth_image(depth_image, i)
        
        # 更新缓冲区（维护历史帧）
        if self.episode_length_buf[i] <= 1:
            self.depth_buffer[i] = torch.stack(
                [depth_image] * self.cfg.depth.buffer_len, dim=0
            )
        else:
            self.depth_buffer[i] = torch.cat([
                self.depth_buffer[i, 1:], 
                depth_image.unsqueeze(0)
            ], dim=0)
    
    self.gym.end_access_image_tensors(self.sim)

def process_depth_image(self, depth_image, env_id):
    """处理单张深度图像"""
    # 1. 裁剪边缘
    depth_image = self.crop_depth_image(depth_image)
    
    # 2. 添加噪声
    depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
    
    # 3. 裁剪深度范围
    depth_image = torch.clip(
        depth_image, 
        -self.cfg.depth.far_clip, 
        -self.cfg.depth.near_clip
    )
    
    # 4. 调整尺寸
    depth_image = self.resize_transform(depth_image[None, :]).squeeze()
    
    # 5. 归一化
    depth_image = self.normalize_depth_image(depth_image)
    
    return depth_image
```

### 4.4 深度编码器网络

在 `modules/depth_backbone.py` 中定义了多种深度编码器：

#### 4.4.1 DepthOnlyFCBackbone (CNN 编码器)
```python
class DepthOnlyFCBackbone58x87(nn.Module):
    """
    输入: [batch, 1, 58, 87] 深度图像
    输出: [batch, 32] 特征向量
    """
    def __init__(self, ...):
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            nn.ELU(),
            nn.Linear(128, 32)  # 输出 32 维特征
        )
```

#### 4.4.2 RecurrentDepthBackbone (RNN 编码器)
```python
class RecurrentDepthBackbone(nn.Module):
    """
    融合深度特征和本体感受特征，使用 GRU 处理时序信息
    """
    def __init__(self, base_backbone, env_cfg):
        self.base_backbone = base_backbone  # CNN 提取空间特征
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            nn.ELU(),
            nn.Linear(128, 32)
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 34),
            nn.Tanh()
        )
```

---

## 5. 添加跑酷功能实施指南

### 5.1 跑酷功能需求分析

跑酷能力需要机器人能够：
1. **感知障碍物**：通过深度相机识别前方障碍
2. **规划跳跃时机**：判断何时需要跳跃
3. **执行跳跃动作**：产生足够的垂直推力
4. **稳定着陆**：控制着陆姿态和冲击力

### 5.2 核心修改点

```
1. 配置文件修改
   ├─ 启用深度相机
   ├─ 调整地形生成（增加障碍物）
   └─ 设计跑酷奖励函数

2. 环境代码修改
   ├─ 添加障碍物检测逻辑
   ├─ 添加跳跃触发机制
   └─ 添加跑酷相关奖励计算

3. 神经网络修改
   ├─ 整合深度特征到策略网络
   └─ 增加跳跃动作输出（可选）

4. 训练策略调整
   ├─ 课程学习（从简单到复杂）
   └─ 调整超参数
```

---

## 6. 具体实现步骤

### 步骤 1: 启用深度相机

修改 `configs/tita_constraint_config.py`：

```python
class depth(LeggedRobotCfg.depth):
    use_camera = True              # ✅ 启用相机
    camera_num_envs = 192          # 使用相机的环境数量
    
    position = [0.27, 0, 0.05]     # 相机位置（稍微抬高）
    angle = [-10, 0]               # 俯仰角（向下看）
    
    update_interval = 1            # 每步更新（跑酷需要更高频率）
    
    original = (106, 60)
    resized = (87, 58)
    horizontal_fov = 87
    buffer_len = 3                 # 增加历史帧（更好的时序信息）
    
    near_clip = 0.1                # 近裁剪面
    far_clip = 3.0                 # 远裁剪面（看得更远）
    dis_noise = 0.01               # 适度噪声
```

### 步骤 2: 设计障碍物地形

修改 `configs/tita_constraint_config.py` 的地形配置：

```python
class terrain(LeggedRobotCfg.terrain):
    mesh_type = 'trimesh'
    curriculum = True
    
    # 调整地形比例，增加障碍物
    terrain_proportions = [
        0.05,  # 平滑斜坡
        0.05,  # 粗糙斜坡
        0.25,  # 上楼梯（需要跳跃）
        0.20,  # 下楼梯
        0.45   # 离散障碍（跑酷重点）⬆️
    ]
    
    # 增加地形难度
    terrain_length = 8.
    terrain_width = 8.
    num_rows = 15              # 增加难度级别
    max_init_terrain_level = 3 # 从中等难度开始
```

### 步骤 3: 添加跑酷奖励函数

在 `envs/legged_robot.py` 中添加新的奖励函数：

```python
# 添加到 _prepare_reward_function 方法中

# 1. 障碍物清除奖励
def _reward_obstacle_clearance(self):
    """奖励足部离地高度（鼓励跳跃）"""
    # 计算足部高度
    foot_heights = self.foot_positions[:, :, 2] - self.measured_heights
    
    # 奖励足部抬高（在有障碍物时）
    obstacle_detected = self._detect_obstacles_ahead()
    clearance = torch.sum(foot_heights * obstacle_detected.unsqueeze(1), dim=1)
    
    return torch.clip(clearance, 0, 0.5)  # 最高 0.5m

# 2. 障碍物检测
def _detect_obstacles_ahead(self):
    """从深度图像检测前方障碍物"""
    if not self.cfg.depth.use_camera:
        return torch.zeros(self.num_envs, device=self.device)
    
    # 分析深度图像的中央区域
    depth_img = self.depth_buffer[:, -1]  # 最新帧
    h, w = depth_img.shape[1], depth_img.shape[2]
    
    # 取中央区域（前方）
    center_region = depth_img[:, h//3:2*h//3, w//3:2*w//3]
    
    # 如果中央区域平均深度小于阈值，说明有障碍物
    avg_depth = center_region.mean(dim=(1, 2))
    obstacle_detected = (avg_depth < 0.5).float()  # 0.5m 内有障碍
    
    return obstacle_detected

# 3. 跳跃时机奖励
def _reward_jump_timing(self):
    """奖励在障碍物前跳跃"""
    obstacle_detected = self._detect_obstacles_ahead()
    feet_in_air = (self.contact_forces[:, self.feet_indices, 2].abs() < 1.0).all(dim=1).float()
    
    # 奖励在检测到障碍物时跳跃
    return obstacle_detected * feet_in_air

# 4. 着陆稳定性奖励
def _reward_landing_stability(self):
    """奖励稳定着陆"""
    # 着陆瞬间（从空中到接触地面）
    was_in_air = self.last_contacts.sum(dim=1) == 0
    is_on_ground = self.contact_forces[:, self.feet_indices, 2].abs() > 1.0
    landing = was_in_air & is_on_ground.all(dim=1)
    
    # 着陆时姿态应该平稳
    orientation_penalty = torch.abs(self.projected_gravity[:, :2]).sum(dim=1)
    stability = torch.exp(-orientation_penalty * 5)
    
    return landing.float() * stability
```

### 步骤 4: 更新奖励权重配置

在 `configs/tita_constraint_config.py` 中：

```python
class rewards(LeggedRobotCfg.rewards):
    base_height_target = 0.35
    
    class scales(LeggedRobotCfg.rewards.scales):
        # 原有奖励
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        orientation = -1.0
        base_height = -1.0
        action_rate = -0.01
        termination = -200
        
        # 新增跑酷奖励 ⭐
        obstacle_clearance = 2.0      # 障碍物清除
        jump_timing = 1.5              # 跳跃时机
        landing_stability = 1.0        # 着陆稳定性
        
        # 调整原有权重
        feet_air_time = 0.5           # 降低（跑酷需要腾空）
        lin_vel_z = -0.5              # 降低（允许垂直速度）
        collision = -5.0              # 增加（严惩碰撞）
```

### 步骤 5: 修改观测空间（添加深度特征）

在 `envs/legged_robot.py` 的 `compute_observations` 方法中：

```python
def compute_observations(self):
    # 原有本体感受观测
    proprio_obs = torch.cat((
        self.base_ang_vel * self.obs_scales.ang_vel,
        self.projected_gravity,
        self.commands[:, :3] * self.commands_scale,
        self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
        self.reindex(self.dof_vel * self.obs_scales.dof_vel),
        self.action_history_buf[:,-1]
    ), dim=-1)
    
    # 如果启用深度相机，添加深度特征
    if self.cfg.depth.use_camera and hasattr(self, 'depth_buffer'):
        # 这里可以直接使用深度图像，或提取特征
        # 选项1：使用原始深度图（需要 CNN 策略网络）
        depth_obs = self.depth_buffer[:, -1].flatten(1, 2)  # [num_envs, H*W]
        
        # 选项2：提取简单统计特征
        depth_features = self._extract_depth_features()
        
        self.obs_buf = torch.cat([proprio_obs, depth_features], dim=-1)
    else:
        self.obs_buf = proprio_obs
    
    # 添加噪声
    noise_vec = ...  # 同之前
    self.obs_buf += noise_vec * noise_level

def _extract_depth_features(self):
    """从深度图像提取简单特征"""
    depth_img = self.depth_buffer[:, -1]  # [num_envs, H, W]
    
    # 分区域统计
    h, w = depth_img.shape[1], depth_img.shape[2]
    
    # 左、中、右三个区域
    left = depth_img[:, :, :w//3]
    center = depth_img[:, :, w//3:2*w//3]
    right = depth_img[:, :, 2*w//3:]
    
    # 上、下两个区域（近、远）
    near = depth_img[:, :h//2, :]
    far = depth_img[:, h//2:, :]
    
    features = torch.cat([
        left.mean(dim=(1,2)).unsqueeze(1),
        center.mean(dim=(1,2)).unsqueeze(1),
        right.mean(dim=(1,2)).unsqueeze(1),
        near.min(dim=1)[0].min(dim=1)[0].unsqueeze(1),  # 最近障碍
        far.mean(dim=(1,2)).unsqueeze(1),
    ], dim=1)
    
    return features  # [num_envs, 5]
```

### 步骤 6: 更新配置观测维度

在 `configs/tita_constraint_config.py` 中：

```python
class env(LeggedRobotCfg.env):
    num_envs = 2048
    
    n_scan = 187
    n_priv_latent = 4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
    n_proprio = 33
    
    # 新增深度特征维度
    n_depth_features = 5  # 如果使用统计特征
    # 或者
    # n_depth_features = 87 * 58  # 如果使用原始深度图
    
    history_len = 10
    
    # 更新总观测维度
    num_observations = (
        n_proprio + 
        n_scan + 
        history_len * n_proprio + 
        n_priv_latent +
        n_depth_features  # ⭐ 新增
    )
```

### 步骤 7: 更新策略网络（支持深度输入）

修改 `configs/tita_constraint_config.py` 中的策略配置：

```python
class policy(LeggedRobotCfgPPO.policy):
    init_noise_std = 1.0
    
    # 如果使用原始深度图，需要 CNN 编码器
    use_depth_encoder = True
    depth_encoder_type = 'cnn'  # 'cnn' or 'recurrent'
    
    # 编码器配置
    scan_encoder_dims = [128, 64, 32]
    depth_encoder_dims = [64, 32, 16]  # 新增
    
    # Actor-Critic 隐藏层
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    
    activation = 'elu'
```

### 步骤 8: 课程学习策略

创建课程学习配置：

```python
# 在训练脚本中添加课程学习
class ParkourCurriculum:
    def __init__(self):
        self.stages = [
            {
                'name': 'stage_1_flat',
                'iterations': 2000,
                'terrain_level': 0,
                'obstacle_height': 0.05,
                'rewards': {
                    'tracking_lin_vel': 1.0,
                    'obstacle_clearance': 0.5,  # 低权重
                }
            },
            {
                'name': 'stage_2_small_obstacles',
                'iterations': 3000,
                'terrain_level': 3,
                'obstacle_height': 0.10,
                'rewards': {
                    'tracking_lin_vel': 1.0,
                    'obstacle_clearance': 1.0,  # 中等权重
                    'jump_timing': 0.5,
                }
            },
            {
                'name': 'stage_3_parkour',
                'iterations': 5000,
                'terrain_level': 7,
                'obstacle_height': 0.15,
                'rewards': {
                    'tracking_lin_vel': 1.0,
                    'obstacle_clearance': 2.0,  # 高权重
                    'jump_timing': 1.5,
                    'landing_stability': 1.0,
                }
            }
        ]
    
    def get_stage(self, iteration):
        cumulative = 0
        for stage in self.stages:
            cumulative += stage['iterations']
            if iteration < cumulative:
                return stage
        return self.stages[-1]  # 最后阶段
```

---

## 7. 调试与优化建议

### 7.1 训练前检查清单

- [ ] 深度相机正确附着到机器人基座
- [ ] 深度图像分辨率和更新频率合理
- [ ] 观测空间维度配置正确
- [ ] 奖励函数权重不冲突
- [ ] 地形生成包含足够障碍物
- [ ] 网络架构支持深度输入

### 7.2 可视化调试

```python
# 在 legged_robot.py 中添加可视化方法
def visualize_depth_and_obstacles(self, env_id=0):
    """可视化深度图像和障碍物检测"""
    if not self.cfg.depth.use_camera:
        return
    
    import matplotlib.pyplot as plt
    
    depth_img = self.depth_buffer[env_id, -1].cpu().numpy()
    obstacle = self._detect_obstacles_ahead()[env_id].item()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_img, cmap='viridis')
    plt.title(f'Depth Image (Obstacle: {obstacle > 0.5})')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth_img > 0.3, cmap='gray')
    plt.title('Obstacle Mask')
    
    plt.savefig(f'debug_depth_{self.global_counter}.png')
    plt.close()
```

### 7.3 超参数调优建议

| 参数 | 初始值 | 调优范围 | 说明 |
|------|--------|----------|------|
| `depth.update_interval` | 1 | 1-5 | 更新频率，越小越精确但越慢 |
| `depth.far_clip` | 3.0 | 2.0-5.0 | 视距，跑酷需要看得远 |
| `reward.obstacle_clearance` | 2.0 | 1.0-5.0 | 跳跃奖励权重 |
| `reward.jump_timing` | 1.5 | 0.5-3.0 | 时机奖励权重 |
| `terrain_proportions[4]` | 0.45 | 0.3-0.6 | 障碍物比例 |
| `learning_rate` | 1e-3 | 5e-4 - 5e-3 | 学习率 |

### 7.4 常见问题与解决

#### 问题 1: 机器人不跳跃，只是减速
**原因**: 跳跃奖励不足以克服稳定性奖励
**解决**: 
- 增加 `obstacle_clearance` 权重
- 降低 `feet_air_time` 惩罚
- 降低 `lin_vel_z` 惩罚

#### 问题 2: 跳跃时机不对
**原因**: 深度感知不准确或延迟
**解决**:
- 减小 `depth.update_interval` 到 1
- 增加 `depth.buffer_len` 到 3-5
- 增加 `jump_timing` 奖励权重

#### 问题 3: 着陆后摔倒
**原因**: 着陆控制不足
**解决**:
- 增加 `landing_stability` 奖励
- 增加 `orientation` 惩罚
- 调整 PD 控制器参数（`stiffness`, `damping`）

#### 问题 4: 训练速度慢
**原因**: 深度图像处理开销大
**解决**:
- 减小 `camera_num_envs` 到 64-128
- 增大 `depth.update_interval`
- 使用更小的图像分辨率
- 使用简单特征而非原始深度图

### 7.5 性能优化

```python
# 使用混合精度训练
import torch.cuda.amp as amp

scaler = amp.GradScaler()

# 在训练循环中
with amp.autocast():
    value, action_log_probs, _, action_mu, action_sigma, _ = actor_critic.act(
        obs, critic_obs, hist_encoding
    )
    
loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 8. 完整训练流程

### 8.1 修改配置文件

```bash
# 编辑配置
nano configs/tita_constraint_config.py

# 修改以下关键参数：
# - depth.use_camera = True
# - terrain_proportions（增加障碍物）
# - rewards.scales（添加跑酷奖励）
```

### 8.2 运行训练

```bash
# 激活环境
conda activate rlgpu

# 开始训练
python train.py --task=tita_constraint --headless

# 如果要可视化
python train.py --task=tita_constraint

# 从检查点恢复
python train.py --task=tita_constraint --resume --load_run=<run_name>
```

### 8.3 监控训练

```bash
# 使用 TensorBoard
tensorboard --logdir=logs/tita_constraint

# 查看关键指标：
# - mean_reward
# - obstacle_clearance_reward
# - jump_timing_reward
# - landing_stability_reward
# - episode_length
```

### 8.4 测试策略

```python
# 创建测试脚本 test_parkour.py
import torch
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from envs import LeggedRobot
from utils.task_registry import task_registry

# 加载策略
env, env_cfg = task_registry.make_env(name='tita_constraint', args=args)
policy = torch.load('model_10000.pt')

# 测试循环
obs = env.reset()
for _ in range(1000):
    actions = policy(obs)
    obs, _, rewards, dones, infos = env.step(actions)
    
    # 记录跳跃成功率
    if 'obstacle_cleared' in infos:
        success_rate = infos['obstacle_cleared'].float().mean()
        print(f"Obstacle clearance rate: {success_rate:.2%}")
```

---

## 9. 预期效果与性能指标

### 9.1 训练收敛时间

- **阶段 1 (平地)**: 1000-2000 iterations (~1-2 小时)
- **阶段 2 (小障碍)**: 2000-3000 iterations (~2-3 小时)
- **阶段 3 (跑酷)**: 3000-5000 iterations (~3-5 小时)
- **总计**: ~10000 iterations (~10 小时，RTX 3060）

### 9.2 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 平均回报 | > 200 | 总体任务完成度 |
| 障碍物清除率 | > 80% | 成功越过障碍物比例 |
| 跳跃成功率 | > 70% | 跳跃后稳定着陆比例 |
| 平均速度 | > 0.8 m/s | 前进速度 |
| Episode 长度 | > 500 steps | 存活时间 |

---

## 10. 参考资料

### 10.1 相关论文

1. **Learning Quadrupedal Locomotion over Challenging Terrain** (2020)
   - 深度视觉 + 本体感受融合

2. **Learning to Walk in Minutes Using Massively Parallel Deep RL** (2021)
   - Isaac Gym 并行训练

3. **Visual-Locomotion** (2023)
   - 视觉引导的四足跑酷

### 10.2 代码参考

- Isaac Gym 官方示例: `python/examples/`
- Legged Gym: https://github.com/leggedrobotics/legged_gym
- N3PO: https://github.com/zeonsunlightyu/LocomotionWithNP3O

---

## 附录 A: 完整配置示例

```python
# configs/tita_parkour_config.py (新建文件)
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO

class TitaParkourCfg(TitaConstraintRoughCfg):
    class env(TitaConstraintRoughCfg.env):
        n_depth_features = 5
        num_observations = 33 + 187 + 10*33 + 36 + 5  # 添加深度特征
    
    class depth(TitaConstraintRoughCfg.depth):
        use_camera = True
        update_interval = 1
        buffer_len = 3
        far_clip = 3.0
    
    class terrain(TitaConstraintRoughCfg.terrain):
        terrain_proportions = [0.05, 0.05, 0.25, 0.20, 0.45]
        num_rows = 15
    
    class rewards(TitaConstraintRoughCfg.rewards):
        class scales(TitaConstraintRoughCfg.rewards.scales):
            tracking_lin_vel = 1.0
            obstacle_clearance = 2.0     # 新增
            jump_timing = 1.5            # 新增
            landing_stability = 1.0      # 新增
            feet_air_time = 0.5          # 降低
            lin_vel_z = -0.5             # 降低
            collision = -5.0             # 增加

class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class policy(TitaConstraintRoughCfgPPO.policy):
        use_depth_encoder = True
        depth_encoder_dims = [64, 32, 16]
    
    class runner(TitaConstraintRoughCfgPPO.runner):
        experiment_name = 'tita_parkour'
        run_name = 'depth_v1'
        max_iterations = 10000
```

---

**文档版本**: v1.0  
**最后更新**: 2025-11-19  
**作者**: GitHub Copilot  
**适用版本**: TITA RL (Isaac Gym Preview 4)

如有问题，请在 Issues 中反馈！🚀
