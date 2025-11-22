# TITA 机器人部署指南

## ❓ 快速问答

### Q: TITA 机器人有 ROS 接口吗？
**A: 有！而且是完整的 ROS2 Humble 接口。** 

TITA 使用 **ros2_control** 框架，通过标准 ROS2 话题和服务实现：
- ✅ 速度控制 (`geometry_msgs/Twist`)
- ✅ 状态反馈 (自定义 `RobotStates` 消息)
- ✅ 传感器数据 (IMU, 关节状态)
- ✅ 遥控器/键盘控制接口
- ✅ RL 推理引擎集成在 `tita_controller` 的 FSM 状态机中

详见下方 [🔌 ROS2 接口详解](#-ros2-接口详解) 章节。

---

## 📁 正确的源代码目录

根据您的工作空间结构分析，**需要部署的 src 目录是：**

```
/home/bubble/桌面/tita/tita_rl_sim2sim2real/src
```

### 为什么是这个目录？

这个 `src` 目录包含了所有实机部署所需的 ROS2 包：

```
src/
├── tita_locomotion/          # 运动控制核心包
│   ├── locomotion_bringup/   ✅ (hw_bringup.launch.py 在这里)
│   ├── tita_controllers/
│   │   └── tita_controller/  ✅ (机器人控制器)
│   ├── interaction/
│   │   ├── joy_controller/    ✅ (遥控器控制)
│   │   └── keyboard_controller/ ✅ (键盘控制)
│   ├── devices/
│   │   └── hw_broadcaster/    ✅ (硬件广播器)
│   ├── tita_description/      (机器人模型描述)
│   └── locomotion_msgs/       (消息定义)
├── tita_bringup/             # 启动配置
└── tita_command/             # 命令管理
    └── teleop_command/
```

## ⚠️ 注意：命令中的包名问题

您提到的部署命令中有一个 **不存在的包名**：

```bash
# ❌ 这个命令有问题
colcon build --packages-up-to locomotion_bringup template_ros2_controller tita_controller joy_controller keyboard_controller hw_broadcaster
```

**问题：`template_ros2_controller` 这个包在 src 目录中不存在！**

### 正确的编译命令应该是：

```bash
# ✅ 正确的命令（移除了不存在的 template_ros2_controller）
colcon build --packages-up-to locomotion_bringup tita_controller joy_controller keyboard_controller hw_broadcaster
```

## 📋 完整的实机部署流程

### 1. 拷贝文件到机器人

```bash
# 从宿主机执行
scp -r /home/bubble/桌面/tita/tita_rl_sim2sim2real/src robot@192.168.42.1:~/tita_ros2/
```

### 2. 连接到机器人

```bash
ssh robot@192.168.42.1
# 密码: apollo
```

### 3. 停止自启动服务

```bash
systemctl stop tita-bringup.service
```

### 4. 编译 ROS2 包

```bash
cd ~/tita_ros2/

source /opt/ros/humble/setup.bash

# ✅ 正确的编译命令
colcon build --packages-up-to \
  locomotion_bringup \
  tita_controller \
  joy_controller \
  keyboard_controller \
  hw_broadcaster

source install/setup.bash
```

### 5. 部署推理引擎文件

在编译之前，需要修改推理引擎路径：

**文件位置：**
```
src/tita_locomotion/tita_controllers/tita_controller/src/fsm/FSMState_RL.cpp
```

**需要修改的内容：**
将 `.engine` 文件路径改为机器人上的实际路径（例如：`~/tita_ros2/model_gn.engine`）

### 6. 转换 ONNX 为 TensorRT 引擎

如果机器人镜像缺少 TensorRT 开发工具：

```bash
# 安装依赖
sudo apt install nvidia-cuda-dev
sudo apt install tensorrt-dev
sudo apt install tensorrt

# 转换模型（使用您训练好的 onnx 文件）
/usr/src/tensorrt/bin/trtexec \
  --onnx=/path/to/your/policy.onnx \
  --saveEngine=~/tita_ros2/model_gn.engine
```

**可用的 ONNX 文件位置：**
- `/home/bubble/桌面/tita/tita_rl/exported/policy.onnx`
- `/home/bubble/桌面/tita/tita_rl/tita_example_10000.onnx`

### 7. 连接遥控器

```bash
crsf-app -bind
```

### 8. 启动机器人系统

```bash
# 终端 1: 启动硬件控制
nohup ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc &

# 终端 2: 启动遥控器控制
nohup ros2 launch joy_controller joy_controller.launch.py &
```

## 🔧 故障排查

### 问题 1: 找不到 template_ros2_controller

**解决方案：** 这个包不存在，从编译命令中移除即可

### 问题 2: 编译失败

检查是否所有依赖都已安装：
```bash
rosdep install --from-paths src --ignore-src -r -y
```

### 问题 3: TensorRT 版本问题

如果是 TensorRT 10.x 版本，参考：
https://github.com/DDTRobot/tita_rl_sim2sim2real/issues/1

### 问题 4: 找不到机器人描述文件

```bash
sudo mkdir -p /usr/share/robot_description
sudo cp -r ~/tita_ros2/src/tita_locomotion/tita_description/tita /usr/share/robot_description/
```

## 📝 总结

✅ **要部署的目录：** `/home/bubble/桌面/tita/tita_rl_sim2sim2real/src`

✅ **正确的包列表：**
- `locomotion_bringup`
- `tita_controller`
- `joy_controller`
- `keyboard_controller`
- `hw_broadcaster`

❌ **不存在的包：** `template_ros2_controller` （需要从命令中移除）

🔑 **关键文件：**
- 推理引擎：`model_gn.engine`（需要从 ONNX 转换）
- 控制器代码：`FSMState_RL.cpp`（需要修改引擎路径）
- 启动文件：`hw_bringup.launch.py`（在 locomotion_bringup 包中）

## � ROS2 接口详解

### ✅ 是的，TITA 有完整的 ROS2 接口！

TITA 机器人通过 **ROS2 Humble** 实现了完整的通信架构，基于 **ros2_control** 框架，所有控制和感知数据都通过 ROS2 话题和服务传输。

### 📡 核心 ROS2 话题列表

#### 1. 控制命令输入 (Subscriptions)

| 话题名称 | 消息类型 | 功能 | 发布者 |
|---------|---------|------|--------|
| `/tita/command/manager/cmd_twist` | `geometry_msgs/msg/Twist` | 速度控制命令 (线速度/角速度) | keyboard_controller / joy_controller |
| `/tita/command/manager/cmd_pose` | `geometry_msgs/msg/PoseStamped` | 位姿控制命令 | keyboard_controller / joy_controller |
| `/tita/command/manager/cmd_key` | `std_msgs/msg/String` | FSM 状态切换命令 | keyboard_controller / joy_controller |
| `/tita/command/teleop/command` | `sensor_msgs/msg/Joy` | 遥控器原始数据 | joy 驱动节点 |

#### 2. 状态反馈输出 (Publications)

| 话题名称 | 消息类型 | 功能 | 订阅者 |
|---------|---------|------|--------|
| `/tita/tita_controller/plan_commands` | `locomotion_msgs/msg/PlanCommands` | 规划命令（目标速度、位姿、关节状态） | 监控/可视化节点 |
| `/tita/tita_controller/robot_states` | `locomotion_msgs/msg/RobotStates` | 机器人完整状态（位姿、速度、关节等） | 监控/可视化节点 |
| `/imu_sensor_broadcaster/imu` | `sensor_msgs/msg/Imu` | IMU 数据（姿态、角速度、加速度） | 控制器/定位 |
| `/joint_states` | `sensor_msgs/msg/JointState` | 关节状态（位置、速度、力矩） | rviz / 控制器 |
| `/locomotion/body/fsm_mode` | `std_msgs/msg/String` | 当前 FSM 状态 | 监控节点 |
| `/locomotion/motors_status` | `diagnostic_msgs/msg/DiagnosticArray` | 电机诊断信息 | 监控节点 |
| `/odom` | `nav_msgs/msg/Odometry` | 里程计数据 | 导航/定位 |

### 📦 自定义消息类型

#### `locomotion_msgs/msg/RobotStates.msg`
```msg
std_msgs/Header header
string fsm_state_name                    # 当前 FSM 状态名称
geometry_msgs/Vector3 twist_linear       # 线速度 (x, y, z)
geometry_msgs/Vector3 twist_angular      # 角速度 (roll, pitch, yaw)
geometry_msgs/Vector3 pose_position      # 位置 (x, y, z)
geometry_msgs/Vector3 pose_rpy           # 姿态 (roll, pitch, yaw)
float64 two_wheel_distance               # 双轮间距
geometry_msgs/Vector3 com_position_relative  # 质心相对位置
float64[] joint_positions                # 关节位置 (8 个关节)
float64[] joint_velocities               # 关节速度
float64[] joint_torques                  # 关节力矩
```

#### `locomotion_msgs/msg/PlanCommands.msg`
```msg
std_msgs/Header header
string fsm_state_name                    # 目标 FSM 状态
geometry_msgs/Vector3 twist_linear       # 目标线速度
geometry_msgs/Vector3 twist_angular      # 目标角速度
geometry_msgs/Vector3 pose_position      # 目标位置
geometry_msgs/Vector3 pose_rpy           # 目标姿态
float64[] joint_positions                # 目标关节位置
float64[] joint_velocities               # 目标关节速度
float64[] joint_torques                  # 目标关节力矩
```

### 🎮 控制流程说明

```
┌─────────────────────┐
│  遥控器/键盘输入     │
└──────────┬──────────┘
           │ Twist/Joy 消息
           ▼
┌─────────────────────────────────────┐
│  joy_controller / keyboard_controller│
│  - 解析输入命令                      │
│  - 发布标准化控制命令                │
└──────────┬──────────────────────────┘
           │ cmd_twist, cmd_key
           ▼
┌─────────────────────────────────────┐
│  tita_controller                     │
│  ┌─────────────────────────────┐   │
│  │ FSM (有限状态机)            │   │
│  │ - PASSIVE (被动模式)        │   │
│  │ - RL (强化学习控制) ← 使用  │   │
│  │ - WBC (全身控制)            │   │
│  └─────────────────────────────┘   │
│                                     │
│  FSMState_RL:                       │
│  - 读取传感器数据 (_GetObs)        │
│  - 调用 TensorRT 推理 (_Forward)   │
│  - 输出关节目标位置                 │
└──────────┬──────────────────────────┘
           │ 关节命令
           ▼
┌─────────────────────────────────────┐
│  ros2_control (controller_manager)   │
│  - 硬件接口层                        │
│  - 电机驱动                          │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  TITA 机器人硬件    │
│  - 8 个关节电机      │
│  - IMU 传感器        │
│  - 力传感器          │
└─────────────────────┘
```

### 🧠 RL 推理引擎集成

在 `FSMState_RL.cpp` 中，RL 策略通过 TensorRT 引擎执行：

```cpp
// 初始化 TensorRT 引擎
cuda_test_ = std::make_shared<CudaTest>("/mnt/dev/tita_rl/final.engine");

// 每个控制周期 (50Hz):
void FSMState_RL::run() {
    // 1. 获取 ROS 命令
    x_vel_cmd_ = _data->state_command->rc_data_->twist_linear[X];
    y_vel_cmd_ = _data->state_command->rc_data_->twist_linear[Y];
    pitch_cmd_ = _data->state_command->rc_data_->twist_angular[Z];
    
    // 2. 构建观测向量 (33 维本体感 + 297 维历史)
    _GetObs();  // 填充 input_0, input_1
    
    // 3. TensorRT 推理 (异步线程)
    _Forward(); // 输出 8 个关节目标位置到 output[]
    
    // 4. 发送关节命令到硬件
    for (int i = 0; i < 8; i++) {
        desired_pos[i] = output[i] * action_scale + default_dof_pos[i];
    }
}
```

### 📊 观测空间构成 (330 维输入)

| 组成部分 | 维度 | 来源 | 说明 |
|---------|-----|------|------|
| 本体感观测 | 33 | 当前传感器 | 线速度(3) + 角速度(3) + 投影重力(3) + 命令(3) + 关节位置(8) + 关节速度(8) + 上次动作(8) |
| 历史观测 | 297 | 观测缓冲区 | 过去 9 步的本体感观测 (33×9) |

### 🔧 实机部署时的 ROS2 接口注意事项

#### 1. Topic 命名空间
所有话题都在 `/tita` 命名空间下，通过 `--ros-args -r __ns:=/tita` 设置

#### 2. QoS 配置
```cpp
auto subscribers_qos = rclcpp::SystemDefaultsQoS();
subscribers_qos.keep_last(1);        // 只保留最新消息
subscribers_qos.best_effort();       // 尽力而为模式，降低延迟
```

#### 3. 控制频率
- **控制器更新频率**: 200Hz (ros2_control 硬件接口)
- **RL 策略推理频率**: 50Hz (每 4 个控制周期执行一次)
- **传感器发布频率**: 100Hz (IMU), 200Hz (关节状态)

#### 4. 关键话题监控命令

```bash
# 查看所有话题
ros2 topic list

# 监控速度命令
ros2 topic echo /tita/command/manager/cmd_twist

# 监控机器人状态
ros2 topic echo /tita/tita_controller/robot_states

# 查看关节状态
ros2 topic echo /joint_states

# 监控 IMU 数据
ros2 topic echo /imu_sensor_broadcaster/imu

# 查看当前 FSM 状态
ros2 topic echo /locomotion/body/fsm_mode

# 发送测试速度命令
ros2 topic pub /tita/command/manager/cmd_twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# 切换到 RL 模式
ros2 topic pub /tita/command/manager/cmd_key std_msgs/msg/String \
  "{data: 'RL'}"
```

### 🎯 推理引擎路径配置

**重要**: 部署前必须修改 TensorRT 引擎路径！

**文件**: `src/tita_locomotion/tita_controllers/tita_controller/src/fsm/FSMState_RL.cpp`

**第 22 行**:
```cpp
// ❌ 错误的路径（宿主机路径）
cuda_test_ = std::make_shared<CudaTest>("/mnt/dev/tita_rl/final.engine");

// ✅ 正确的路径（机器人上的实际路径）
cuda_test_ = std::make_shared<CudaTest>("/home/robot/tita_ros2/model_gn.engine");
```

### 📋 ROS2 包依赖关系

```
locomotion_bringup (启动文件)
    ├── tita_controller (核心控制器)
    │   ├── FSMState_RL (RL 推理)
    │   ├── FSMState_WBC (全身控制)
    │   └── FSMState_PASSIVE (被动模式)
    ├── hw_broadcaster (硬件状态广播)
    ├── joy_controller (遥控器接口)
    ├── keyboard_controller (键盘接口)
    └── locomotion_msgs (自定义消息)
```

## �📚 相关文档

- 仿真环境配置：`/home/bubble/桌面/tita/tita_rl_sim2sim2real/ReadMe.md`
- 训练配置：`/home/bubble/桌面/tita/README.md`
- ONNX 导出：`/home/bubble/桌面/tita/tita_rl/export_direct.py`
- ROS2 话题定义：`/home/bubble/桌面/tita/tita_rl_sim2sim2real/src/tita_bringup/include/tita_utils/topic_names.hpp`
