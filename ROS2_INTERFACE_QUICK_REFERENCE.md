# TITA ROS2 接口速查表

## 🚀 快速启动命令

```bash
# 启动硬件控制 (RL 模式)
ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc

# 启动遥控器
ros2 launch joy_controller joy_controller.launch.py

# 或者启动键盘控制
ros2 run keyboard_controller keyboard_controller_node --ros-args -r __ns:=/tita
```

## 📡 核心话题速查

### 控制命令 (发送到机器人)

```bash
# 速度控制 (前进 0.5 m/s)
ros2 topic pub /tita/command/manager/cmd_twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# 转向 (原地旋转 0.5 rad/s)
ros2 topic pub /tita/command/manager/cmd_twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"

# 切换到 RL 模式
ros2 topic pub /tita/command/manager/cmd_key std_msgs/msg/String "{data: 'RL'}"

# 切换到被动模式 (安全停止)
ros2 topic pub /tita/command/manager/cmd_key std_msgs/msg/String "{data: 'PASSIVE'}"
```

### 状态监控 (从机器人接收)

```bash
# 查看所有话题
ros2 topic list | grep tita

# 监控机器人完整状态
ros2 topic echo /tita/tita_controller/robot_states

# 监控关节状态
ros2 topic echo /joint_states

# 监控 IMU 数据
ros2 topic echo /imu_sensor_broadcaster/imu

# 查看当前控制模式
ros2 topic echo /locomotion/body/fsm_mode

# 查看电机诊断信息
ros2 topic echo /locomotion/motors_status
```

## 🎮 键盘控制说明

运行 `keyboard_controller_node` 后，可以使用以下按键：

| 按键 | 功能 | 对应话题 |
|-----|------|---------|
| `W` | 前进 | `/tita/command/manager/cmd_twist` |
| `S` | 后退 | `/tita/command/manager/cmd_twist` |
| `A` | 左转 | `/tita/command/manager/cmd_twist` |
| `D` | 右转 | `/tita/command/manager/cmd_twist` |
| `Q` | 左平移 | `/tita/command/manager/cmd_twist` |
| `E` | 右平移 | `/tita/command/manager/cmd_twist` |
| `空格` | 停止 | `/tita/command/manager/cmd_twist` |
| `R` | 切换到 RL 模式 | `/tita/command/manager/cmd_key` |
| `P` | 切换到被动模式 | `/tita/command/manager/cmd_key` |

## 📊 消息结构速查

### RobotStates (机器人状态)

```yaml
header:
  stamp: {sec: 1234567890, nanosec: 123456789}
  frame_id: "base_link"
fsm_state_name: "RL"
twist_linear: {x: 0.5, y: 0.0, z: 0.0}
twist_angular: {x: 0.0, y: 0.0, z: 0.2}
pose_position: {x: 1.2, y: 0.3, z: 0.35}
pose_rpy: {x: 0.01, y: -0.02, z: 1.57}
joint_positions: [0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0]  # 8 个关节
joint_velocities: [0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.2, 0.0]
joint_torques: [0.0, 5.2, 8.3, 0.0, 0.0, 5.2, 8.3, 0.0]
```

### Twist (速度命令)

```yaml
linear:
  x: 0.5   # 前进速度 (m/s)
  y: 0.0   # 侧向速度 (m/s)
  z: 0.0   # 未使用
angular:
  x: 0.0   # 未使用
  y: 0.0   # 未使用
  z: 0.3   # 转向角速度 (rad/s)
```

## 🔍 调试命令

```bash
# 查看节点信息
ros2 node list | grep tita
ros2 node info /tita/tita_controller

# 查看话题信息
ros2 topic info /tita/command/manager/cmd_twist
ros2 topic hz /tita/tita_controller/robot_states    # 查看发布频率

# 记录数据包
ros2 bag record /tita/tita_controller/robot_states /joint_states /imu_sensor_broadcaster/imu

# 回放数据包
ros2 bag play rosbag2_2024_01_15-12_34_56

# 可视化 (需要 rviz2)
rviz2 -d tita_config.rviz

# 查看 TF 树
ros2 run tf2_tools view_frames
```

## 🧪 测试序列

### 1. 基础功能测试

```bash
# 终端 1: 启动机器人
ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc

# 终端 2: 监控状态
ros2 topic echo /tita/tita_controller/robot_states

# 终端 3: 发送命令
# 1) 切换到 RL 模式
ros2 topic pub --once /tita/command/manager/cmd_key std_msgs/msg/String "{data: 'RL'}"

# 2) 等待 2 秒

# 3) 前进测试
ros2 topic pub --rate 10 /tita/command/manager/cmd_twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# 4) Ctrl+C 停止，等待机器人稳定

# 5) 转向测试
ros2 topic pub --rate 10 /tita/command/manager/cmd_twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}"
```

### 2. RL 推理性能测试

```bash
# 监控推理延迟
ros2 topic hz /tita/tita_controller/robot_states
# 预期: ~50 Hz (RL 策略推理频率)

ros2 topic hz /joint_states
# 预期: ~200 Hz (硬件控制频率)

# 检查 CPU/GPU 占用
ssh robot@192.168.42.1
htop
nvidia-smi -l 1
```

## 🛠️ 常见问题排查

### 问题 1: 没有收到状态反馈

```bash
# 检查节点是否运行
ros2 node list | grep tita_controller

# 检查话题是否存在
ros2 topic list | grep robot_states

# 检查话题连接
ros2 topic info /tita/tita_controller/robot_states
```

### 问题 2: 发送命令无响应

```bash
# 检查当前 FSM 状态
ros2 topic echo /locomotion/body/fsm_mode

# 确认必须在 RL 或 WBC 模式下才能响应速度命令
# 如果在 PASSIVE 模式，先切换:
ros2 topic pub --once /tita/command/manager/cmd_key std_msgs/msg/String "{data: 'RL'}"
```

### 问题 3: RL 推理失败

```bash
# 检查 TensorRT 引擎文件
ssh robot@192.168.42.1
ls -lh ~/tita_ros2/model_gn.engine

# 查看控制器日志
ros2 run rqt_console rqt_console

# 检查 FSMState_RL.cpp 中的引擎路径是否正确
# 应该是: /home/robot/tita_ros2/model_gn.engine
```

## 📈 性能指标参考

| 指标 | 预期值 | 说明 |
|-----|--------|------|
| 控制器更新频率 | 200 Hz | ros2_control 硬件接口 |
| RL 推理频率 | 50 Hz | TensorRT 前向传播 |
| 关节状态发布频率 | 200 Hz | 实时反馈 |
| IMU 发布频率 | 100 Hz | 传感器采样率 |
| 推理延迟 | < 5 ms | GPU 加速 |
| 端到端延迟 | < 20 ms | 命令到执行 |

## 🔗 相关资源

- **完整部署指南**: `DEPLOYMENT_GUIDE.md`
- **训练文档**: `README.md`
- **话题命名定义**: `src/tita_bringup/include/tita_utils/topic_names.hpp`
- **自定义消息**: `src/tita_locomotion/locomotion_msgs/msg/`
- **控制器源码**: `src/tita_locomotion/tita_controllers/tita_controller/`
- **RL 推理代码**: `src/tita_locomotion/tita_controllers/tita_controller/src/fsm/FSMState_RL.cpp`

---

**提示**: 所有命令都假设已经正确配置 ROS2 环境 (`source /opt/ros/humble/setup.bash && source install/setup.bash`)
