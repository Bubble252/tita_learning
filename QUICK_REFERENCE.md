# 🚀 TITA 部署快速参考卡

## 一句话总结

**本地训练 → SCP 传输 → 机器人编译运行**

---

## 📍 当前问题解答

### 错误：`Package 'joy_controller' not found`

**原因：** 您在本地运行 `ros2 launch joy_controller`，但该包在 `tita_rl_sim2sim2real` 工作空间中，**未编译或未 source**。

**解决方案：**

#### ✅ 推荐方式：只在机器人上运行

```bash
# 本地：不需要编译，直接传输
scp -r ~/桌面/tita/tita_rl_sim2sim2real/src robot@192.168.42.1:~/tita_ros2/

# 机器人：SSH 登录后编译运行
ssh robot@192.168.42.1
cd ~/tita_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-up-to joy_controller
source install/setup.bash
ros2 launch joy_controller joy_controller.launch.py  # ✅ 现在可以运行了
```

#### 🔧 可选方式：本地编译（仅用于 Webots 仿真测试）

```bash
cd ~/桌面/tita/tita_rl_sim2sim2real
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch joy_controller joy_controller.launch.py  # ✅ 本地也能运行
```

---

## 🔄 标准部署流程（3 步）

### 步骤 1: 本地传输源码和模型

```bash
# 传输源码
scp -r ~/桌面/tita/tita_rl_sim2sim2real/src robot@192.168.42.1:~/tita_ros2/

# 传输 ONNX 模型
scp ~/桌面/tita/tita_rl/exported/policy.onnx robot@192.168.42.1:~/tita_ros2/
```

### 步骤 2: 机器人编译

```bash
ssh robot@192.168.42.1  # 密码: apollo

systemctl stop tita-bringup.service

cd ~/tita_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-up-to locomotion_bringup tita_controller joy_controller keyboard_controller hw_broadcaster
source install/setup.bash
```

### 步骤 3: 转换引擎并启动

```bash
# 转换 TensorRT 引擎
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/robot/tita_ros2/policy.onnx \
  --saveEngine=/home/robot/tita_ros2/model_gn.engine

# 启动
nohup ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc &
nohup ros2 launch joy_controller joy_controller.launch.py &
```

---

## ⚠️ 重要注意事项

### 修改推理引擎路径（必须！）

编译前修改文件：
```bash
# 在机器人上编辑
nano ~/tita_ros2/src/tita_locomotion/tita_controllers/tita_controller/src/fsm/FSMState_RL.cpp
```

第 22 行改为：
```cpp
cuda_test_ = std::make_shared<CudaTest>("/home/robot/tita_ros2/model_gn.engine");
```

---

## 📋 检查清单

部署前确认：

- [ ] ONNX 模型已导出（`policy.onnx`）
- [ ] 源码已通过 SCP 传输到机器人
- [ ] 已修改 `FSMState_RL.cpp` 中的引擎路径
- [ ] 机器人上已安装 TensorRT（`sudo apt install tensorrt`）
- [ ] 已停止自启动服务（`systemctl stop tita-bringup.service`）

部署后验证：

- [ ] ROS2 包编译成功（无错误）
- [ ] TensorRT 引擎文件存在（`ls ~/tita_ros2/model_gn.engine`）
- [ ] 话题可见（`ros2 topic list | grep tita`）
- [ ] 机器人状态正常（`ros2 topic echo /tita/tita_controller/robot_states`）

---

## 🆘 紧急停止

```bash
# SSH 登录机器人
ssh robot@192.168.42.1

# 停止所有 ROS2 进程
pkill -9 ros2

# 或重启服务
systemctl restart tita-bringup.service

# 或切换到被动模式
ros2 topic pub --once /tita/command/manager/cmd_key std_msgs/msg/String "{data: 'PASSIVE'}"
```

---

## 📖 详细文档

完整指南请查看：
- **SSH 部署指南**: `SSH_DEPLOYMENT_GUIDE.md`
- **ROS2 接口详解**: `DEPLOYMENT_GUIDE.md`
- **话题速查**: `ROS2_INTERFACE_QUICK_REFERENCE.md`
