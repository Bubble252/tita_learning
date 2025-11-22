# TITA 机器人 SSH 部署完整指南

## 📋 部署流程概览

```
本地电脑 (开发环境)                    机器人 (192.168.42.1)
    │                                        │
    │ 1. 编译测试 (可选)                     │
    │ 2. SCP 传输源码                        │
    ├────────────────────────────────────────>│
    │                                        │ 3. 编译 ROS2 包
    │                                        │ 4. 转换 TensorRT 引擎
    │                                        │ 5. 启动机器人
```

---

## 🖥️ 本地操作（开发环境）

### 问题诊断：为什么本地运行 `ros2 launch joy_controller` 会报错？

**错误信息：**
```
Package 'joy_controller' not found: "package 'joy_controller' not found, searching: ['/opt/ros/humble']"
```

**原因：**
1. ✅ `joy_controller` 包确实存在于 `tita_rl_sim2sim2real/src/tita_locomotion/interaction/joy_controller/`
2. ❌ 但您**没有在本地编译**这个工作空间
3. ❌ 或者**没有 source** 编译后的 `install/setup.bash`

### 解决方案 1: 本地编译测试（可选）

如果您想在本地测试（例如使用 Webots 仿真），需要先编译：

```bash
# 进入工作空间
cd ~/桌面/tita/tita_rl_sim2sim2real

# 确保 ROS2 环境已加载
source /opt/ros/humble/setup.bash

# 编译所有包（或仅编译需要的包）
colcon build --packages-up-to \
  locomotion_bringup \
  tita_controller \
  joy_controller \
  keyboard_controller \
  hw_broadcaster

# source 编译结果
source install/setup.bash

# 现在可以运行了
ros2 launch joy_controller joy_controller.launch.py
```

### 解决方案 2: 仅传输源码到机器人（推荐）

**如果您只是要部署到机器人，不需要在本地编译**，直接传输源码：

```bash
# 从本地传输源码到机器人
cd ~/桌面/tita

# 方式 1: 传输整个 src 目录（推荐）
scp -r tita_rl_sim2sim2real/src robot@192.168.42.1:~/tita_ros2/

# 方式 2: 如果机器人上已有旧版本，先删除再传输
ssh robot@192.168.42.1 "rm -rf ~/tita_ros2/src"
scp -r tita_rl_sim2sim2real/src robot@192.168.42.1:~/tita_ros2/

# 方式 3: 使用 rsync 同步（更高效，增量传输）
rsync -avz --progress tita_rl_sim2sim2real/src/ robot@192.168.42.1:~/tita_ros2/src/
```

### 额外：传输训练好的模型文件

```bash
# 传输 ONNX 模型文件到机器人
scp tita_rl/exported/policy.onnx robot@192.168.42.1:~/tita_ros2/

# 或传输已有的 .pt 检查点（需要在机器人上转换）
scp tita_rl/model_11700.pt robot@192.168.42.1:~/tita_ros2/
```

---

## 🤖 机器人上的操作（SSH 进入后）

### 步骤 1: 连接到机器人

```bash
# 从本地电脑连接
ssh robot@192.168.42.1
# 密码: apollo
```

### 步骤 2: 停止自启动服务

```bash
# 停止机器人的自启动 ROS2 服务
systemctl stop tita-bringup.service

# 验证服务已停止
systemctl status tita-bringup.service
```

### 步骤 3: 检查源码是否传输成功

```bash
cd ~/tita_ros2

# 检查 src 目录结构
ls -la src/

# 应该看到:
# src/
# ├── tita_bringup/
# ├── tita_command/
# └── tita_locomotion/
#     ├── locomotion_bringup/
#     ├── tita_controllers/
#     ├── interaction/
#     │   ├── joy_controller/
#     │   └── keyboard_controller/
#     └── devices/
#         └── hw_broadcaster/
```

### 步骤 4: 修改 TensorRT 引擎路径（重要！）

在编译之前，必须修改推理引擎路径：

```bash
# 编辑 FSMState_RL.cpp
nano ~/tita_ros2/src/tita_locomotion/tita_controllers/tita_controller/src/fsm/FSMState_RL.cpp

# 找到第 22 行左右，修改为机器人上的实际路径:
# 原来可能是: /mnt/dev/tita_rl/final.engine
# 改为: /home/robot/tita_ros2/model_gn.engine
```

**修改示例：**
```cpp
// 第 22 行附近
// ❌ 错误（宿主机路径或 Docker 路径）
cuda_test_ = std::make_shared<CudaTest>("/mnt/dev/tita_rl/final.engine");

// ✅ 正确（机器人实际路径）
cuda_test_ = std::make_shared<CudaTest>("/home/robot/tita_ros2/model_gn.engine");
```

保存并退出（Ctrl+O, Enter, Ctrl+X）

### 步骤 5: 编译 ROS2 包

```bash
cd ~/tita_ros2

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash

# 编译（这次在机器人上编译）
colcon build --packages-up-to \
  locomotion_bringup \
  tita_controller \
  joy_controller \
  keyboard_controller \
  hw_broadcaster

# 如果编译成功，加载编译结果
source install/setup.bash
```

**可能的编译问题：**

```bash
# 如果缺少依赖，安装它们
rosdep install --from-paths src --ignore-src -r -y

# 如果之前编译过，清理后重新编译
rm -rf build/ install/ log/
colcon build --packages-up-to locomotion_bringup tita_controller joy_controller keyboard_controller hw_broadcaster
```

### 步骤 6: 转换 ONNX 为 TensorRT 引擎

```bash
# 检查是否有 ONNX 文件
ls -lh ~/tita_ros2/policy.onnx

# 如果 TensorRT 工具未安装，先安装
sudo apt update
sudo apt install -y nvidia-cuda-dev tensorrt-dev tensorrt

# 转换 ONNX 为 TensorRT 引擎
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/robot/tita_ros2/policy.onnx \
  --saveEngine=/home/robot/tita_ros2/model_gn.engine \
  --fp16  # 可选：使用 FP16 加速

# 验证引擎文件生成
ls -lh ~/tita_ros2/model_gn.engine
```

**注意事项：**
- TensorRT 版本需要与训练时的版本兼容
- 如果是 TensorRT 10.x，参考 [这个 issue](https://github.com/DDTRobot/tita_rl_sim2sim2real/issues/1)

### 步骤 7: 连接遥控器（如果使用遥控器）

```bash
# 绑定 CRSF 遥控器
crsf-app -bind

# 等待遥控器连接成功的提示
```

### 步骤 8: 启动机器人系统

#### 方式 1: 前台启动（调试用）

```bash
# 终端 1: 启动硬件控制
source /opt/ros/humble/setup.bash
source ~/tita_ros2/install/setup.bash
ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc

# 终端 2: SSH 再开一个终端，启动遥控器
ssh robot@192.168.42.1
source /opt/ros/humble/setup.bash
source ~/tita_ros2/install/setup.bash
ros2 launch joy_controller joy_controller.launch.py
```

#### 方式 2: 后台启动（生产用）

```bash
# 加载环境
source /opt/ros/humble/setup.bash
source ~/tita_ros2/install/setup.bash

# 后台启动硬件控制
nohup ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc > ~/tita_hw.log 2>&1 &

# 后台启动遥控器
nohup ros2 launch joy_controller joy_controller.launch.py > ~/tita_joy.log 2>&1 &

# 查看日志
tail -f ~/tita_hw.log
tail -f ~/tita_joy.log

# 查看进程
ps aux | grep ros2
```

### 步骤 9: 验证系统运行

```bash
# 新开一个 SSH 终端
ssh robot@192.168.42.1

# 加载环境
source /opt/ros/humble/setup.bash
source ~/tita_ros2/install/setup.bash

# 查看所有 ROS2 话题
ros2 topic list | grep tita

# 查看机器人状态
ros2 topic echo /tita/tita_controller/robot_states

# 查看当前 FSM 模式
ros2 topic echo /locomotion/body/fsm_mode

# 发送测试命令（切换到 RL 模式）
ros2 topic pub --once /tita/command/manager/cmd_key std_msgs/msg/String "{data: 'RL'}"

# 发送速度命令测试
ros2 topic pub --rate 10 /tita/command/manager/cmd_twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

## 🔄 工作流程总结

### 典型开发迭代流程

```bash
# ========== 本地电脑 ==========
# 1. 修改代码（例如调整奖励函数、训练参数）
cd ~/桌面/tita/tita_rl
vim configs/tita_constraint_config.py

# 2. 训练新模型
conda activate tita2
python train.py --task=tita_constraint --headless

# 3. 导出 ONNX
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

# 4. 传输到机器人
scp tita_rl/exported/policy.onnx robot@192.168.42.1:~/tita_ros2/

# ========== 机器人上 ==========
# 5. SSH 登录
ssh robot@192.168.42.1

# 6. 转换引擎
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/robot/tita_ros2/policy.onnx \
  --saveEngine=/home/robot/tita_ros2/model_gn.engine

# 7. 重启控制器
systemctl restart tita-bringup.service
# 或手动启动测试
```

---

## ❓ 常见问题排查

### Q1: 本地运行 `ros2 launch joy_controller` 报错找不到包

**A:** 这是正常的！`joy_controller` 在 `tita_rl_sim2sim2real` 工作空间中，需要：

```bash
# 选项 1: 在本地编译（用于仿真测试）
cd ~/桌面/tita/tita_rl_sim2sim2real
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch joy_controller joy_controller.launch.py

# 选项 2: 只在机器人上运行（部署时）
# 本地不需要运行，直接 scp 传输源码即可
```

### Q2: SCP 传输时提示 "Permission denied"

**A:** 检查 SSH 密钥或密码：

```bash
# 测试 SSH 连接
ssh robot@192.168.42.1
# 密码: apollo

# 如果需要配置免密登录
ssh-keygen -t rsa
ssh-copy-id robot@192.168.42.1
```

### Q3: 机器人编译失败，提示缺少依赖

**A:** 安装依赖：

```bash
# 在机器人上
cd ~/tita_ros2
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### Q4: TensorRT 转换失败

**A:** 检查 ONNX 文件和 TensorRT 版本：

```bash
# 检查 ONNX 文件
python3 -c "import onnx; model = onnx.load('policy.onnx'); print(onnx.checker.check_model(model))"

# 检查 TensorRT 版本
dpkg -l | grep tensorrt

# 如果是 TensorRT 10.x，可能需要特殊处理
# 参考: https://github.com/DDTRobot/tita_rl_sim2sim2real/issues/1
```

### Q5: 机器人启动后无响应

**A:** 检查日志和状态：

```bash
# 查看控制器日志
journalctl -u tita-bringup.service -f

# 或查看手动启动的日志
tail -f ~/tita_hw.log

# 检查 ROS2 节点
ros2 node list

# 检查话题连接
ros2 topic info /tita/command/manager/cmd_twist
```

---

## 📝 快速参考命令

### 本地到机器人传输

```bash
# 传输源码
scp -r ~/桌面/tita/tita_rl_sim2sim2real/src robot@192.168.42.1:~/tita_ros2/

# 传输模型
scp ~/桌面/tita/tita_rl/exported/policy.onnx robot@192.168.42.1:~/tita_ros2/

# 使用 rsync（推荐，增量传输）
rsync -avz --progress ~/桌面/tita/tita_rl_sim2sim2real/src/ robot@192.168.42.1:~/tita_ros2/src/
```

### 机器人编译和启动

```bash
# 完整启动流程（一键复制）
ssh robot@192.168.42.1 << 'EOF'
systemctl stop tita-bringup.service
cd ~/tita_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-up-to locomotion_bringup tita_controller joy_controller keyboard_controller hw_broadcaster
source install/setup.bash
/usr/src/tensorrt/bin/trtexec --onnx=/home/robot/tita_ros2/policy.onnx --saveEngine=/home/robot/tita_ros2/model_gn.engine
nohup ros2 launch locomotion_bringup hw_bringup.launch.py ctrl_mode:=wbc > ~/tita_hw.log 2>&1 &
nohup ros2 launch joy_controller joy_controller.launch.py > ~/tita_joy.log 2>&1 &
EOF
```

---

## 🎯 总结

- **本地电脑**：训练模型 → 导出 ONNX → SCP 传输
- **机器人**：编译 ROS2 包 → 转换 TensorRT → 启动系统
- **不需要在本地编译 ROS2 包**（除非要用 Webots 仿真）
- **所有 ROS2 命令都在机器人上运行**

相关文档：
- [完整部署指南](DEPLOYMENT_GUIDE.md)
- [ROS2 接口速查](ROS2_INTERFACE_QUICK_REFERENCE.md)
