# OpenCV 深度图像可视化指南

## ✅ 简短回答

**是的！** 原始代码中已经集成了 **OpenCV** 来实时查看深度图像。代码在 `envs/legged_robot.py` 中。

---

## 1️⃣ 代码位置

### 文件：`envs/legged_robot.py`

```python
# 第 17 行：导入 OpenCV
import cv2

# 第 1030-1033 行：可视化深度图像
if self.cfg.depth.use_camera:
    window_name = "Depth Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
    cv2.waitKey(1)
```

---

## 2️⃣ 如何启用深度图像可视化

### 方法 1：训练时自动显示

如果启用了深度相机（`use_camera = True`），训练时会自动弹出 OpenCV 窗口显示深度图像。

**配置**：
```python
# configs/tita_parkour_config.py
class depth:
    use_camera = True  # ✅ 启用深度相机
```

**运行训练**：
```bash
# 不要用 --headless，这样才能看到可视化
python train_parkour.py --task=tita_parkour
```

### 方法 2：使用 simple_play 脚本

如果只想可视化已训练的模型：

```bash
python simple_play.py --task=tita_constraint
```

---

## 3️⃣ 可视化效果

### 深度图像窗口

**窗口名称**：`"Depth Image"`

**显示内容**：
- 黑色区域：远处（2米以上）
- 白色区域：近处（0米附近）
- 灰色区域：中等距离

**分辨率**：`87×58` 像素（配置中的 `resized`）

**更新频率**：每帧更新（`cv2.waitKey(1)`）

### 示例效果

```
深度图像窗口：
┌────────────────────────────┐
│  ████████      ░░░░░░░░    │  ← 前方有障碍物（白色）
│  ████████      ░░░░░░░░    │
│  ████████      ░░░░░░░░    │
│                            │  ← 中间是平地（灰色）
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
└────────────────────────────┘
```

---

## 4️⃣ 代码详解

### 完整的可视化代码

```python
# envs/legged_robot.py

def _draw_debug_vis(self):
    """绘制调试可视化"""
    # ...前面是地形高度点的可视化...
    
    # ========== 深度图像可视化 ==========
    if self.cfg.depth.use_camera:
        # 创建窗口（如果不存在）
        window_name = "Depth Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 获取深度图像
        # self.lookat_id: 当前关注的环境ID（通常是相机跟随的机器人）
        # self.depth_buffer: [num_envs, buffer_len, H, W] 深度图像缓冲
        # [-1]: 取最新的一帧
        depth_img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
        
        # 归一化到 [0, 1]（原始范围是 [-0.5, 0.5]）
        depth_img = depth_img + 0.5
        
        # 显示图像
        cv2.imshow("Depth Image", depth_img)
        
        # 等待1ms（允许窗口更新）
        cv2.waitKey(1)
```

### 深度图像的数据流

```
1. 相机采集
   ↓
   self.gym.get_camera_image_gpu_tensor()
   
2. 处理流程
   ↓
   process_depth_image():
     - crop_depth_image()      # 裁剪边缘
     - 添加噪声               # dis_noise
     - clip 到范围            # [near_clip, far_clip]
     - resize                 # 调整大小
     - normalize_depth_image() # 归一化到 [-0.5, 0.5]
   
3. 存储
   ↓
   self.depth_buffer[env_id, frame_id] = processed_image
   
4. 可视化
   ↓
   cv2.imshow("Depth Image", depth_buffer[lookat_id, -1] + 0.5)
```

---

## 5️⃣ 自定义可视化

### 增强 1：添加颜色映射

让深度图像更直观（使用彩色）：

```python
# 在 _draw_debug_vis 方法中修改
if self.cfg.depth.use_camera:
    window_name = "Depth Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 获取深度图像
    depth_img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
    depth_img = depth_img + 0.5
    
    # ✅ 应用彩色映射（热力图）
    depth_img_uint8 = (depth_img * 255).astype('uint8')
    depth_img_colored = cv2.applyColorMap(depth_img_uint8, cv2.COLORMAP_JET)
    
    # 显示彩色深度图
    cv2.imshow("Depth Image (Color)", depth_img_colored)
    cv2.waitKey(1)
```

**效果**：
- 蓝色 = 远处
- 绿色 = 中等距离
- 红色 = 近处（障碍物）

### 增强 2：同时显示多个环境

查看多个机器人的视角：

```python
if self.cfg.depth.use_camera:
    # 显示前 4 个环境的深度图像
    for i in range(min(4, self.num_envs)):
        depth_img = self.depth_buffer[i, -1].cpu().numpy() + 0.5
        cv2.imshow(f"Depth Image - Env {i}", depth_img)
    cv2.waitKey(1)
```

### 增强 3：保存深度图像

保存特定时刻的深度图像：

```python
if self.cfg.depth.use_camera:
    depth_img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
    depth_img = (depth_img + 0.5) * 255
    
    # ✅ 保存图像
    if self.common_step_counter % 1000 == 0:  # 每 1000 步保存一次
        filename = f"depth_image_step_{self.common_step_counter}.png"
        cv2.imwrite(filename, depth_img.astype('uint8'))
        print(f"Saved depth image: {filename}")
    
    cv2.imshow("Depth Image", depth_img / 255.0)
    cv2.waitKey(1)
```

### 增强 4：添加障碍物检测标记

在深度图像上标注检测到的障碍物：

```python
if self.cfg.depth.use_camera:
    depth_img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
    depth_img = depth_img + 0.5
    
    # 转换为 BGR 格式（可以画彩色标记）
    depth_img_bgr = cv2.cvtColor(
        (depth_img * 255).astype('uint8'), 
        cv2.COLOR_GRAY2BGR
    )
    
    # ✅ 标注障碍物区域
    obstacle_mask = depth_img > 0.7  # 近距离区域
    depth_img_bgr[obstacle_mask] = [0, 0, 255]  # 红色标记
    
    # 添加文字说明
    cv2.putText(
        depth_img_bgr, 
        f"Step: {self.common_step_counter}", 
        (10, 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 255, 0), 
        1
    )
    
    cv2.imshow("Depth Image with Detection", depth_img_bgr)
    cv2.waitKey(1)
```

---

## 6️⃣ 完整示例：增强可视化脚本

创建一个新文件 `visualize_depth.py`：

```python
"""
深度图像可视化增强脚本
用法：python visualize_depth.py --task=tita_parkour
"""

import cv2
import numpy as np

# 在 legged_robot.py 中添加这个方法
def _draw_debug_vis_enhanced(self):
    """增强的深度图像可视化"""
    
    if self.cfg.depth.use_camera:
        # 获取深度图像
        depth_img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
        depth_img_normalized = depth_img + 0.5  # [0, 1]
        
        # 创建多个窗口
        
        # 窗口 1：原始灰度图
        cv2.namedWindow("Depth - Grayscale", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth - Grayscale", depth_img_normalized)
        
        # 窗口 2：彩色热力图
        depth_uint8 = (depth_img_normalized * 255).astype('uint8')
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        cv2.namedWindow("Depth - Heatmap", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth - Heatmap", depth_colored)
        
        # 窗口 3：障碍物检测
        obstacle_mask = depth_img_normalized > 0.6
        depth_with_detection = depth_colored.copy()
        depth_with_detection[obstacle_mask] = [0, 0, 255]  # 红色标记
        
        # 添加信息文字
        info_text = [
            f"Step: {self.common_step_counter}",
            f"Env: {self.lookat_id}",
            f"Obstacles: {obstacle_mask.sum()}",
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(
                depth_with_detection,
                text,
                (10, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        cv2.namedWindow("Depth - Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth - Detection", depth_with_detection)
        
        # 窗口 4：直方图
        hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
        hist = cv2.calcHist([depth_uint8], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
        
        for i in range(256):
            cv2.line(
                hist_img,
                (i, 200),
                (i, 200 - int(hist[i])),
                (255, 255, 255),
                1
            )
        
        cv2.namedWindow("Depth - Histogram", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth - Histogram", hist_img)
        
        # 更新所有窗口
        cv2.waitKey(1)
```

---

## 7️⃣ 快速使用指南

### 步骤 1：确保 OpenCV 已安装

```bash
# 检查 OpenCV
python -c "import cv2; print(cv2.__version__)"

# 如果没安装，安装 OpenCV
pip install opencv-python
```

### 步骤 2：启用深度相机

```python
# configs/tita_parkour_config.py
class depth:
    use_camera = True  # ✅ 必须启用
```

### 步骤 3：运行训练（不使用 headless）

```bash
# ✅ 正确：会显示可视化窗口
python train_parkour.py --task=tita_parkour

# ❌ 错误：headless 模式不会显示窗口
python train_parkour.py --task=tita_parkour --headless
```

### 步骤 4：查看深度图像窗口

训练开始后，会自动弹出 **"Depth Image"** 窗口，实时显示深度图像。

**窗口位置**：
- 通常在 Isaac Gym 主窗口旁边
- 可以拖动调整位置
- 可以调整窗口大小（`cv2.WINDOW_NORMAL`）

---

## 8️⃣ 常见问题

### Q1: 窗口没有弹出？

**原因 1**：使用了 `--headless` 模式
```bash
# ❌ 错误
python train_parkour.py --task=tita_parkour --headless

# ✅ 正确
python train_parkour.py --task=tita_parkour
```

**原因 2**：深度相机未启用
```python
# 检查配置
class depth:
    use_camera = True  # 必须是 True
```

**原因 3**：OpenCV 显示问题（WSL/远程服务器）
```bash
# 如果在 WSL 或远程服务器，需要 X11 转发
export DISPLAY=:0
```

### Q2: 图像太小/太大？

调整窗口大小：
```python
# 在代码中修改
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 可调整大小
cv2.resizeWindow(window_name, 800, 600)  # 设置窗口大小
```

### Q3: 如何保存深度图像？

```python
# 在 _draw_debug_vis 中添加
if self.common_step_counter % 500 == 0:
    depth_img = self.depth_buffer[self.lookat_id, -1].cpu().numpy()
    depth_img = ((depth_img + 0.5) * 255).astype('uint8')
    cv2.imwrite(f"depth_{self.common_step_counter}.png", depth_img)
```

### Q4: 如何切换查看不同环境？

```python
# 使用键盘切换 lookat_id
key = cv2.waitKey(1)
if key == ord('n'):  # 按 'n' 切换到下一个环境
    self.lookat_id = (self.lookat_id + 1) % self.num_envs
elif key == ord('p'):  # 按 'p' 切换到上一个环境
    self.lookat_id = (self.lookat_id - 1) % self.num_envs
```

### Q5: 在远程服务器上如何可视化？

**方法 1**：保存图像，然后下载查看
```python
# 每 N 步保存一次
if self.common_step_counter % 100 == 0:
    cv2.imwrite(f"depth/depth_{self.common_step_counter}.png", depth_img)
```

**方法 2**：使用 TensorBoard
```python
# 在训练循环中
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if self.common_step_counter % 100 == 0:
    depth_img = self.depth_buffer[self.lookat_id, -1]
    writer.add_image('Depth/Image', depth_img, self.common_step_counter)
```

---

## 9️⃣ 性能注意事项

### 可视化的性能开销

| 操作 | CPU 时间 | 影响 |
|------|---------|------|
| `cv2.imshow()` | ~1-2 ms | 很小 |
| `cv2.applyColorMap()` | ~0.5 ms | 可忽略 |
| `cv2.imwrite()` | ~5-10 ms | 中等（频繁保存会慢） |

### 优化建议

1. **降低更新频率**：
```python
# 不是每帧都显示
if self.common_step_counter % 10 == 0:  # 每 10 帧显示一次
    cv2.imshow("Depth Image", depth_img)
    cv2.waitKey(1)
```

2. **训练时关闭可视化**：
```python
# 配置中添加开关
class depth:
    use_camera = True
    visualize = False  # 训练时关闭可视化，测试时开启
```

3. **使用更小的窗口**：
```python
# 缩小显示尺寸（不改变实际分辨率）
depth_img_small = cv2.resize(depth_img, (174, 116))  # 2x 缩放
cv2.imshow("Depth Image", depth_img_small)
```

---

## 🔟 总结

### ✅ 核心功能

原始代码已经集成了 OpenCV 可视化：
- **位置**：`envs/legged_robot.py` 第 1030-1033 行
- **触发**：`use_camera = True` + 非 headless 模式
- **窗口**：实时显示 87×58 的深度图像
- **更新**：每帧自动更新

### 🎨 可定制性

可以轻松添加：
- ✅ 彩色热力图
- ✅ 障碍物检测标记
- ✅ 多环境同时显示
- ✅ 保存图像到文件
- ✅ 添加信息文字
- ✅ 直方图分析

### 🚀 快速开始

```bash
# 1. 确保 OpenCV 安装
pip install opencv-python

# 2. 启用深度相机
# configs/tita_parkour_config.py: use_camera = True

# 3. 运行（不用 headless）
python train_parkour.py --task=tita_parkour

# 4. 查看弹出的 "Depth Image" 窗口
```

---

**最终答案**：是的，原始代码已经用 **OpenCV** 实时显示深度图像了！只需启用深度相机并在非 headless 模式下运行即可看到。🎉
