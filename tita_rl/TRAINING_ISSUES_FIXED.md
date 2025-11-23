# 🐛 训练启动问题修复记录

## 问题 1：导入顺序错误 ❌

### 错误信息
```
ImportError: PyTorch was imported before isaacgym modules. 
Please import torch after isaacgym modules.
```

### 原因
Isaac Gym **必须在 PyTorch 之前导入**，这是 Isaac Gym 的硬性要求。

原始代码（错误）：
```python
import numpy as np
import os
import torch  # ❌ torch 在第 16 行
from datetime import datetime

# ...

import isaacgym  # ❌ isaacgym 在第 26 行（太晚了）
```

### 解决方案 ✅

```python
import numpy as np
import os
from datetime import datetime

# ========== 重要：Isaac Gym 必须在 PyTorch 之前导入 ==========
from global_config import ROOT_DIR
import isaacgym  # ✅ 先导入 isaacgym
from utils.helpers import get_args
from utils.task_registry import task_registry

# 现在可以安全导入 PyTorch 相关模块
import torch  # ✅ 后导入 torch

# 导入配置和环境
from configs.tita_parkour_config import TitaParkourCfg, TitaParkourCfgPPO
from envs.parkour_robot import ParkourRobot
from utils.parkour_curriculum import ParkourCurriculum
```

### 参考：原始 train.py 的导入顺序

```python
# train.py（正确示例）
import numpy as np
import os
from datetime import datetime
from configs.tita_constraint_config import ...
from envs.no_constrains_legged_robot import Tita

from global_config import ROOT_DIR, ENVS_DIR
import isaacgym  # ✅ 在导入任何会间接引入 torch 的模块之前
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry
```

---

## 问题 2：观测维度不匹配导致段错误 ❌

### 错误信息
```
段错误 (核心已转储)
Segmentation Fault (core dumped)
```

### 原因

1. **配置中添加了深度特征维度**：
```python
# tita_parkour_config.py
class env:
    n_depth_features = 5  # ❌ 新增 5 维深度特征
    num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_depth_features
```

2. **ParkourRobot 修改了观测维度**：
```python
# parkour_robot.py
def compute_observations(self):
    super().compute_observations()
    if self.cfg.depth.use_camera:
        depth_features = self._extract_depth_features()  # 5 维
        self.obs_buf = torch.cat([self.obs_buf, depth_features], dim=-1)  # ❌ 增加了 5 维
```

3. **预训练模型期望固定维度**：
   - 预训练模型 `model_11700.pt` 的网络结构是固定的
   - Actor 网络输入层：`num_observations = 585`（原始维度）
   - 如果添加 5 维深度特征：`585 + 5 = 590`
   - **维度不匹配** → 加载模型时崩溃

### 解决方案 ✅

**方案 A：不添加深度特征到观测空间（推荐，用于微调）**

```python
# configs/tita_parkour_config.py
class env:
    n_scan = 187
    n_priv_latent = 4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
    n_proprio = 33
    history_len = 10
    
    # ✅ 保持与原始配置相同的维度
    num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent
    # = 33 + 187 + 10*33 + 36 = 586
```

```python
# envs/parkour_robot.py
def compute_observations(self):
    """直接使用父类的观测，不添加深度特征"""
    super().compute_observations()
    
    # ✅ 深度信息仅用于内部（奖励计算），不添加到观测
    # 这样可以使用预训练模型
```

**优点**：
- ✅ 可以使用预训练模型（微调）
- ✅ 训练更快（有基础）
- ✅ 网络结构不变

**缺点**：
- ⚠️ 策略不能直接"看到"深度信息
- ⚠️ 只能通过地形高度扫描感知环境

---

**方案 B：添加深度特征，从头训练（不推荐）**

```python
# configs/tita_parkour_config.py
class env:
    n_depth_features = 5
    num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_depth_features
```

```python
# envs/parkour_robot.py
def compute_observations(self):
    super().compute_observations()
    if self.cfg.depth.use_camera:
        depth_features = self._extract_depth_features()
        self.obs_buf = torch.cat([self.obs_buf, depth_features], dim=-1)
```

```python
# configs/tita_parkour_config.py
class runner:
    resume = False  # ❌ 不能使用预训练模型
    resume_path = None
    max_iterations = 50000  # 需要更多迭代（从头训练）
```

**优点**：
- ✅ 策略可以直接使用深度信息
- ✅ 可能表现更好（理论上）

**缺点**：
- ❌ 不能使用预训练模型
- ❌ 训练时间更长（4-8小时 → 10-20小时）
- ❌ 不稳定（从零开始）

---

## 最终配置（已修复） ✅

### 文件 1：`train_parkour.py`

```python
# ✅ 正确的导入顺序
import numpy as np
import os
from datetime import datetime

# Isaac Gym 必须在 PyTorch 之前
from global_config import ROOT_DIR
import isaacgym
from utils.helpers import get_args
from utils.task_registry import task_registry

# 现在可以导入 PyTorch
import torch

# 其他导入
from configs.tita_parkour_config import TitaParkourCfg, TitaParkourCfgPPO
from envs.parkour_robot import ParkourRobot
from utils.parkour_curriculum import ParkourCurriculum
```

### 文件 2：`configs/tita_parkour_config.py`

```python
class env(TitaConstraintRoughCfg.env):
    num_envs = 2048
    
    # ✅ 观测维度与原始配置相同
    n_scan = 187
    n_priv_latent = 4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
    n_proprio = 33
    history_len = 10
    
    # ✅ 不添加 n_depth_features
    num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent
```

### 文件 3：`envs/parkour_robot.py`

```python
def compute_observations(self):
    """直接使用父类观测，保持维度一致"""
    super().compute_observations()
    
    # ✅ 不修改 obs_buf，深度信息仅用于奖励计算
```

---

## 训练启动验证 ✅

### 成功输出

```bash
$ python train_parkour.py --task=tita_parkour --headless

Importing module 'gym_38'
Setting GYM_USD_PLUG_INFO_PATH...
PyTorch version 2.4.1+cu121
Device count 1
Loading extension module gymtorch...
📝 注册跑酷任务...

======================================================================
🎯 TITA 跑酷训练（带课程学习）
======================================================================

📦 创建训练环境...
Setting seed: 1
Not connected to PVD
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
Creating env...
✅ 环境创建成功！
✅ ParkourRobot initialized with parkour-specific reward functions
```

### 检查训练进程

```bash
# 查看后台进程
$ ps aux | grep train_parkour

# 查看 GPU 使用
$ nvidia-smi

# 查看日志
$ tail -f logs/tita_parkour/parkour_with_curriculum/<时间戳>/training.log
```

---

## 关键要点总结 🎯

### 1. 导入顺序规则

```python
# ✅ 正确顺序
import isaacgym  # 第一
import torch     # 第二

# ❌ 错误顺序
import torch      # ❌ 先导入 torch
import isaacgym   # ❌ 会报错
```

### 2. 观测维度规则

**使用预训练模型（微调）**：
- ✅ 观测维度必须与预训练模型完全一致
- ✅ 不能添加新的观测特征
- ✅ 可以修改：奖励权重、地形配置、训练超参数

**从头训练**：
- ✅ 可以任意修改观测维度
- ✅ 可以添加新特征（如深度特征）
- ❌ 不能使用预训练模型

### 3. 深度相机使用方式

**当前方案（推荐）**：
```python
# ✅ 深度相机启用
class depth:
    use_camera = True

# ✅ 深度信息用于内部计算（奖励函数）
def _reward_obstacle_clearance(self):
    obstacles = self._detect_obstacles_ahead()  # 使用 depth_buffer
    return reward

# ✅ 不添加到观测空间
def compute_observations(self):
    super().compute_observations()  # 保持原始维度
```

### 4. 配置一致性检查

| 配置项 | tita_constraint | tita_parkour | 状态 |
|--------|----------------|--------------|------|
| `num_envs` | 4096 | 2048 | ✅ 可修改 |
| `num_observations` | 586 | 586 | ✅ 必须相同 |
| `depth.use_camera` | False | True | ✅ 可修改 |
| `terrain.curriculum` | True | True | ✅ 可修改 |
| `algorithm.learning_rate` | 1e-3 | 4e-4 | ✅ 可修改 |
| `resume` | False | True | ✅ 可修改 |

---

## 常见错误对照表

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `PyTorch was imported before isaacgym` | 导入顺序错误 | 先导入 isaacgym，后导入 torch |
| `Segmentation Fault` | 观测维度不匹配 | 保持 num_observations 与预训练模型一致 |
| `RuntimeError: size mismatch` | 网络输入维度错误 | 检查 obs_buf 维度是否正确 |
| `CUDA out of memory` | GPU 内存不足 | 降低 num_envs（2048→1024） |
| `ModuleNotFoundError: isaacgym` | Isaac Gym 未安装 | 重新安装 Isaac Gym |

---

## 下一步

✅ **训练已启动！**

监控训练进度：
```bash
# TensorBoard
tensorboard --logdir=logs/tita_parkour --port=6006

# 查看实时日志
tail -f logs/tita_parkour/parkour_with_curriculum/*/training.log

# GPU 使用情况
watch -n 1 nvidia-smi
```

训练完成后（约 4-8 小时）：
```bash
# 测试模型
python test_parkour.py --task=tita_parkour --checkpoint=31000

# 导出 ONNX
python export_policy_as_onnx.py --checkpoint=logs/.../model_31000.pt
```

🎉 祝训练成功！
