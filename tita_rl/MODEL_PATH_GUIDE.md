# 预训练模型路径配置指南

## 📍 你的模型文件位置

根据你的项目结构，发现了以下 `.pt` 模型文件：

### ✅ 项目根目录（推荐使用这些）

```bash
/home/bubble/桌面/tita/tita_rl/
├── tita_example_10000.pt      # ✅ 13 MB，9月30日
├── model_10000.pt              # ✅ 13 MB，10月1日
├── model_11700.pt              # ✅ 13 MB，10月1日
├── model_29900.pt              # ✅ 13 MB，10月8日
└── model.pt                    # ⚠️  2.3 MB，可能不完整
```

### 📁 logs 目录

```bash
/home/bubble/桌面/tita/tita_rl/logs/tita_constraint/
├── tita_example_10000.pt       # ✅ 备份副本
├── Oct07_21-08-35_test_barlowtwins_feetcontact/
│   ├── model_0.pt
│   └── model_100.pt
├── Oct07_21-12-41_test_barlowtwins_feetcontact/
│   ├── model_0.pt
│   ├── model_100.pt
│   └── model_200.pt
└── ...
```

---

## 🎯 路径配置方式（3种）

### 方式 1：只写文件名（最简单，推荐）⭐

如果模型在**项目根目录**（`tita_rl/`），只需要写文件名：

```python
# configs/tita_parkour_config.py
class runner:
    resume = True
    resume_path = 'tita_example_10000.pt'  # ✅ 只写文件名
```

**系统会自动搜索**：
1. 先在项目根目录找：`/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt`
2. 再在 logs 目录找：`logs/tita_constraint/*/tita_example_10000.pt`

### 方式 2：相对路径

如果模型在 logs 子目录：

```python
class runner:
    resume = True
    # 相对于项目根目录的路径
    resume_path = 'logs/tita_constraint/Oct07_21-08-35_test_barlowtwins_feetcontact/model_100.pt'
```

### 方式 3：绝对路径（最保险）

```python
class runner:
    resume = True
    # 完整的绝对路径
    resume_path = '/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt'
```

---

## 📝 推荐配置

### 配置 1：使用 tita_example_10000.pt（推荐）

```python
# configs/tita_parkour_config.py

class runner(TitaConstraintRoughCfgPPO.runner):
    run_name = 'parkour_finetune_from_10k'
    experiment_name = 'tita_parkour'
    
    # ✅ 方式 1：只写文件名（推荐）
    resume = True
    resume_path = 'tita_example_10000.pt'
    
    # ✅ 方式 3：绝对路径（最保险）
    # resume_path = '/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt'
```

**为什么推荐这个？**
- ✅ 文件大小正常（13 MB）
- ✅ 命名清晰（example，示例模型）
- ✅ 已经在根目录，路径最简单

### 配置 2：使用最新的 model_29900.pt

如果想用训练到 29900 次的模型（最新）：

```python
class runner:
    resume = True
    resume_path = 'model_29900.pt'  # 10月8日的最新模型
```

### 配置 3：使用特定训练运行的模型

如果想用 logs 中特定训练的模型：

```python
class runner:
    resume = True
    load_run = 'Oct07_21-12-41_test_barlowtwins_feetcontact'
    checkpoint = 200  # 或 -1 表示最新
```

---

## 🔍 如何选择使用哪个模型？

### 模型对比

| 模型文件 | 大小 | 日期 | 迭代次数 | 推荐度 |
|---------|------|------|---------|-------|
| `tita_example_10000.pt` | 13 MB | 9月30日 | 10000 | ⭐⭐⭐⭐⭐ |
| `model_10000.pt` | 13 MB | 10月1日 | 10000 | ⭐⭐⭐⭐ |
| `model_11700.pt` | 13 MB | 10月1日 | 11700 | ⭐⭐⭐⭐ |
| `model_29900.pt` | 13 MB | 10月8日 | 29900 | ⭐⭐⭐⭐⭐ |
| `model.pt` | 2.3 MB | 11月12日 | ? | ⚠️ 太小，可能损坏 |

### 推荐使用

**1. 快速开始：`tita_example_10000.pt`** ⭐推荐
- 训练稳定（10000 次迭代）
- 命名清晰（example）
- 文档中的示例都用这个

**2. 最佳性能：`model_29900.pt`** ⭐⭐推荐
- 训练最久（29900 次迭代）
- 可能学得最好
- 但也可能过拟合

**3. 中间选择：`model_11700.pt`**
- 折中方案
- 既有一定训练，又不会过拟合

---

## 🛠️ 完整配置示例

### 示例 1：使用 tita_example_10000.pt（推荐新手）

```python
# configs/tita_parkour_config.py

class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        learning_rate = 3.e-4  # 微调学习率
        entropy_coef = 0.01
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1
    
    class runner(TitaConstraintRoughCfgPPO.runner):
        run_name = 'parkour_finetune_from_example_10k'
        experiment_name = 'tita_parkour'
        
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        
        max_iterations = 30000
        num_steps_per_env = 24
        
        # ✅ 微调配置
        resume = True
        resume_path = 'tita_example_10000.pt'  # 只写文件名
        load_run = None
        checkpoint = -1
        
        save_interval = 500
```

### 示例 2：使用最新模型 model_29900.pt（推荐有经验用户）

```python
class runner(TitaConstraintRoughCfgPPO.runner):
    run_name = 'parkour_finetune_from_29900'
    experiment_name = 'tita_parkour'
    
    max_iterations = 20000  # 可以减少，因为基础更好
    
    # ✅ 使用最新模型
    resume = True
    resume_path = 'model_29900.pt'
```

### 示例 3：使用绝对路径（最保险）

```python
class runner(TitaConstraintRoughCfgPPO.runner):
    run_name = 'parkour_finetune_absolute_path'
    experiment_name = 'tita_parkour'
    
    # ✅ 绝对路径，不会出错
    resume = True
    resume_path = '/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt'
```

---

## 🚀 快速开始（复制粘贴即用）

### 步骤 1：确认文件存在

```bash
cd /home/bubble/桌面/tita/tita_rl
ls -lh tita_example_10000.pt
```

**应该看到**：
```
-rw-rw-r-- 1 bubble bubble 13M 9月 30 20:40 tita_example_10000.pt
```

### 步骤 2：修改配置文件

```bash
vim configs/tita_parkour_config.py
```

找到这两处并修改：

```python
# 第 228 行附近
learning_rate = 3.e-4  # 从 1.e-3 改为 3.e-4

# 第 281 行附近（在 runner 类中）
resume = True                        # 从 False 改为 True
resume_path = 'tita_example_10000.pt'  # 从 None 改为文件名
```

### 步骤 3：开始训练

```bash
cd /home/bubble/桌面/tita/tita_rl
python train_parkour.py --task=tita_parkour --headless
```

**应该看到**：
```
📝 注册跑酷任务...
📦 创建训练环境...
🧠 创建策略网络...

Loading model from: /home/bubble/桌面/tita/tita_rl/tita_example_10000.pt
✅ 成功加载预训练模型！
  - Actor 网络: 已加载
  - Critic 网络: 已加载
  - 优化器状态: 已加载

📚 初始化课程学习管理器...
🚀 开始训练...
```

---

## ❓ 常见问题

### Q1: 如果找不到文件怎么办？

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'tita_example_10000.pt'
```

**解决方法**：

1. 检查文件是否存在：
```bash
ls -lh /home/bubble/桌面/tita/tita_rl/tita_example_10000.pt
```

2. 如果文件存在，使用绝对路径：
```python
resume_path = '/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt'
```

3. 如果文件不存在，使用其他模型：
```python
resume_path = 'model_29900.pt'  # 使用其他模型
```

### Q2: 多个模型怎么选？

**建议**：
- **首次微调**：用 `tita_example_10000.pt`（稳定）
- **追求性能**：用 `model_29900.pt`（最新最好）
- **快速测试**：用 `model_10000.pt`（任何一个 10000 的都行）

### Q3: 绝对路径 vs 相对路径？

| 方式 | 优点 | 缺点 |
|------|------|------|
| **只写文件名** | 简洁，跨平台 | 需要文件在根目录 |
| **相对路径** | 灵活，可以在子目录 | 可能出错 |
| **绝对路径** | 最保险，不会出错 | 不可移植 |

**推荐**：
- 开发阶段：用绝对路径（不会出错）
- 最终版本：改为文件名（可移植）

### Q4: 如何验证路径正确？

在 Python 中测试：

```python
import os

# 测试路径
model_path = '/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt'
print(f"文件存在: {os.path.exists(model_path)}")
print(f"文件大小: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
```

应该输出：
```
文件存在: True
文件大小: 13.0 MB
```

---

## 📊 路径配置总结表

| 配置方式 | 代码示例 | 适用场景 |
|---------|---------|---------|
| **只写文件名** ⭐ | `resume_path = 'tita_example_10000.pt'` | 文件在根目录 |
| **相对路径** | `resume_path = 'logs/tita_constraint/.../model_100.pt'` | 文件在子目录 |
| **绝对路径** | `resume_path = '/home/bubble/.../tita_example_10000.pt'` | 保险起见 |
| **load_run** | `load_run = 'Oct07_21-12-41_...'` | 指定训练运行 |

---

## 💡 最佳实践

1. **文件命名**：统一放在根目录，使用清晰的文件名
2. **路径写法**：开发用绝对路径，部署改为文件名
3. **版本管理**：保存多个检查点，便于回退
4. **文件检查**：训练前先用 `ls` 确认文件存在
5. **备份重要模型**：将好的模型复制一份，防止覆盖

---

**快速答案**：
```python
# 最简单的配置（推荐）
resume = True
resume_path = 'tita_example_10000.pt'  # 文件已经在项目根目录了！
```

文件路径：`/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt` ✅
