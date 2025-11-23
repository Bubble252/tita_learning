# 基于预训练模型的跑酷微调指南

## ✅ 简短回答

**完全可以！** 你可以基于原来训练好的 `tita_constraint` 模型（如 `tita_example_10000.pt`）来微调跑酷任务。这样可以：
- ✅ **加速训练**：从已有的行走能力开始，不用从头学
- ✅ **提高稳定性**：基础运动已经掌握，只需学习跳跃
- ✅ **节省时间**：可能减少 30-50% 的训练时间

---

## 1️⃣ 微调的两种策略

### 策略 A：完全微调（推荐用于跑酷）

**适用场景**：新任务与旧任务相似度高（行走 → 跑酷）

```python
# 在 tita_parkour_config.py 中修改
class runner(TitaConstraintRoughCfgPPO.runner):
    run_name = 'parkour_finetune_from_10000'
    experiment_name = 'tita_parkour'
    
    max_iterations = 37000
    
    # ✅ 启用恢复训练
    resume = True
    resume_path = 'tita_example_10000.pt'  # 你的预训练模型
    
    # ⚠️ 学习率建议降低（微调用）
    # 在 algorithm 类中设置
```

**优点**：
- 保留所有已学能力（行走、平衡、转向）
- 在此基础上学习新技能（跳跃、障碍识别）
- 训练更稳定

**缺点**：
- 如果预训练模型过拟合旧任务，可能需要更多迭代打破旧习惯

### 策略 B：部分冻结（高级用法）

**适用场景**：想保持某些层不变，只训练新加的层

```python
# 需要修改训练脚本，冻结部分网络
# 这个策略比较复杂，通常策略 A 就够用了
```

---

## 2️⃣ 快速开始：三步微调

### 步骤 1：修改配置文件

打开 `configs/tita_parkour_config.py`，找到 `runner` 类：

```python
class runner(TitaConstraintRoughCfgPPO.runner):
    run_name = 'parkour_finetune_from_10000'  # 改名，避免覆盖
    experiment_name = 'tita_parkour'
    
    # 使用跑酷专用策略和运行器
    policy_class_name = 'ActorCriticBarlowTwins'
    runner_class_name = 'OnConstraintPolicyRunner'
    algorithm_class_name = 'NP3O'
    
    # 训练配置
    max_iterations = 37000
    num_steps_per_env = 24
    
    # ✅ 关键：启用恢复并指定路径
    resume = True
    resume_path = 'tita_example_10000.pt'  # 你的预训练模型文件名
    
    # 可选：指定加载哪个运行
    load_run = None  # 如果 resume_path 只是文件名，会自动搜索
    checkpoint = -1  # -1 表示最新，也可以指定具体迭代数
    
    # 检查点保存
    save_interval = 500
```

### 步骤 2：确认模型文件位置

确保你的预训练模型在正确的位置：

```bash
# 选项 1：放在原训练的日志目录中
logs/tita_constraint/
├── Dec15_10-30-45_test_barlowtwins_feetcontact/
│   ├── model_10000.pt  ← 你的模型
│   ├── model_20000.pt
│   └── model_30000.pt

# 选项 2：直接放在项目根目录
tita_rl/
├── tita_example_10000.pt  ← 你的模型
├── train_parkour.py
└── ...
```

### 步骤 3：开始微调训练

```bash
# 直接运行即可，会自动加载预训练模型
python train_parkour.py --task=tita_parkour --headless

# 如果需要手动指定检查点
python train_parkour.py \
    --task=tita_parkour \
    --headless \
    --resume \
    --load_run=test_barlowtwins_feetcontact \
    --checkpoint=10000
```

---

## 3️⃣ 配置参数详解

### 关键参数说明

```python
class runner:
    # ========== 恢复训练相关 ==========
    resume = True                        # 是否从检查点恢复
    resume_path = 'model_10000.pt'      # 模型文件名
    load_run = None                      # 运行名称（可选）
    checkpoint = -1                      # 检查点编号（-1=最新）
    
    # ========== 如果要调整学习率（推荐）==========
    # 需要在 algorithm 类中设置
```

### 学习率调整（推荐）

微调时通常使用**更小的学习率**，避免破坏已学知识：

```python
class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        # 原始学习率：1e-3
        # 微调建议：原始的 1/3 到 1/10
        learning_rate = 3.e-4  # 或 1.e-4（更保守）
        
        entropy_coef = 0.01
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
```

---

## 4️⃣ 微调策略对比

### 策略对比表

| 微调方式 | 学习率 | 迭代次数 | 适用场景 | 预期效果 |
|---------|--------|---------|---------|---------|
| **激进微调** | 1e-3（不变） | 37000 | 任务差异大 | 快速适应新任务，可能遗忘部分旧技能 |
| **保守微调** | 3e-4（1/3） | 30000 | 任务相似 | 保留旧技能，平滑学习新技能 ⭐推荐 |
| **极保守微调** | 1e-4（1/10） | 25000 | 几乎相同 | 最大程度保留，微小调整 |

### 推荐配置（保守微调）

```python
class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        learning_rate = 3.e-4  # 降低到 1/3
        entropy_coef = 0.01
        
    class runner(TitaConstraintRoughCfgPPO.runner):
        run_name = 'parkour_finetune_conservative'
        experiment_name = 'tita_parkour'
        
        max_iterations = 30000  # 可以减少迭代（因为有基础）
        resume = True
        resume_path = 'tita_example_10000.pt'
```

---

## 5️⃣ 课程学习 + 微调的最佳实践

### 方案 A：保守课程（推荐）

**思路**：既然已经会走路了，可以跳过阶段 1，直接从障碍训练开始

修改 `utils/parkour_curriculum.py`：

```python
class ParkourCurriculum:
    def __init__(self, skip_stage1=False):
        self.skip_stage1 = skip_stage1
        
        if skip_stage1:
            # 跳过平地行走，直接从障碍开始
            self.stages = [
                # 阶段 1：小障碍（原阶段2）
                {
                    'name': 'stage_1_small_obstacles',
                    'iterations': 10000,
                    'terrain_level': 3,
                    'obstacle_height': 0.10,
                    'rewards': {
                        'tracking_lin_vel': 1.0,
                        'obstacle_clearance': 1.0,
                        'jump_timing': 0.5,
                        'collision': -1.0,
                        # ...
                    }
                },
                # 阶段 2：完美跑酷（原阶段3）
                {
                    'name': 'stage_2_parkour_mastery',
                    'iterations': 20000,
                    'terrain_level': 7,
                    'obstacle_height': 0.15,
                    'rewards': {
                        'obstacle_clearance': 2.0,
                        'jump_timing': 1.5,
                        'landing_stability': 1.0,
                        'collision': -5.0,
                        # ...
                    }
                }
            ]
        else:
            # 原始三阶段
            self.stages = [...]  # 保持不变
```

在 `train_parkour.py` 中使用：

```python
# 如果从预训练模型开始，跳过阶段1
if train_cfg.runner.resume:
    curriculum = ParkourCurriculum(skip_stage1=True)
    print("🎓 检测到预训练模型，跳过基础行走阶段")
else:
    curriculum = ParkourCurriculum(skip_stage1=False)
    print("🎓 从头训练，使用完整三阶段课程")
```

### 方案 B：激进课程

直接进入最难阶段：

```python
# 单阶段，直接跑酷
self.stages = [
    {
        'name': 'stage_1_direct_parkour',
        'iterations': 25000,  # 总迭代减少
        'terrain_level': 7,
        'obstacle_height': 0.15,
        'rewards': {
            'obstacle_clearance': 2.0,
            'jump_timing': 1.5,
            'landing_stability': 1.0,
            # ...
        }
    }
]
```

---

## 6️⃣ 完整示例配置

### 示例 1：保守微调（最稳妥）

```python
# configs/tita_parkour_config.py

class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        learning_rate = 3.e-4  # 降低学习率
        entropy_coef = 0.01
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
    
    class runner(TitaConstraintRoughCfgPPO.runner):
        run_name = 'parkour_finetune_from_10k'
        experiment_name = 'tita_parkour'
        
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        
        max_iterations = 30000  # 减少到 30000
        num_steps_per_env = 24
        
        # ✅ 启用微调
        resume = True
        resume_path = 'tita_example_10000.pt'
        load_run = None
        checkpoint = -1
        
        save_interval = 500
```

### 示例 2：激进微调（快速适应）

```python
class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        learning_rate = 1.e-3  # 保持原学习率
        entropy_coef = 0.02    # 增加探索
        max_grad_norm = 0.01
    
    class runner(TitaConstraintRoughCfgPPO.runner):
        run_name = 'parkour_finetune_aggressive'
        experiment_name = 'tita_parkour'
        
        max_iterations = 20000  # 大幅减少
        
        resume = True
        resume_path = 'tita_example_10000.pt'
```

---

## 7️⃣ 训练命令

### 基础命令

```bash
# 最简单：直接运行（配置文件中已设置 resume=True）
python train_parkour.py --task=tita_parkour --headless

# 显式指定恢复
python train_parkour.py \
    --task=tita_parkour \
    --headless \
    --resume

# 指定具体检查点
python train_parkour.py \
    --task=tita_parkour \
    --headless \
    --resume \
    --load_run=test_barlowtwins_feetcontact \
    --checkpoint=10000
```

### 高级选项

```bash
# 从不同的运行中加载
python train_parkour.py \
    --task=tita_parkour \
    --headless \
    --resume \
    --load_run=previous_experiment \
    --checkpoint=20000

# 使用 GPU 1（如果有多个GPU）
python train_parkour.py \
    --task=tita_parkour \
    --headless \
    --resume \
    --rl_device=cuda:1
```

---

## 8️⃣ 监控与调试

### 查看加载日志

训练开始时会显示：

```
📝 注册跑酷任务...
📦 创建训练环境...
🧠 创建策略网络...

Loading model from: logs/tita_constraint/.../model_10000.pt
✅ 成功加载预训练模型！
  - Actor 网络: 已加载
  - Critic 网络: 已加载
  - 优化器状态: 已加载
  - 迭代次数: 从 10000 继续

📚 初始化课程学习管理器...
🚀 开始训练...
```

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=logs/tita_parkour

# 对比微调前后
tensorboard --logdir=logs --port=6006
```

**关键指标**：
- `Train/mean_reward`: 应该从较高值开始（因为已经会走）
- `Train/mean_episode_length`: 初期应该较长（不容易摔倒）
- `Policy/learning_rate`: 确认学习率是否正确
- `Curriculum/stage_index`: 课程学习阶段

### 常见问题排查

#### 问题 1：加载失败

```
Error: Cannot load model from tita_example_10000.pt
```

**解决方法**：
```bash
# 检查文件是否存在
ls -lh tita_example_10000.pt
ls -lh logs/tita_constraint/*/model_*.pt

# 使用绝对路径
resume_path = '/home/bubble/桌面/tita/tita_rl/tita_example_10000.pt'
```

#### 问题 2：模型结构不匹配

```
Error: Size mismatch for actor.xxx
```

**原因**：跑酷配置的观测维度与原模型不同（添加了深度特征）

**解决方法**：
```python
# 在 tita_parkour_config.py 中
class env(TitaConstraintRoughCfg.env):
    # ⚠️ 确保观测维度与预训练模型一致
    # 如果原模型没有深度特征，这里也不要加
    n_depth_features = 0  # 改为 0
    
    num_observations = (
        n_proprio + 
        n_scan + 
        history_len * n_proprio + 
        n_priv_latent
        # 不加 n_depth_features
    )
```

或者使用部分加载（高级）：
```python
# 在 train_parkour.py 中修改加载逻辑
# 只加载兼容的部分，忽略不匹配的层
```

#### 问题 3：训练不稳定

初期奖励大幅下降：

**原因**：新任务（跑酷）与旧任务（平地行走）差异大

**解决方法**：
1. 降低学习率到 `1e-4` 或 `3e-4`
2. 使用渐进课程（从阶段 2 开始，不要直接跳到阶段 3）
3. 增加 `entropy_coef` 鼓励探索

---

## 9️⃣ 预期效果对比

### 从头训练 vs 微调

| 指标 | 从头训练 | 微调训练 |
|------|---------|---------|
| **训练时间** | ~11 小时 (37k iter) | ~6-8 小时 (25-30k iter) |
| **初期奖励** | 5-10 | 40-60（已经会走） |
| **稳定时间** | 5000-8000 iter | 2000-4000 iter |
| **最终性能** | 150-200 | 150-200（相同） |
| **风险** | 低（从零开始） | 中（可能破坏旧技能） |

### 学习曲线对比

```
从头训练：
奖励 │                  ╱╱╱╱  最终性能
     │              ╱╱╱
     │          ╱╱╱
     │      ╱╱╱
     │  ╱╱╱
  0  └─────────────────────▶ 迭代
     0    10k   20k   30k

微调训练：
奖励 │              ╱╱╱╱  最终性能
     │          ╱╱╱
     │      ╱╱╱
     │  ╱╱╱                  已有基础！
 50  │╱╱                      
  0  └─────────────────────▶ 迭代
     0    10k   20k   30k
```

---

## 🔟 总结与建议

### ✅ 推荐方案（保守微调）

```python
# 1. 修改 tita_parkour_config.py
class algorithm:
    learning_rate = 3.e-4  # 降低到 1/3

class runner:
    run_name = 'parkour_finetune_conservative'
    max_iterations = 30000  # 减少 7000 次迭代
    resume = True
    resume_path = 'tita_example_10000.pt'

# 2. 修改课程学习（可选）
# 在 train_parkour.py 中检测 resume，跳过阶段 1

# 3. 运行训练
python train_parkour.py --task=tita_parkour --headless
```

### 📊 决策流程图

```
有预训练模型？
    ├─ 是 → 任务相似度高？（行走→跑酷）
    │         ├─ 是 → ✅ 使用微调（推荐）
    │         │         ├─ 保守：lr=3e-4, 30k iter
    │         │         └─ 激进：lr=1e-3, 20k iter
    │         └─ 否 → 从头训练
    └─ 否 → 从头训练
```

### 🎯 最佳实践

1. **优先尝试微调**：如果有预训练模型，先试微调
2. **降低学习率**：微调时用 1/3 到 1/10 的学习率
3. **监控奖励**：如果初期奖励大幅下降，说明学习率太高
4. **课程调整**：可以跳过基础阶段，直接从障碍训练开始
5. **保存检查点**：每 500 次迭代保存，方便回退

### 🚀 快速开始

```bash
# 1. 确保模型文件存在
ls tita_example_10000.pt

# 2. 修改配置（3行）
vim configs/tita_parkour_config.py
# 设置: resume=True, resume_path='tita_example_10000.pt', learning_rate=3e-4

# 3. 开始训练
python train_parkour.py --task=tita_parkour --headless

# 4. 监控进度
tensorboard --logdir=logs/tita_parkour
```

---

**结论**：不仅可以基于预训练模型微调，而且**强烈推荐**这样做！这样可以利用已学的行走能力，只需专注学习跑酷的新技能（跳跃、障碍识别），训练时间可减少 30-50%！🎉
