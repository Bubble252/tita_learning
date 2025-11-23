# NP3O 算法说明文档

## ✅ 简短回答

**是的！** 跑酷训练依旧使用 **NP3O (Natural Policy Gradient with Proximal Policy Optimization)** 算法。

这是原始 `tita_constraint` 训练就在使用的算法，跑酷训练**完全继承**了这个算法，没有改变。

---

## 1️⃣ 什么是 NP3O？

### 核心定义

**NP3O** 是一个**带约束的强化学习算法**，结合了：
- **PPO (Proximal Policy Optimization)**：稳定的策略优化
- **约束优化 (Constrained Optimization)**：确保机器人安全运动

### 配置中的体现

```python
# configs/tita_parkour_config.py
class runner(TitaConstraintRoughCfgPPO.runner):
    policy_class_name = 'ActorCriticBarlowTwins'  # 策略网络
    runner_class_name = 'OnConstraintPolicyRunner'  # 约束运行器
    algorithm_class_name = 'NP3O'  # ✅ NP3O 算法
```

---

## 2️⃣ NP3O vs PPO 对比

### 核心差异

| 特性 | 标准 PPO | NP3O |
|------|---------|------|
| **优化目标** | 最大化奖励 | 最大化奖励 + 满足约束 |
| **约束处理** | 无 | 有（通过 Cost Critic） |
| **安全保证** | 无明确保证 | 有软约束保证 |
| **适用场景** | 一般任务 | 机器人控制（需要安全性） |

### 数学表达

**标准 PPO**：
```
最大化: E[Σ reward_t]
```

**NP3O**：
```
最大化: E[Σ reward_t]
满足约束: E[Σ cost_t] ≤ d  (期望约束值)
```

---

## 3️⃣ NP3O 的关键组件

### 组件 1：双 Critic 网络

```python
class ActorCritic:
    def __init__(self):
        self.actor = Actor()        # 输出动作
        self.critic = Critic()      # 评估价值（奖励）
        self.cost_critic = CostCritic()  # ✅ 评估代价（约束）
```

**作用**：
- **Critic**：预测"这个状态能获得多少奖励"
- **Cost Critic**：预测"这个状态会违反多少约束"

### 组件 2：约束损失函数

```python
# algorithm/np3o.py
class NP3O:
    def __init__(self, cost_value_loss_coef=1.0, cost_viol_loss_coef=1.0):
        self.cost_value_loss_coef = cost_value_loss_coef  # 代价值损失系数
        self.cost_viol_loss_coef = cost_viol_loss_coef    # 代价违反损失系数
```

**配置**：
```python
# configs/tita_parkour_config.py
class algorithm:
    cost_value_loss_coef = 0.1  # 代价值损失权重
    cost_viol_loss_coef = 0.1   # 代价违反损失权重
```

### 组件 3：代价（Cost）定义

在 `tita_constraint_config.py` 中定义了 6 种代价：

```python
class costs:
    class scales:
        pos_limit = 0.3            # 关节位置限制
        torque_limit = 0.3         # 力矩限制
        dof_vel_limits = 0.3       # 关节速度限制
        acc_smoothness = 0.1       # 加速度平滑性
        feet_contact_forces = 0.1  # 脚接触力
        stumble = 0.1              # 绊倒惩罚
    
    class d_values:  # 期望约束值（目标）
        pos_limit = 0.0
        torque_limit = 0.0
        dof_vel_limits = 0.0
        acc_smoothness = 0.0
        feet_contact_forces = 0.0
        stumble = 0.0

class cost:
    num_costs = 6  # 总共 6 个约束
```

---

## 4️⃣ NP3O 的工作流程

### 训练循环

```python
for iteration in range(max_iterations):
    # 1. 收集数据（Rollout）
    for step in range(num_steps_per_env):
        action = policy.act(obs)
        obs, reward, cost, done = env.step(action)  # ✅ 获取 reward 和 cost
        
        storage.add(obs, action, reward, cost)  # 存储奖励和代价
    
    # 2. 计算优势函数（Advantage）
    reward_advantages = compute_gae(rewards)  # 基于奖励
    cost_advantages = compute_gae(costs)      # ✅ 基于代价
    
    # 3. 更新策略（NP3O 特有）
    for epoch in range(num_learning_epochs):
        # 策略损失（PPO）
        policy_loss = compute_ppo_loss(advantages=reward_advantages)
        
        # 价值损失（奖励 Critic）
        value_loss = compute_value_loss(predicted_values, target_values)
        
        # ✅ 代价价值损失（Cost Critic）
        cost_value_loss = compute_cost_value_loss(predicted_costs, target_costs)
        
        # ✅ 代价违反损失（惩罚违反约束）
        cost_viol_loss = compute_cost_violation_loss(costs, d_values)
        
        # 总损失
        total_loss = (policy_loss + 
                      value_loss + 
                      cost_value_loss_coef * cost_value_loss +  # ✅ 新增
                      cost_viol_loss_coef * cost_viol_loss)     # ✅ 新增
        
        optimizer.step()
```

### 关键差异

| 步骤 | 标准 PPO | NP3O |
|------|---------|------|
| 环境反馈 | `reward` | `reward` + `cost` ✅ |
| 优势计算 | 只计算奖励优势 | 奖励优势 + 代价优势 ✅ |
| 损失函数 | 策略损失 + 价值损失 | + 代价价值损失 + 违反损失 ✅ |
| 网络结构 | Actor + Critic | Actor + Critic + Cost Critic ✅ |

---

## 5️⃣ 为什么跑酷用 NP3O？

### 优势 1：安全性保证

**场景**：机器人在高速跑酷时容易：
- 关节超限（损坏硬件）
- 力矩过大（电机烧毁）
- 运动过猛（摔倒受伤）

**NP3O 的作用**：
```python
# 通过约束限制这些危险行为
cost_scales = {
    'pos_limit': 0.3,      # 限制关节角度
    'torque_limit': 0.3,   # 限制电机力矩
    'stumble': 0.1,        # 避免绊倒
}

# 如果违反约束，Cost Critic 会预测高代价
# 策略会学习避免这些高代价的动作
```

### 优势 2：多目标平衡

**跑酷的多重目标**：
1. 跟随速度命令（奖励）
2. 清除障碍（奖励）
3. 避免关节超限（约束）
4. 避免力矩过大（约束）
5. 保持运动平滑（约束）

**NP3O 的处理**：
```python
# 奖励：鼓励好的行为
rewards = {
    'tracking_lin_vel': 1.0,
    'obstacle_clearance': 2.0,
    'jump_timing': 1.5,
}

# 约束：限制危险行为
costs = {
    'pos_limit': 0.3,
    'torque_limit': 0.3,
    'acc_smoothness': 0.1,
}

# NP3O 同时优化奖励和约束
# 找到"高奖励 + 低代价"的最优策略
```

### 优势 3：泛化到真实机器人

**问题**：仿真到现实的迁移（Sim2Real）

**标准 PPO**：
- 可能学到"作弊"动作（仿真有效，现实失败）
- 例如：仿真中可以瞬间加速，现实中电机跟不上

**NP3O**：
- 通过约束限制不切实际的动作
- 学到的策略更接近真实机器人的物理限制
- Sim2Real 迁移成功率更高

---

## 6️⃣ 跑酷训练中的 NP3O 配置

### 完整配置

```python
# configs/tita_parkour_config.py

class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        # PPO 参数
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        
        # ✅ NP3O 特有参数
        cost_value_loss_coef = 0.1   # 代价价值损失权重
        cost_viol_loss_coef = 0.1    # 代价违反损失权重
    
    class policy(TitaConstraintRoughCfgPPO.policy):
        # 网络结构
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        
        # ✅ NP3O 特有
        num_costs = 6  # 6 个约束
    
    class runner(TitaConstraintRoughCfgPPO.runner):
        # ✅ NP3O 算法
        algorithm_class_name = 'NP3O'
        runner_class_name = 'OnConstraintPolicyRunner'
        policy_class_name = 'ActorCriticBarlowTwins'
```

### 继承关系

```
跑酷配置 继承自 → 约束配置 继承自 → 基础配置
TitaParkourCfgPPO → TitaConstraintRoughCfgPPO → LeggedRobotCfgPPO

所有配置都使用 NP3O ✅
```

---

## 7️⃣ 课程学习 + NP3O

### 课程学习如何影响 NP3O？

**关键点**：课程学习**只调整奖励权重**，不改变算法

```python
# 阶段 1：平地行走
rewards = {
    'tracking_lin_vel': 1.0,
    'obstacle_clearance': 0.5,  # 低权重
    'collision': -0.5,          # 轻度惩罚
}
costs = {  # ✅ 约束保持不变
    'pos_limit': 0.3,
    'torque_limit': 0.3,
    ...
}

# 阶段 3：完美跑酷
rewards = {
    'tracking_lin_vel': 1.0,
    'obstacle_clearance': 2.0,  # 高权重
    'collision': -5.0,          # 严厉惩罚
}
costs = {  # ✅ 约束依然不变
    'pos_limit': 0.3,
    'torque_limit': 0.3,
    ...
}
```

**NP3O 的作用**：
- **奖励**：随课程学习动态调整（鼓励跑酷技能）
- **约束**：始终保持（确保安全性）

### 课程学习的更新方式

```python
# train_parkour.py 中的课程学习
for iteration in range(max_iterations):
    # 检查是否需要切换阶段
    if iteration == 10000:  # 阶段 1 → 阶段 2
        # ✅ 只更新奖励权重
        env.cfg.rewards.scales.obstacle_clearance = 1.0  # 从 0.5 增加
        env.cfg.rewards.scales.jump_timing = 0.5        # 新增
        
        # ❌ 不改变约束
        # env.cfg.costs.scales 保持不变
    
    # NP3O 正常训练（同时优化奖励和约束）
    ppo_runner.update()
```

---

## 8️⃣ NP3O 实现细节

### 代码位置

```bash
tita_rl/
├── algorithm/
│   ├── np3o.py              # ✅ NP3O 核心实现
│   └── ppo.py               # 标准 PPO（对比参考）
├── runner/
│   └── on_constraint_policy_runner.py  # ✅ 约束策略运行器
├── modules/
│   └── actor_critic.py      # ✅ Actor-Critic 网络（含 Cost Critic）
└── configs/
    └── tita_parkour_config.py  # 配置 NP3O 参数
```

### 关键代码片段

```python
# algorithm/np3o.py

class NP3O:
    def __init__(self, cost_value_loss_coef=1.0, cost_viol_loss_coef=1.0):
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        
        # 优化器（同时优化 Actor、Critic、Cost Critic）
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    
    def process_env_step(self, rewards, costs, dones, infos):
        """处理环境反馈（奖励 + 代价）"""
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()  # ✅ 存储代价
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
    
    def update(self):
        """更新策略（NP3O 优化）"""
        # 计算奖励优势
        reward_advantages = self.compute_advantages(self.storage.rewards)
        
        # ✅ 计算代价优势
        cost_advantages = self.compute_advantages(self.storage.costs)
        
        # 策略损失（基于奖励优势）
        policy_loss = self.compute_policy_loss(reward_advantages)
        
        # 价值损失（奖励 Critic）
        value_loss = self.compute_value_loss()
        
        # ✅ 代价价值损失（Cost Critic）
        cost_value_loss = self.compute_cost_value_loss()
        
        # ✅ 代价违反损失
        cost_viol_loss = self.compute_cost_violation_loss()
        
        # 总损失
        total_loss = (
            policy_loss + 
            value_loss + 
            self.cost_value_loss_coef * cost_value_loss +
            self.cost_viol_loss_coef * cost_viol_loss
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

---

## 9️⃣ 总结

### ✅ 关键结论

1. **跑酷训练使用 NP3O**：完全继承自 `tita_constraint`
2. **算法没有改变**：只是调整了奖励权重（课程学习）
3. **约束始终存在**：确保机器人安全运动
4. **微调也用 NP3O**：加载预训练模型后继续用 NP3O 优化

### 📊 配置确认

```python
# configs/tita_parkour_config.py

class runner:
    algorithm_class_name = 'NP3O'  # ✅ 确认使用 NP3O
    runner_class_name = 'OnConstraintPolicyRunner'  # 约束运行器
    policy_class_name = 'ActorCriticBarlowTwins'   # 策略网络

class algorithm:
    cost_value_loss_coef = 0.1  # ✅ NP3O 特有参数
    cost_viol_loss_coef = 0.1   # ✅ NP3O 特有参数

class policy:
    num_costs = 6  # ✅ 6 个约束
```

### 🎯 为什么选择 NP3O？

| 原因 | 说明 |
|------|------|
| **安全性** | 跑酷动作激烈，需要约束保护硬件 |
| **多目标** | 同时优化性能（奖励）和安全（约束） |
| **Sim2Real** | 约束限制使策略更接近真实物理 |
| **继承性** | 原始训练就用 NP3O，保持一致 |

### 🔄 完整训练流程

```
1. 初始化 NP3O 算法
   ├─ Actor（策略网络）
   ├─ Critic（价值网络）
   └─ Cost Critic（代价网络）✅

2. 课程学习阶段 1
   ├─ 奖励：基础行走权重
   └─ 约束：始终保持 ✅

3. 课程学习阶段 2
   ├─ 奖励：增加跳跃权重
   └─ 约束：始终保持 ✅

4. 课程学习阶段 3
   ├─ 奖励：完整跑酷权重
   └─ 约束：始终保持 ✅

5. 每次迭代
   └─ NP3O 优化（奖励最大化 + 约束满足）✅
```

---

**最终答案**：是的，依旧使用 **NP3O 算法**！这是一个带约束的 PPO 变体，非常适合跑酷这种需要高性能 + 高安全性的机器人控制任务。课程学习只是调整奖励权重，不改变底层的 NP3O 算法。🎉
