#!/usr/bin/env python3
"""
地形比例分析工具（纯文本版）
不需要 matplotlib，只打印文本分析
"""

def analyze_terrain_proportions(proportions, config_name="默认配置"):
    """
    分析地形比例配置（纯文本输出）
    
    Args:
        proportions: 地形比例列表，例如 [0.1, 0.1, 0.35, 0.25, 0.2]
        config_name: 配置名称
    """
    names = [
        '平滑斜坡',
        '粗糙斜坡', 
        '楼梯',
        '离散障碍物',
        '踏脚石/沟壑/深坑'
    ]
    
    # 计算累积比例（与 terrain.py 中相同的逻辑）
    cumulative = []
    total = 0
    for p in proportions:
        total += p
        cumulative.append(total)
    
    print("\n" + "=" * 80)
    print(f"  {config_name}")
    print("=" * 80)
    
    print(f"\n📊 原始配置:")
    print(f"   terrain_proportions = {proportions}")
    
    print(f"\n📈 累积比例 (self.proportions):")
    print(f"   {cumulative}")
    
    print("\n" + "-" * 80)
    print("详细映射关系:")
    print("-" * 80)
    
    # 表头
    print(f"{'地形类型':<18} {'比例':<8} {'Choice范围':<18} {'代码判断':<30}")
    print("-" * 80)
    
    # 打印详细映射
    prev = 0.0
    for i, (name, prop, cum) in enumerate(zip(names, proportions, cumulative)):
        prop_str = f"{prop*100:.1f}%"
        range_str = f"[{prev:.3f}, {cum:.3f})"
        
        if i == 0:
            code_str = f"if choice < {cum:.3f}"
        else:
            code_str = f"elif choice < {cum:.3f}"
        
        print(f"{name:<18} {prop_str:<8} {range_str:<18} {code_str:<30}")
        prev = cum
    
    print("-" * 80)
    
    # ASCII 条形图
    print("\n📊 可视化分布 (每个 # 代表 2%):")
    print("-" * 80)
    
    max_width = 50
    for name, prop in zip(names, proportions):
        bar_length = int(prop * max_width / 0.02)  # 每个#代表2%
        bar = '#' * bar_length
        print(f"{name:<18} {bar} {prop*100:.1f}%")
    
    print("-" * 80)
    
    # 关键指标
    print("\n🎯 关键指标:")
    max_idx = proportions.index(max(proportions))
    min_idx = proportions.index(min(proportions))
    print(f"   最多的地形: {names[max_idx]} ({proportions[max_idx]*100:.1f}%)")
    print(f"   最少的地形: {names[min_idx]} ({proportions[min_idx]*100:.1f}%)")
    
    # 跑酷相关
    parkour_idx = 3  # 离散障碍物
    print(f"\n🏃 跑酷训练相关:")
    print(f"   离散障碍物比例: {proportions[parkour_idx]*100:.1f}%")
    if proportions[parkour_idx] >= 0.4:
        print(f"   评价: ✅ 高比例，适合跑酷训练")
    elif proportions[parkour_idx] >= 0.25:
        print(f"   评价: ⚠️  中等比例，可以训练跑酷")
    else:
        print(f"   评价: ❌ 比例较低，不适合专门跑酷训练")

def simulate_generation(proportions, num_samples=10000):
    """
    模拟地形生成（不使用numpy）
    """
    import random
    
    cumulative = []
    total = 0
    for p in proportions:
        total += p
        cumulative.append(total)
    
    names = ['平滑斜坡', '粗糙斜坡', '楼梯', '离散障碍物', '踏脚石/沟壑/深坑']
    
    # 统计计数
    counts = [0] * len(proportions)
    
    for _ in range(num_samples):
        choice = random.random()
        for i, cum in enumerate(cumulative):
            if choice < cum:
                counts[i] += 1
                break
    
    print(f"\n🎲 模拟生成统计 ({num_samples:,} 个样本):")
    print("-" * 80)
    print(f"{'地形类型':<18} {'期望比例':<12} {'实际比例':<12} {'实际数量':<12} {'误差'}")
    print("-" * 80)
    
    for i, (name, expected, actual) in enumerate(zip(names, proportions, counts)):
        expected_pct = expected * 100
        actual_pct = (actual / num_samples) * 100
        error = abs(expected_pct - actual_pct)
        print(f"{name:<18} {expected_pct:6.2f}%      {actual_pct:6.2f}%      {actual:6d}      {error:5.2f}%")
    
    print("-" * 80)

def compare_configs():
    """
    对比不同配置
    """
    configs = {
        "默认平衡配置": [0.1, 0.1, 0.35, 0.25, 0.2],
        "跑酷优化配置": [0.05, 0.05, 0.25, 0.45, 0.20],
        "极端跑酷配置": [0.0, 0.0, 0.0, 1.0, 0.0],
        "楼梯专精配置": [0.0, 0.0, 1.0, 0.0, 0.0],
    }
    
    print("\n" + "=" * 80)
    print("  配置对比分析")
    print("=" * 80)
    
    names = ['平滑斜坡', '粗糙斜坡', '楼梯', '离散障碍物', '踏脚石/沟壑/深坑']
    
    # 表头
    print(f"\n{'地形类型':<18}", end='')
    for config_name in configs.keys():
        print(f"{config_name:<20}", end='')
    print()
    print("-" * 98)
    
    # 每种地形的比例对比
    for i, name in enumerate(names):
        print(f"{name:<18}", end='')
        for config_name, props in configs.items():
            print(f"{props[i]*100:6.1f}%             ", end='')
        print()
    
    print("-" * 98)
    
    # 推荐使用场景
    print("\n💡 推荐使用场景:")
    print("-" * 80)
    print("  默认平衡配置: 通用训练，各种地形都能应对")
    print("  跑酷优化配置: 专注跑酷能力，增加障碍物训练 ⭐推荐")
    print("  极端跑酷配置: 极限跑酷训练，仅障碍物")
    print("  楼梯专精配置: 台阶导航专项训练")
    print("-" * 80)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  🎯 TITA RL 地形比例配置分析工具（纯文本版）")
    print("=" * 80)
    
    # 分析默认配置
    default_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    analyze_terrain_proportions(default_proportions, "默认平衡配置")
    simulate_generation(default_proportions)
    
    # 分析跑酷配置
    parkour_proportions = [0.05, 0.05, 0.25, 0.45, 0.20]
    analyze_terrain_proportions(parkour_proportions, "跑酷优化配置")
    simulate_generation(parkour_proportions)
    
    # 配置对比
    compare_configs()
    
    print("\n" + "=" * 80)
    print("  ✨ 分析完成！")
    print("=" * 80)
    
    print("\n📝 如何修改配置？")
    print("-" * 80)
    print("  1. 编辑配置文件: configs/legged_robot_config.py (第75行)")
    print("     terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]")
    print()
    print("  2. 或在 TITA 配置中覆盖: configs/tita_constraint_config.py")
    print("     class terrain(LeggedRobotCfg.terrain):")
    print("         terrain_proportions = [0.05, 0.05, 0.25, 0.45, 0.20]  # 跑酷优化")
    print("-" * 80)
    print()
