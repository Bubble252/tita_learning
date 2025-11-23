#!/usr/bin/env python3
"""
地形比例可视化工具
用于理解 terrain_proportions 如何映射到实际地形分布
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_terrain_proportions(proportions, names=None):
    """
    可视化地形比例分布
    
    Args:
        proportions: 地形比例列表，例如 [0.1, 0.1, 0.35, 0.25, 0.2]
        names: 地形名称列表
    """
    if names is None:
        names = [
            '平滑斜坡',
            '粗糙斜坡', 
            '楼梯',
            '离散障碍物',
            '踏脚石/沟壑/深坑'
        ]
    
    # 计算累积比例（与 terrain.py 中相同的逻辑）
    cumulative = [np.sum(proportions[:i+1]) for i in range(len(proportions))]
    
    print("=" * 60)
    print("地形比例配置分析")
    print("=" * 60)
    print(f"\n原始配置: terrain_proportions = {proportions}")
    print(f"累积比例: self.proportions = {cumulative}")
    print("\n" + "-" * 60)
    
    # 打印详细映射
    prev = 0.0
    for i, (name, prop, cum) in enumerate(zip(names, proportions, cumulative)):
        print(f"\n地形 {i+1}: {name}")
        print(f"  - 原始比例: {prop*100:.1f}%")
        print(f"  - Choice 范围: [{prev:.2f}, {cum:.2f})")
        print(f"  - 代码判断: {'if' if i == 0 else 'elif'} choice < self.proportions[{i}]  # {cum}")
        prev = cum
    
    print("\n" + "=" * 60)
    
    # 绘制饼图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：比例饼图
    colors = ['#FF6B6B', '#FFA06B', '#FFD93D', '#6BCF7F', '#4ECDC4']
    explode = [0.05 if p == max(proportions) else 0 for p in proportions]
    
    axes[0].pie(proportions, labels=names, autopct='%1.1f%%',
                colors=colors, explode=explode, startangle=90)
    axes[0].set_title('地形类型分布', fontsize=14, fontweight='bold')
    
    # 右图：choice 值范围
    prev = 0
    for i, (name, cum) in enumerate(zip(names, cumulative)):
        height = cum - prev
        axes[1].barh(i, height, left=prev, color=colors[i], 
                     edgecolor='black', linewidth=1.5)
        # 添加文本标签
        axes[1].text(prev + height/2, i, f'{name}\n{height*100:.0f}%', 
                     ha='center', va='center', fontsize=10, fontweight='bold')
        prev = cum
    
    axes[1].set_xlabel('Choice 值范围 (0.0 - 1.0)', fontsize=12)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels([f'地形 {i+1}' for i in range(len(names))])
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Choice 值映射关系', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('terrain_proportions_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化图表已保存: terrain_proportions_visualization.png")
    plt.show()

def simulate_terrain_generation(proportions, num_samples=1000):
    """
    模拟地形生成，统计实际分布
    
    Args:
        proportions: 地形比例配置
        num_samples: 模拟样本数
    """
    cumulative = [np.sum(proportions[:i+1]) for i in range(len(proportions))]
    
    # 模拟随机选择
    choices = np.random.uniform(0, 1, num_samples)
    
    # 统计每种地形的实际生成次数
    terrain_counts = [0] * len(proportions)
    
    for choice in choices:
        for i, cum in enumerate(cumulative):
            if choice < cum:
                terrain_counts[i] += 1
                break
    
    print("\n" + "=" * 60)
    print(f"模拟地形生成 ({num_samples} 个样本)")
    print("=" * 60)
    
    names = ['平滑斜坡', '粗糙斜坡', '楼梯', '离散障碍物', '踏脚石/沟壑/深坑']
    
    for i, (name, expected, actual) in enumerate(zip(names, proportions, terrain_counts)):
        expected_pct = expected * 100
        actual_pct = (actual / num_samples) * 100
        print(f"{name:12s}: 期望 {expected_pct:5.1f}% | 实际 {actual_pct:5.1f}% | 差异 {abs(expected_pct - actual_pct):4.1f}%")

if __name__ == "__main__":
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("\n🎯 TITA RL 地形比例配置工具\n")
    
    # 默认配置
    print("📊 配置 1: 默认平衡配置")
    default_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    visualize_terrain_proportions(default_proportions)
    simulate_terrain_generation(default_proportions)
    
    print("\n" + "="*60 + "\n")
    
    # 跑酷配置
    print("📊 配置 2: 跑酷优化配置（更多障碍物）")
    parkour_proportions = [0.05, 0.05, 0.25, 0.45, 0.20]
    visualize_terrain_proportions(parkour_proportions)
    simulate_terrain_generation(parkour_proportions)
    
    print("\n" + "="*60 + "\n")
    
    # 极端配置
    print("📊 配置 3: 极端跑酷配置（仅障碍物）")
    extreme_proportions = [0.0, 0.0, 0.0, 1.0, 0.0]
    visualize_terrain_proportions(extreme_proportions)
    simulate_terrain_generation(extreme_proportions)
    
    print("\n✨ 完成！")
