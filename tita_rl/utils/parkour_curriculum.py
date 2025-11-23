"""
跑酷课程学习管理器
用于动态调整训练阶段、奖励权重和环境配置
"""

import numpy as np


class ParkourCurriculum:
    """
    TITA 跑酷课程学习管理器
    
    实现三阶段训练策略：
    - 阶段1：基础行走 (10000 iterations)
    - 阶段2：障碍跨越 (10000 iterations)
    - 阶段3：完美跑酷 (17000 iterations)
    """
    
    def __init__(self):
        self.current_stage = 0
        self.stage_start_iteration = 0
        
        # 定义三个训练阶段
        # 注意：从头训练需要更多迭代次数
        # 当前配置：12000 + 12000 + 16000 = 40000 总迭代
        self.stages = [
            # ============ 阶段1：基础行走训练 ============
            {
                'name': 'stage_1_flat_walking',
                'description': '学习基础行走和姿态控制（从头训练）',
                'iterations': 12000,           # 从头训练：增加到 12000
                'terrain_level': 0,             # 地形难度：平地
                'obstacle_height': 0.05,        # 障碍物高度：5cm（很小）
                'terrain_proportions': [0.2, 0.2, 0.3, 0.2, 0.1],  # 少量障碍
                
                # 奖励权重配置
                'rewards': {
                    # 基础运动
                    'tracking_lin_vel': 1.0,      # 主要任务
                    'tracking_ang_vel': 0.5,
                    'orientation': -1.0,
                    'base_height': -1.0,
                    
                    # 跑酷技能（初级）
                    'obstacle_clearance': 0.5,    # 低权重，只需感知
                    # 'jump_timing': 不存在        # 还不需要时机
                    # 'landing_stability': 不存在   # 还不需要着陆
                    
                    # 运动约束
                    'lin_vel_z': -0.0,            # 不惩罚垂直速度
                    'ang_vel_xy': -0.05,
                    'feet_air_time': 0.0,         # 不奖励腾空
                    
                    # 安全惩罚（宽松）
                    'collision': -0.5,            # 轻度惩罚，鼓励探索
                    'termination': -100,          # 轻度失败代价
                    
                    # 能量与平滑性
                    'powers': -2e-5,
                    'dof_acc': -2.5e-7,
                    'action_rate': -0.01,
                }
            },
            
            # ============ 阶段2：障碍跨越训练 ============
            {
                'name': 'stage_2_small_obstacles',
                'description': '学习识别和跨越小障碍（使用深度信息）',
                'iterations': 12000,           # 从头训练：增加到 12000
                'terrain_level': 3,             # 地形难度：中等
                'obstacle_height': 0.10,        # 障碍物高度：10cm
                'terrain_proportions': [0.1, 0.1, 0.3, 0.2, 0.3],  # 增加障碍
                
                'rewards': {
                    # 基础运动
                    'tracking_lin_vel': 1.0,
                    'tracking_ang_vel': 0.5,
                    'orientation': -1.0,
                    'base_height': -1.0,
                    
                    # 跑酷技能（进阶）
                    'obstacle_clearance': 1.0,    # 中等权重，开始重视
                    'jump_timing': 0.5,           # 新增：开始学习时机
                    # 'landing_stability': 不存在   # 还不要求着陆质量
                    
                    # 运动约束
                    'lin_vel_z': -0.2,            # 轻度惩罚垂直速度
                    'ang_vel_xy': -0.05,
                    'feet_air_time': 0.3,         # 开始奖励腾空
                    
                    # 安全惩罚（标准）
                    'collision': -1.0,            # 标准惩罚
                    'termination': -150,          # 中度失败代价
                    
                    # 能量与平滑性
                    'powers': -2e-5,
                    'dof_acc': -2.5e-7,
                    'action_rate': -0.01,
                }
            },
            
            # ============ 阶段3：完美跑酷训练 ============
            {
                'name': 'stage_3_parkour_mastery',
                'description': '完美跑酷：利用深度信息，精确时机、稳定着陆',
                'iterations': 16000,            # 从头训练：增加到 16000
                'terrain_level': 7,             # 地形难度：高难度
                'obstacle_height': 0.15,        # 障碍物高度：15cm
                'terrain_proportions': [0.05, 0.05, 0.25, 0.20, 0.45],  # 大量障碍
                
                'rewards': {
                    # 基础运动
                    'tracking_lin_vel': 1.0,
                    'tracking_ang_vel': 0.5,
                    'orientation': -1.0,
                    'base_height': -1.0,
                    
                    # 跑酷技能（高级）
                    'obstacle_clearance': 2.0,    # 高权重，必须清除
                    'jump_timing': 1.5,           # 高权重，精确时机
                    'landing_stability': 1.0,     # 新增：稳定着陆
                    
                    # 运动约束
                    'lin_vel_z': -0.5,            # 严格惩罚垂直速度
                    'ang_vel_xy': -0.05,
                    'feet_air_time': 0.5,         # 鼓励腾空
                    
                    # 安全惩罚（严格）
                    'collision': -5.0,            # 严厉惩罚，必须避免
                    'termination': -200,          # 高失败代价
                    
                    # 能量与平滑性
                    'powers': -2e-5,
                    'dof_acc': -2.5e-7,
                    'action_rate': -0.01,
                }
            }
        ]
    
    def get_stage(self, iteration):
        """
        根据当前迭代次数返回对应阶段配置
        
        Args:
            iteration: 当前训练迭代次数
            
        Returns:
            stage: 当前阶段的配置字典
        """
        cumulative = 0
        for i, stage in enumerate(self.stages):
            cumulative += stage['iterations']
            if iteration < cumulative:
                # 检测是否进入新阶段
                if self.current_stage != i:
                    self.current_stage = i
                    self.stage_start_iteration = iteration
                    self._print_stage_info(stage, iteration)
                return stage
        
        # 如果超过总迭代次数，返回最后阶段
        return self.stages[-1]
    
    def _print_stage_info(self, stage, iteration):
        """打印阶段切换信息"""
        print(f"\n{'='*70}")
        print(f"🎓 课程学习阶段切换")
        print(f"{'='*70}")
        print(f"📌 阶段名称: {stage['name']}")
        print(f"📝 描述: {stage['description']}")
        print(f"🔢 当前迭代: {iteration}")
        print(f"⏱️  阶段时长: {stage['iterations']} iterations")
        print(f"🏔️  地形难度: Level {stage['terrain_level']}")
        print(f"📏 障碍高度: {stage['obstacle_height']*100:.1f} cm")
        print(f"{'='*70}")
        print(f"🎯 关键奖励权重:")
        for reward_name in ['obstacle_clearance', 'jump_timing', 'landing_stability', 
                            'collision', 'termination']:
            if reward_name in stage['rewards']:
                print(f"   {reward_name:25s}: {stage['rewards'][reward_name]:7.1f}")
            else:
                print(f"   {reward_name:25s}: {'不存在':>7s}")
        print(f"{'='*70}\n")
    
    def update_env_config(self, env, stage):
        """
        根据阶段更新环境配置
        
        Args:
            env: 环境实例
            stage: 当前阶段配置
        """
        # 更新地形难度
        env.cfg.terrain.max_init_terrain_level = stage['terrain_level']
        
        # 更新地形比例（如果存在）
        if 'terrain_proportions' in stage and hasattr(env, 'terrain'):
            # 注意：这需要重新生成地形，可能比较耗时
            # 在实际应用中可能需要重新创建环境
            pass
        
        print(f"✅ 环境配置已更新 - 地形难度: Level {stage['terrain_level']}")
    
    def update_reward_scales(self, env, stage):
        """
        根据阶段动态更新奖励函数权重
        
        Args:
            env: 环境实例
            stage: 当前阶段配置
        """
        updated_count = 0
        added_count = 0
        
        for reward_name, scale in stage['rewards'].items():
            # 检查奖励函数是否存在
            if hasattr(env.cfg.rewards.scales, reward_name):
                # 获取旧值
                old_value = getattr(env.cfg.rewards.scales, reward_name)
                
                # 更新权重
                setattr(env.cfg.rewards.scales, reward_name, scale)
                
                # 打印变化（如果有显著变化）
                if abs(old_value - scale) > 0.01:
                    change_symbol = "⬆️" if scale > old_value else "⬇️" if scale < old_value else "➡️"
                    print(f"  {change_symbol} {reward_name:25s}: {old_value:7.2f} → {scale:7.2f}")
                    updated_count += 1
            else:
                # 动态添加新的奖励函数权重
                setattr(env.cfg.rewards.scales, reward_name, scale)
                print(f"  ✨ {reward_name:25s}: 新增 = {scale:7.2f}")
                added_count += 1
        
        print(f"\n📊 奖励权重更新统计: 更新 {updated_count} 个, 新增 {added_count} 个\n")
    
    def get_progress(self, iteration):
        """
        获取当前训练进度
        
        Args:
            iteration: 当前迭代次数
            
        Returns:
            dict: 包含进度信息的字典
        """
        stage = self.get_stage(iteration)
        
        # 计算阶段内进度
        stage_progress = (iteration - self.stage_start_iteration) / stage['iterations']
        stage_progress = min(1.0, stage_progress)
        
        # 计算总体进度
        total_iterations = sum(s['iterations'] for s in self.stages)
        total_progress = iteration / total_iterations
        
        return {
            'stage_index': self.current_stage,
            'stage_name': stage['name'],
            'stage_progress': stage_progress,
            'total_progress': total_progress,
            'iterations_in_stage': iteration - self.stage_start_iteration,
            'total_iterations_in_stage': stage['iterations']
        }
    
    def should_update_config(self, iteration, update_interval=100):
        """
        判断是否应该更新配置
        
        Args:
            iteration: 当前迭代次数
            update_interval: 更新间隔
            
        Returns:
            bool: 是否应该更新
        """
        return iteration % update_interval == 0
    
    def get_total_iterations(self):
        """获取总迭代次数"""
        return sum(stage['iterations'] for stage in self.stages)
    
    def print_curriculum_summary(self):
        """打印课程学习配置摘要"""
        print("\n" + "="*70)
        print("📚 课程学习配置摘要")
        print("="*70)
        
        total_iters = self.get_total_iterations()
        
        for i, stage in enumerate(self.stages, 1):
            percentage = (stage['iterations'] / total_iters) * 100
            print(f"\n阶段 {i}: {stage['name']}")
            print(f"  ├─ 描述: {stage['description']}")
            print(f"  ├─ 迭代次数: {stage['iterations']:,} ({percentage:.1f}%)")
            print(f"  ├─ 地形难度: Level {stage['terrain_level']}")
            print(f"  ├─ 障碍高度: {stage['obstacle_height']*100:.0f} cm")
            print(f"  └─ 关键奖励: ", end="")
            
            key_rewards = []
            if 'obstacle_clearance' in stage['rewards']:
                key_rewards.append(f"障碍清除={stage['rewards']['obstacle_clearance']}")
            if 'jump_timing' in stage['rewards']:
                key_rewards.append(f"跳跃时机={stage['rewards']['jump_timing']}")
            if 'landing_stability' in stage['rewards']:
                key_rewards.append(f"着陆稳定={stage['rewards']['landing_stability']}")
            
            print(", ".join(key_rewards) if key_rewards else "基础训练")
        
        print(f"\n总迭代次数: {total_iters:,}")
        print(f"预计训练时间: ~{total_iters * 0.5 / 3600:.1f} 小时 (RTX 3060)")
        print("="*70 + "\n")


# 辅助函数：创建默认课程
def create_default_curriculum():
    """创建默认的跑酷课程"""
    return ParkourCurriculum()


# 如果直接运行此文件，打印课程配置
if __name__ == "__main__":
    curriculum = create_default_curriculum()
    curriculum.print_curriculum_summary()
    
    # 模拟训练过程
    print("\n🔄 模拟训练过程:\n")
    test_iterations = [0, 5000, 10000, 15000, 20000, 27000, 37000]
    
    for iteration in test_iterations:
        stage = curriculum.get_stage(iteration)
        progress = curriculum.get_progress(iteration)
        print(f"Iteration {iteration:5d}: {stage['name']:30s} "
              f"(阶段进度: {progress['stage_progress']*100:5.1f}%, "
              f"总进度: {progress['total_progress']*100:5.1f}%)")
