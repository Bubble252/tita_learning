"""
TITA 跑酷策略测试和评估脚本

使用方法：
python test_parkour.py --load_run=<run_name> --checkpoint=<checkpoint_name>

示例：
python test_parkour.py --load_run=parkour_with_curriculum --checkpoint=model_final.pt
python test_parkour.py --load_run=parkour_with_curriculum --checkpoint=model_37000.pt --num_envs=100
"""

import numpy as np
import os
import torch
from datetime import datetime

# 导入配置
from configs.tita_parkour_config import TitaParkourCfg, TitaParkourCfgPPO

# 导入环境
from envs.parkour_robot import ParkourRobot

# 导入工具
from global_config import ROOT_DIR
import isaacgym
from utils.helpers import get_args
from utils.task_registry import task_registry


class ParkourEvaluator:
    """跑酷策略评估器"""
    
    def __init__(self, env, policy_path, device='cuda:0'):
        """
        Args:
            env: 环境实例
            policy_path: 策略文件路径
            device: 设备
        """
        self.env = env
        self.device = device
        
        # 加载策略
        print(f"📦 加载策略: {policy_path}")
        self.policy = torch.jit.load(policy_path).to(device)
        self.policy.eval()
        
        # 统计缓冲区
        self.reset_statistics()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.obstacle_success_count = 0
        self.jump_count = 0
        self.fall_count = 0
        self.collision_count = 0
        
        # 当前回合统计
        self.current_reward = torch.zeros(self.env.num_envs, device=self.device)
        self.current_length = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int)
    
    def evaluate(self, num_episodes=100, max_steps=1000):
        """
        评估策略
        
        Args:
            num_episodes: 评估回合数
            max_steps: 每个回合最大步数
            
        Returns:
            dict: 评估结果
        """
        print(f"\n{'='*70}")
        print(f"🎯 开始评估")
        print(f"{'='*70}")
        print(f"评估回合数: {num_episodes}")
        print(f"每回合最大步数: {max_steps}")
        print(f"{'='*70}\n")
        
        self.reset_statistics()
        obs = self.env.reset()
        
        completed_episodes = 0
        step = 0
        
        with torch.no_grad():
            while completed_episodes < num_episodes and step < max_steps * num_episodes:
                # 执行动作
                actions = self.policy(obs)
                obs, _, rewards, _, dones, infos = self.env.step(actions)
                
                # 更新统计
                self.current_reward += rewards
                self.current_length += 1
                
                # 检查完成的回合
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if len(done_ids) > 0:
                    for env_id in done_ids:
                        self.episode_rewards.append(self.current_reward[env_id].item())
                        self.episode_lengths.append(self.current_length[env_id].item())
                        
                        # 重置该环境的统计
                        self.current_reward[env_id] = 0
                        self.current_length[env_id] = 0
                        
                        completed_episodes += 1
                        
                        # 进度显示
                        if completed_episodes % 10 == 0:
                            print(f"  完成 {completed_episodes}/{num_episodes} 回合...")
                
                step += 1
        
        # 计算统计结果
        results = self._compute_statistics()
        
        return results
    
    def _compute_statistics(self):
        """计算评估统计"""
        results = {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': np.min(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
            
            'num_episodes': len(self.episode_rewards),
        }
        
        return results
    
    def print_results(self, results):
        """打印评估结果"""
        print(f"\n{'='*70}")
        print("📊 评估结果")
        print(f"{'='*70}")
        print(f"评估回合数: {results['num_episodes']}")
        print(f"\n奖励统计:")
        print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  最小奖励: {results['min_reward']:.2f}")
        print(f"  最大奖励: {results['max_reward']:.2f}")
        print(f"\n回合长度:")
        print(f"  平均长度: {results['mean_length']:.1f} ± {results['std_length']:.1f} steps")
        print(f"{'='*70}\n")


def test_parkour(args):
    """
    测试跑酷策略
    
    Args:
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("🎯 TITA 跑酷策略测试")
    print("="*70)
    
    # 创建环境
    print("\n📦 创建测试环境...")
    env_cfg = TitaParkourCfg()
    env_cfg.env.num_envs = min(args.num_envs, 100) if hasattr(args, 'num_envs') else 100
    
    from utils import class_to_dict
    from isaacgym import gymutil
    
    # 创建仿真参数
    sim_params = gymutil.parse_sim_config(vars(env_cfg.sim))
    
    # 创建环境
    env = ParkourRobot(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless
    )
    
    # 构建策略路径
    if hasattr(args, 'load_run') and args.load_run:
        log_root = os.path.join(ROOT_DIR, 'logs', args.task)
        log_dir = os.path.join(log_root, args.load_run)
        
        if hasattr(args, 'checkpoint') and args.checkpoint:
            policy_path = os.path.join(log_dir, args.checkpoint)
        else:
            policy_path = os.path.join(log_dir, 'model_final.pt')
    else:
        raise ValueError("请指定 --load_run 参数")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"策略文件不存在: {policy_path}")
    
    print(f"策略文件: {policy_path}")
    print(f"测试环境数: {env.num_envs}")
    
    # 创建评估器
    evaluator = ParkourEvaluator(
        env=env,
        policy_path=policy_path,
        device=args.sim_device
    )
    
    # 运行评估
    num_episodes = args.num_test_episodes if hasattr(args, 'num_test_episodes') else 100
    results = evaluator.evaluate(num_episodes=num_episodes)
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存结果
    if hasattr(args, 'save_results') and args.save_results:
        results_path = os.path.join(log_dir, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(results_path, 'w') as f:
            f.write("TITA 跑酷评估结果\n")
            f.write("="*50 + "\n\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"💾 结果已保存到: {results_path}")


def main():
    """主函数"""
    # 注册任务
    print("📝 注册跑酷任务...")
    task_registry.register(
        "tita_parkour",
        ParkourRobot,
        TitaParkourCfg(),
        TitaParkourCfgPPO()
    )
    
    # 获取参数
    args = get_args()
    
    # 设置默认参数
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'tita_parkour'
    
    if not hasattr(args, 'num_envs'):
        args.num_envs = 100
    
    if not hasattr(args, 'num_test_episodes'):
        args.num_test_episodes = 100
    
    if not hasattr(args, 'headless'):
        args.headless = False  # 测试时默认显示可视化
    
    # 开始测试
    test_parkour(args)


if __name__ == '__main__':
    main()
