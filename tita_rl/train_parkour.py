"""
TITA 跑酷训练脚本（带课程学习）

使用方法：
python train_parkour.py --task=tita_parkour --headless

选项：
--task=tita_parkour      # 任务名称
--headless               # 无头模式（无可视化）
--resume                 # 从检查点恢复训练
--load_run=<run_name>    # 指定要加载的运行名称
"""

import numpy as np
import os
from datetime import datetime

# ========== 重要：Isaac Gym 必须在 PyTorch 之前导入 ==========
from global_config import ROOT_DIR
import isaacgym
from utils.helpers import get_args
from utils.task_registry import task_registry

# 现在可以安全导入 PyTorch 相关模块
import torch

# 导入配置
from configs.tita_parkour_config import TitaParkourCfg, TitaParkourCfgPPO

# 导入环境
from envs.parkour_robot import ParkourRobot

# 导入工具
from utils.parkour_curriculum import ParkourCurriculum


def train_with_curriculum(args):
    """
    使用课程学习训练跑酷策略
    
    Args:
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("🎯 TITA 跑酷训练（带课程学习）")
    print("="*70)
    
    # 创建环境和算法
    print("\n📦 创建训练环境...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print("🧠 创建策略网络...")
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 创建课程学习管理器
    print("\n📚 初始化课程学习管理器...")
    curriculum = ParkourCurriculum()
    curriculum.print_curriculum_summary()
    
    # 保存配置
    logs_path = os.path.join(ROOT_DIR, "logs")
    task_config_folder = os.path.join(logs_path, f"{args.task}")
    
    if os.path.exists(task_config_folder) and os.path.isdir(task_config_folder):
        print(f"💾 保存配置文件到: {task_config_folder}")
        task_registry.save_cfgs(name=args.task, train_cfg=train_cfg)
    else:
        print(f"⚠️  任务配置文件夹不存在: {task_config_folder}")
    
    # ============ 修改 PPO Runner 以支持课程学习 ============
    original_learn = ppo_runner.learn
    
    def learn_with_curriculum(num_learning_iterations, init_at_random_ep_len=True):
        """
        重写 learn 方法，添加课程学习逻辑
        """
        # 初始化 TensorBoard writer（对照原始代码）
        if ppo_runner.log_dir is not None and ppo_runner.writer is None:
            from tensorboardX import SummaryWriter
            ppo_runner.writer = SummaryWriter(log_dir=ppo_runner.log_dir, flush_secs=10)
        
        # 初始化随机回合长度（对照原始代码）
        if init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(env.episode_length_buf,
                                                       high=int(env.max_episode_length))
        
        print("\n" + "="*70)
        print("🚀 开始训练")
        print("="*70)
        print(f"总迭代次数: {num_learning_iterations}")
        print(f"课程学习: 启用")
        print("="*70 + "\n")
        
        # 初始化观测
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(ppo_runner.device), critic_obs.to(ppo_runner.device)
        
        # 初始化 infos（对照 on_constraint_policy_runner.py）
        infos = {}
        if_depth = hasattr(ppo_runner, 'if_depth') and ppo_runner.if_depth
        infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device) if if_depth else None
        
        ppo_runner.alg.actor_critic.train()
        
        # 使用 deque 而不是 list（对照原始代码）
        from collections import deque
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=ppo_runner.device)
        cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=ppo_runner.device)
        
        tot_iter = 0
        
        # 获取初始阶段
        current_stage = curriculum.get_stage(0)
        curriculum.update_reward_scales(env, current_stage)
        last_update_iter = 0
        
        # 训练循环
        for it in range(num_learning_iterations):
            tot_iter += 1
            
            # ============ 课程学习：检查是否需要切换阶段 ============
            if curriculum.should_update_config(it, update_interval=100):
                stage = curriculum.get_stage(it)
                
                # 如果进入新阶段，更新配置
                if stage != current_stage:
                    current_stage = stage
                    print(f"\n{'='*70}")
                    print(f"📊 迭代 {it}: 切换到新阶段")
                    print(f"{'='*70}")
                    curriculum.update_env_config(env, stage)
                    curriculum.update_reward_scales(env, stage)
                    last_update_iter = it
            
            # ============ 正常训练步骤 ============
            start = time.time()
            
            # Rollout
            with torch.inference_mode():
                for i in range(train_cfg.runner.num_steps_per_env):
                    actions = ppo_runner.alg.act(obs, critic_obs, infos)
                    obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, costs, dones = obs.to(ppo_runner.device), critic_obs.to(ppo_runner.device), rewards.to(ppo_runner.device), costs.to(ppo_runner.device), dones.to(ppo_runner.device)
                    ppo_runner.alg.process_env_step(rewards, costs, dones, infos)
                    
                    # 统计
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                
                stop = time.time()
                collection_time = stop - start
                
                # 计算值函数（对照原始代码：需要同时计算 returns 和 cost_returns）
                start = stop
                ppo_runner.alg.compute_returns(critic_obs)
                ppo_runner.alg.compute_cost_returns(critic_obs)
            
            # 更新 k 值（对照原始代码：用于约束优化）
            k_value = ppo_runner.alg.update_k_value(it)
            
            # 学习
            mean_value_loss, mean_cost_value_loss, mean_viol_loss, mean_surrogate_loss, mean_imitation_loss = ppo_runner.alg.update()
            stop = time.time()
            learn_time = stop - start
            
            # ============ 日志记录 ============
            if it % train_cfg.runner.log_interval == 0:
                # 获取进度信息
                progress = curriculum.get_progress(it)
                
                # 打印基本信息
                print(f"\n{'='*70}")
                print(f"📊 迭代 {it}/{num_learning_iterations}")
                print(f"{'='*70}")
                print(f"阶段: {progress['stage_name']}")
                print(f"阶段进度: {progress['stage_progress']*100:.1f}%")
                print(f"总进度: {progress['total_progress']*100:.1f}%")
                
                if len(rewbuffer) > 0:
                    print(f"平均奖励: {np.mean(rewbuffer):.2f}")
                    print(f"平均回合长度: {np.mean(lenbuffer):.1f}")
                
                print(f"采集时间: {collection_time:.3f}s")
                print(f"学习时间: {learn_time:.3f}s")
                print(f"{'='*70}\n")
                
                # 记录到 TensorBoard（如果有）
                if hasattr(ppo_runner, 'writer') and ppo_runner.writer is not None:
                    ppo_runner.writer.add_scalar('Curriculum/stage_index', progress['stage_index'], it)
                    ppo_runner.writer.add_scalar('Curriculum/stage_progress', progress['stage_progress'], it)
                    ppo_runner.writer.add_scalar('Curriculum/total_progress', progress['total_progress'], it)
                    
                    if len(rewbuffer) > 0:
                        ppo_runner.writer.add_scalar('Train/mean_reward', np.mean(rewbuffer), it)
                        ppo_runner.writer.add_scalar('Train/mean_episode_length', np.mean(lenbuffer), it)
                
                rewbuffer.clear()
                lenbuffer.clear()
            
            # ============ 保存检查点 ============
            if it % train_cfg.runner.save_interval == 0:
                print(f"💾 保存检查点 (iteration {it})...")
                ppo_runner.save(os.path.join(ppo_runner.log_dir, f'model_{it}.pt'))
        
        # 训练结束，保存最终模型
        print(f"\n{'='*70}")
        print("✅ 训练完成！")
        print(f"{'='*70}")
        print(f"💾 保存最终模型...")
        ppo_runner.save(os.path.join(ppo_runner.log_dir, 'model_final.pt'))
        print(f"模型保存位置: {ppo_runner.log_dir}")
        print(f"{'='*70}\n")
    
    # 替换 learn 方法
    import time
    ppo_runner.learn_with_curriculum = learn_with_curriculum
    
    # 开始训练
    print("\n🎓 使用课程学习训练策略...")
    print(f"   - 阶段1: {curriculum.stages[0]['iterations']} iterations")
    print(f"   - 阶段2: {curriculum.stages[1]['iterations']} iterations")
    print(f"   - 阶段3: {curriculum.stages[2]['iterations']} iterations")
    print(f"   - 总计: {curriculum.get_total_iterations()} iterations\n")
    
    ppo_runner.learn_with_curriculum(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )


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
    
    # 如果没有指定任务，默认使用 tita_parkour
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'tita_parkour'
        print(f"⚙️  使用默认任务: {args.task}")
    
    # 开始训练
    train_with_curriculum(args)


if __name__ == '__main__':
    main()
