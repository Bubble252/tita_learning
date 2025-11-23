# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
TITA 跑酷专用配置文件
基于 tita_constraint_config.py，添加跑酷相关功能
"""

from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from global_config import ROOT_DIR

class TitaParkourCfg(TitaConstraintRoughCfg):
    """TITA 跑酷环境配置"""
    
    class env(TitaConstraintRoughCfg.env):
        num_envs = 2048
        
        # 观测空间配置（添加深度特征 - 从头训练）
        n_scan = 187
        n_priv_latent = 4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
        n_proprio = 33
        n_depth_features = 5  # 深度特征：左、中、右、近、远
        history_len = 10
        
        # 总观测维度（暂时不包含深度特征，先测试环境是否能创建）
        # TODO: 深度特征导致段错误，需要调试
        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent
        # = 33 + 187 + 330 + 36 = 586（与原始配置相同）
        
        episode_length_s = 20
    
    class init_state(TitaConstraintRoughCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # 稍微抬高初始高度
        rot = [0, 0.0, 0.0, 1]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        
        default_joint_angles = {
            'joint_left_leg_1': 0,
            'joint_right_leg_1': 0,
            'joint_left_leg_2': 0.8,
            'joint_right_leg_2': 0.8,
            'joint_left_leg_3': -1.5,
            'joint_right_leg_3': -1.5,
            'joint_left_leg_4': 0,
            'joint_right_leg_4': 0,
        }
    
    class control(TitaConstraintRoughCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40}
        damping = {'joint': 1.0}
        action_scale = 0.5
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True
    
    class depth(TitaConstraintRoughCfg.depth):
        """深度相机配置 - 保持与原始配置一致"""
        use_camera = False             # ❌ 暂时禁用深度相机（避免显存不足）
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        
        # 以下参数与 tita_constraint_config.py 完全一致，不做修改
        position = [0.27, 0, 0.03]     # front camera
        angle = [-5, 5]                # positive pitch down
        
        update_interval = 1            # 5 works without retraining, 8 worse
        
        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
    
    class terrain(TitaConstraintRoughCfg.terrain):
        """地形配置 - 跑酷专用"""
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True              # 启用课程学习
        
        # 跑酷地形比例（大幅增加障碍物）
        terrain_proportions = [
            0.05,  # 平滑斜坡 (减少)
            0.05,  # 粗糙斜坡 (减少)
            0.25,  # 上楼梯 (需要跳跃)
            0.20,  # 下楼梯
            0.45   # 离散障碍 (跑酷重点，大幅增加)
        ]
        
        # 地形尺寸
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 15                  # 增加难度级别（从10到15）
        num_cols = 20
        max_init_terrain_level = 3     # 从中等难度开始
        
        # 测量点
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 
                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        slope_treshold = 0.75
    
    class commands(TitaConstraintRoughCfg.commands):
        """命令配置"""
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        global_reference = False
        
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]
    
    class rewards(TitaConstraintRoughCfg.rewards):
        """奖励函数配置 - 跑酷专用"""
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        
        class scales(TitaConstraintRoughCfg.rewards.scales):
            # ========== 基础运动奖励（保持不变） ==========
            tracking_lin_vel = 1.0      # 跟踪线速度（主要任务）
            tracking_ang_vel = 0.5      # 跟踪角速度
            orientation = -1.0          # 惩罚姿态偏差
            base_height = -1.0          # 惩罚高度偏差
            
            # ========== 跑酷专用奖励（课程学习会动态调整） ==========
            # 注意：这些是默认值，课程学习会覆盖这些权重
            obstacle_clearance = 2.0    # 障碍物清除（会动态调整：0.5→1.0→2.0）
            jump_timing = 1.5           # 跳跃时机（会动态调整：0→0.5→1.5）
            landing_stability = 1.0     # 着陆稳定性（会动态调整：0→0→1.0）
            
            # ========== 运动约束（部分会动态调整） ==========
            lin_vel_z = -0.5            # 惩罚垂直速度（会调整：-0.0→-0.2→-0.5）
            ang_vel_xy = -0.05          # 惩罚横滚/俯仰角速度
            feet_air_time = 0.5         # 奖励腾空时间（从0.0改为0.5）
            
            # ========== 安全惩罚（会动态调整） ==========
            collision = -5.0            # 碰撞惩罚（会调整：-0.5→-1.0→-5.0）
            termination = -200          # 终止惩罚（会调整：-100→-150→-200）
            
            # ========== 能量与平滑性（保持不变） ==========
            torques = 0.0
            powers = -2e-5
            dof_vel = 0.0
            dof_acc = -2.5e-7
            action_rate = -0.01
            action_smoothness = 0
            
            # ========== 其他约束 ==========
            feet_stumble = 0.0
            stand_still = 0.0
            foot_clearance = -0.0
    
    class domain_rand(TitaConstraintRoughCfg.domain_rand):
        """域随机化配置"""
        randomize_friction = True
        friction_range = [0.2, 2.75]
        
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        
        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        
        randomize_lag_timesteps = True
        lag_timesteps = 3
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1
        
        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
    
    class asset(TitaConstraintRoughCfg.asset):
        file = '{ROOT_DIR}/resources/tita/urdf/tita_description.urdf'
        foot_name = "leg_4"
        name = "tita"
        penalize_contacts_on = ["leg_3"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        flip_visual_attachments = False


class TitaParkourCfgPPO(TitaConstraintRoughCfgPPO):
    """TITA 跑酷 PPO 训练配置"""
    
    class algorithm(TitaConstraintRoughCfgPPO.algorithm):
        entropy_coef = 0.01
        
        # 学习率配置
        # - 从头训练: 1.e-3（标准学习率）
        # - 微调训练: 3.e-4（降低学习率）
        learning_rate = 1.e-3  # ✅ 从头训练使用标准学习率
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1
    
    class policy(TitaConstraintRoughCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        
        # 编码器配置
        scan_encoder_dims = [128, 64, 32]
        depth_encoder_dims = [64, 32, 16]    # 深度编码器（如果使用原始深度图）
        
        # Actor-Critic 网络
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        
        # 特权编码器
        priv_encoder_dims = []
        
        activation = 'elu'
        
        # RNN 配置（如果使用）
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        
        tanh_encoder_output = False
        num_costs = 6
        
        # 从头训练：禁用教师网络和模仿学习
        teacher_act = False
        imi_flag = False

    class runner(TitaConstraintRoughCfgPPO.runner):
        run_name = 'parkour_with_curriculum'
        experiment_name = 'tita_parkour'
        
        # 使用跑酷专用策略和运行器
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        
        # 训练配置（从头训练需要更多迭代）
        max_iterations = 40000              # 从头训练：增加到 40000 次（约 10-15 小时）
        num_steps_per_env = 24
        
        # 检查点保存
        save_interval = 500
        
        # ========== 从头训练配置（使用深度特征）==========
        # 注意：由于添加了深度特征，观测维度从 586 变为 591
        # 因此不能使用预训练模型，必须从头训练
        resume = False                     # ❌ 不能使用预训练模型
        resume_path = None                 # ❌ 不加载模型
        load_run = None
        checkpoint = -1
        
        # 微调建议：
        # 1. 设置 resume=True, resume_path='tita_example_10000.pt'
        # 2. 将 algorithm.learning_rate 改为 3.e-4（降低到1/3）
        # 3. 可选：将 max_iterations 减少到 25000-30000（因为有基础）
        # 4. 运行: python train_parkour.py --task=tita_parkour --headless
        
        # 示例配置（激活微调）：
        # resume = True
        # resume_path = 'tita_example_10000.pt'
        # max_iterations = 30000
        # algorithm.learning_rate = 3.e-4  # 在 algorithm 类中修改
        
        # 日志
        log_interval = 10
        
        # 课程学习配置（会被 train_parkour.py 使用）
        use_curriculum = True
        
        # 评估
        eval_interval = 500
        num_eval_envs = 100
