"""
TITA 跑酷机器人环境
继承自 LeggedRobot，添加跑酷专用奖励函数和深度特征提取
"""

import torch
from envs.legged_robot import LeggedRobot
from configs import LeggedRobotCfg


class ParkourRobot(LeggedRobot):
    """
    TITA 跑酷环境
    
    新增功能：
    1. 障碍物检测 (_detect_obstacles_ahead)
    2. 深度特征提取 (_extract_depth_features)
    3. 跑酷专用奖励函数：
       - obstacle_clearance: 障碍物清除
       - jump_timing: 跳跃时机
       - landing_stability: 着陆稳定性
    """
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 跑酷专用缓冲区
        self.last_feet_in_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.jump_triggered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        print("✅ ParkourRobot initialized with parkour-specific reward functions")
    
    def compute_observations(self):
        """
        重写观测计算
        
        暂时不添加深度特征，先确保环境能正常创建
        深度信息仍然通过 depth_buffer 在内部使用（奖励计算）
        """
        # 直接调用父类方法，保持原始观测维度 586
        super().compute_observations()
        
        # TODO: 添加深度特征导致段错误，需要进一步调试
        # 可能的原因：
        # 1. depth_buffer 在首次 compute_observations 时还未初始化
        # 2. 观测维度变化导致网络输入层大小不匹配
        # 3. 某些缓冲区（如 obs_buf）的预分配大小基于原始维度
    
    def _extract_depth_features(self):
        """
        从深度图像提取简单统计特征
        
        Returns:
            torch.Tensor: [num_envs, 5] 深度特征
                - left_depth: 左侧区域平均深度
                - center_depth: 中央区域平均深度  
                - right_depth: 右侧区域平均深度
                - nearest_depth: 最近障碍物距离
                - far_depth: 远处平均深度
        """
        if not hasattr(self, 'depth_buffer'):
            return torch.zeros(self.num_envs, 5, device=self.device)
        
        # 获取最新的深度图像 [num_envs, H, W]
        depth_img = self.depth_buffer[:, -1]
        h, w = depth_img.shape[1], depth_img.shape[2]
        
        # 分区域提取特征
        # 左、中、右三个区域（垂直分割）
        left = depth_img[:, :, :w//3]
        center = depth_img[:, :, w//3:2*w//3]
        right = depth_img[:, :, 2*w//3:]
        
        # 上、下两个区域（水平分割，对应近、远）
        near = depth_img[:, :h//2, :]
        far = depth_img[:, h//2:, :]
        
        # 计算统计特征
        features = torch.cat([
            left.mean(dim=(1,2)).unsqueeze(1),           # 左侧平均深度
            center.mean(dim=(1,2)).unsqueeze(1),         # 中央平均深度
            right.mean(dim=(1,2)).unsqueeze(1),          # 右侧平均深度
            near.min(dim=1)[0].min(dim=1)[0].unsqueeze(1),  # 最近障碍（最小值）
            far.mean(dim=(1,2)).unsqueeze(1),            # 远处平均深度
        ], dim=1)
        
        return features  # [num_envs, 5]
    
    def _detect_obstacles_ahead(self):
        """
        从深度图像检测前方是否有障碍物
        
        Returns:
            torch.Tensor: [num_envs] 0或1，表示是否检测到障碍物
        """
        if not self.cfg.depth.use_camera or not hasattr(self, 'depth_buffer'):
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取最新深度图像
        depth_img = self.depth_buffer[:, -1]  # [num_envs, H, W]
        h, w = depth_img.shape[1], depth_img.shape[2]
        
        # 分析中央区域（机器人正前方）
        center_region = depth_img[:, h//3:2*h//3, w//3:2*w//3]
        
        # 计算中央区域的平均深度
        avg_depth = center_region.mean(dim=(1, 2))
        
        # 如果平均深度小于阈值，说明有障碍物
        # 注意：深度值已经归一化，需要根据实际情况调整阈值
        obstacle_threshold = 0.3  # 可调整（对应约0.5-1.0米的距离）
        obstacle_detected = (avg_depth < obstacle_threshold).float()
        
        return obstacle_detected
    
    # ============ 跑酷专用奖励函数 ============
    
    def _reward_obstacle_clearance(self):
        """
        奖励足部离地高度（鼓励跳跃清除障碍）
        
        计算逻辑：
        1. 检测前方是否有障碍物
        2. 如果有障碍物，奖励足部抬高
        3. 使用足部位置相对于地面高度
        """
        # 检测障碍物
        obstacle_detected = self._detect_obstacles_ahead()
        
        # 计算足部相对地面的高度
        # foot_positions: [num_envs, num_feet, 3]
        # measured_heights: [num_envs, num_height_points]
        foot_heights = self.foot_positions[:, :, 2]  # Z坐标
        
        # 使用基座高度作为参考
        base_height = self.root_states[:, 2]
        relative_foot_height = foot_heights - (base_height.unsqueeze(1) - 0.35)  # 0.35是目标基座高度
        
        # 计算最大足部高度
        max_foot_height = relative_foot_height.max(dim=1)[0]
        
        # 只在检测到障碍物时奖励足部抬高
        clearance = obstacle_detected * torch.clip(max_foot_height, 0, 0.5)
        
        return clearance
    
    def _reward_jump_timing(self):
        """
        奖励在障碍物前适时跳跃
        
        计算逻辑：
        1. 检测障碍物
        2. 检测是否双脚离地（跳跃）
        3. 在障碍物前跳跃时给予奖励
        """
        # 检测障碍物
        obstacle_detected = self._detect_obstacles_ahead()
        
        # 检测是否双脚离地（接触力小于阈值）
        feet_in_air = (self.contact_forces[:, self.feet_indices, 2].abs() < 1.0).all(dim=1).float()
        
        # 在检测到障碍物且双脚离地时给予奖励
        jump_timing_reward = obstacle_detected * feet_in_air
        
        return jump_timing_reward
    
    def _reward_landing_stability(self):
        """
        奖励稳定着陆
        
        计算逻辑：
        1. 检测着陆瞬间（从空中到接触地面）
        2. 评估着陆时的姿态稳定性
        3. 姿态越平稳，奖励越高
        """
        # 检测着陆瞬间
        was_in_air = self.last_feet_in_air
        is_on_ground = (self.contact_forces[:, self.feet_indices, 2].abs() > 1.0).all(dim=1)
        landing = was_in_air & is_on_ground
        
        # 更新状态
        self.last_feet_in_air = ~is_on_ground
        
        # 计算姿态稳定性
        # projected_gravity 接近 [0, 0, -1] 表示姿态平稳
        orientation_penalty = torch.abs(self.projected_gravity[:, :2]).sum(dim=1)
        stability = torch.exp(-orientation_penalty * 5)  # 指数函数，姿态越平稳值越大
        
        # 只在着陆瞬间给予奖励
        landing_reward = landing.float() * stability
        
        return landing_reward
    
    # ============ 重写奖励准备函数 ============
    
    def _prepare_reward_function(self):
        """
        重写父类方法，确保跑酷奖励函数被正确注册
        """
        # 调用父类方法
        super()._prepare_reward_function()
        
        # 验证跑酷奖励函数是否存在
        parkour_rewards = ['obstacle_clearance', 'jump_timing', 'landing_stability']
        for reward_name in parkour_rewards:
            if reward_name in self.reward_scales:
                print(f"  ✅ 跑酷奖励已注册: {reward_name} (权重: {self.reward_scales[reward_name]:.3f})")
    
    def reset_idx(self, env_ids):
        """
        重写reset，清空跑酷专用缓冲区
        """
        super().reset_idx(env_ids)
        
        # 重置跑酷状态
        if len(env_ids) > 0:
            self.last_feet_in_air[env_ids] = False
            self.jump_triggered[env_ids] = False
    
    def post_physics_step(self):
        """
        重写post_physics_step，添加跑酷相关的状态更新
        """
        # 调用父类方法
        super().post_physics_step()
        
        # 这里可以添加额外的跑酷相关状态更新
        # 例如：记录跳跃统计、障碍物通过率等
        pass


# 用于注册任务的辅助函数
def create_parkour_robot(cfg, sim_params, physics_engine, sim_device, headless):
    """创建跑酷机器人环境"""
    return ParkourRobot(cfg, sim_params, physics_engine, sim_device, headless)
