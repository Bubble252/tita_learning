from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from envs.no_constrains_legged_robot import Tita
from envs import *
from isaacgym import gymapi
from utils import get_args, task_registry
from global_config import ROOT_DIR

import os
import numpy as np
import torch
import cv2
import onnxruntime as ort


class BarlowTwinsONNXPolicy:
    """
    ONNX 推理包装器，专门针对 ActorCriticBarlowTwins。
    自动处理 obs 切片、历史维度 reshape，支持动态 batch。
    """
    def __init__(self, onnx_path, env_cfg, device='cpu'):
        self.env_cfg = env_cfg
        self.device = device

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        self.input_dim = self.ort_session.get_inputs()[0].shape[1]
        self.n_proprio = env_cfg.env.n_proprio  # 33
        self.n_scan = env_cfg.env.n_scan        # 187
        self.history_len = env_cfg.env.history_len
        self.n_history = self.n_proprio * self.history_len

        expected_dim = self.n_proprio + self.n_history
        if self.input_dim != expected_dim:
            print(f"[WARN] ONNX 输入维度 {self.input_dim} 与计算得到 {expected_dim} 不一致")

    def get_action(self, obs):
        if isinstance(obs, torch.Tensor):
            obs_numpy = obs.cpu().numpy()
        else:
            obs_numpy = obs

        # 提取本体感觉 + 历史本体
        proprio = obs_numpy[:, :self.n_proprio]
        history = obs_numpy[:, self.n_proprio + self.n_scan : self.n_proprio + self.n_scan + self.n_history]
        obs_input = np.concatenate([proprio, history], axis=1).astype(np.float32)

        ort_outputs = self.ort_session.run([self.output_name], {self.input_name: obs_input})
        actions = torch.from_numpy(ort_outputs[0]).to(self.device)
        return actions


def play_on_constraint_policy_runner(args):
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.control.use_filter = True

    # 初始化环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # ONNX 推理器
    onnx_model_path = os.path.join(ROOT_DIR, 'test.onnx')
    policy = BarlowTwinsONNXPolicy(onnx_model_path, env_cfg, device=env.device)

    # 摄像头设置
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512
    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    # 视频设置
    video = None
    img_idx = 0
    video_duration = 40
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0
    last_actions = None

    RECORD_FRAMES = False

    for i in range(num_frames):
        # 指令初始化
        env.commands[:, 0] = 1
        env.commands[:, 1] = 0
        env.commands[:, 2] = 0
        env.commands[:, 3] = 0

        # 使用 ONNX 推理
        actions = policy.get_action(obs)

        # 计算 action_rate
        if last_actions is not None:
            action_rate += torch.sum(torch.abs(last_actions - actions), dim=1)
        last_actions = actions.clone()

        # 计算速度指标
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        # 交互步进
        obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

        # 记录帧
        if RECORD_FRAMES:
            img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((512, 512, 4))[:, :, :3]
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1], img.shape[0]))
            video.write(img)
            img_idx += 1

    print("action rate:", action_rate / (num_frames - 1))
    print("z vel:", z_vel / num_frames)
    print("xy_vel:", xy_vel / num_frames)
    print("feet air reward", feet_air_time / num_frames)

    if video is not None:
        video.release()


if __name__ == '__main__':
    # 注册任务
    task_registry.register("tita_constraint", LeggedRobot, TitaConstraintRoughCfg(), TitaConstraintRoughCfgPPO())

    args = get_args()
    play_on_constraint_policy_runner(args)

