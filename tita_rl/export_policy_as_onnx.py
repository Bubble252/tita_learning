# from global_config import ROOT_DIR
# import os
# import subprocess

# import isaacgym
# from envs import *
# from utils import get_args, task_registry, get_load_path, class_to_dict
# from modules import ActorCritic, ActorCriticRecurrent

# import numpy as np
# import torch
# import copy
# from modules import ActorCriticRMA,ActorCriticBarlowTwins
# from envs.no_constrains_legged_robot import Tita




# #from utils import task_registry
# #from envs import LeggedRobot
# #from envs.no_constrains_legged_robot import Tita
# #from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO

# #task_registry.register(
# #    "tita_constraint",
# #    LeggedRobot,
# #    TitaConstraintRoughCfg(),
# #    TitaConstraintRoughCfgPPO()
# #)



# def export_policy_as_onnx(args):


#     #env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     _, train_cfg = task_registry.get_cfgs(name=args.task)
    
#     print("env instance:", env)
#     print("train_cfg.runner:", train_cfg.runner)
    
#     print("this is env_cfg.env:")
#     print(vars(env_cfg.env))    
#     print(task_registry.train_cfgs.keys())

#     print("train_cfg.runner.................:", train_cfg.runner)

#     log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
#     #resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
#     #resume_path = get_load_path(log_root, load_run=-1, checkpoint=-1)
#     resume_path = os.path.join(ROOT_DIR, 'logs', 'tita_constraint', 'tita_example_10000.pt')

#     #resume_path = os.path.join(log_root,"Oct01_12-30-13_test_barlowtwins_feetcontact", "model_11700.pt")
#     loaded_dict = torch.load(resume_path)
#     actor_critic_class = eval(train_cfg.runner.policy_class_name)
#     if env_cfg.env.num_privileged_obs is None:
#         env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
#     actor_critic = actor_critic_class(
#         env_cfg.env.num_propriceptive_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
#     ).to(args.rl_device)
#     actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    
    
    
    
    
    
    
    
    
    
#     # export policy as an onnx file
#     path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#     os.makedirs(path, exist_ok=True)
#     path = os.path.join(path, "policy.onnx")
#     model = copy.deepcopy(actor_critic.actor).to("cpu")
#     model.eval()

#     dummy_input = torch.randn(env_cfg.env.num_propriceptive_obs)
#     input_names = ["nn_input"]
#     output_names = ["nn_output"]

#     torch.onnx.export(
#         model,
#         dummy_input,
#         path,
#         verbose=True,
#         input_names=input_names,
#         output_names=output_names,
#         export_params=True,
#         opset_version=13,
#     )
#     engine_path =  os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies/policy.engine')

#     if os.path.exists(engine_path):
#         print(f"Engine file already exists: {engine_path}")
#     else:   
#         print("Exported policy as onnx script to: ", engine_path)
#         convert_onnx_to_engine(engine_path)

# def convert_onnx_to_engine(engine_path):
#     onnx_path = engine_path.replace(".engine", ".onnx")
#     trtexec_path = "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec"

#     command = [
#         trtexec_path,
#         f"--onnx={onnx_path}",
#         f"--saveEngine={engine_path}",
#         "--fp16"  # Use FP16 precision if supported
#     ]

#     try:
#         print("Converting ONNX to TensorRT engine...")
#         subprocess.run(command, check=True)
#         print("Converted TensorRT engine saved to:", engine_path)
#     except subprocess.CalledProcessError as e:
#         print("Error during ONNX to TensorRT conversion:", e)


# if __name__ == '__main__':
#     args = get_args()
#     export_policy_as_onnx(args)





























# from global_config import ROOT_DIR
# import os
# import subprocess
# import isaacgym
# from utils import get_args, task_registry, class_to_dict
# import torch
# import copy
# from modules import ActorCriticBarlowTwins
# from envs.no_constrains_legged_robot import Tita

# def export_policy_as_onnx(args):

#     # ================================
#     # 1️⃣ 实例化环境，获取真实属性
#     # ================================
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     _, train_cfg = task_registry.get_cfgs(name=args.task)

#     print("env instance:", env)
#     print("train_cfg.runner:", train_cfg.runner)

#     # ================================
#     # 2️⃣ 加载训练好的模型
#     # ================================
#     resume_path = os.path.join(ROOT_DIR, 'logs', 'tita_constraint', 'tita_example_10000.pt')
#     loaded_dict = torch.load(resume_path)

#     actor_critic_class = eval(train_cfg.runner.policy_class_name)
#     actor_critic = actor_critic_class(
#         env.num_propriceptive_obs,
#         env.num_privileged_obs if hasattr(env, 'num_privileged_obs') else 0,
#         env.num_actions,
#         **class_to_dict(train_cfg.policy)
#     ).to(args.rl_device)
#     actor_critic.load_state_dict(loaded_dict['model_state_dict'])

#     # ================================
#     # 3️⃣ 导出 ONNX
#     # ================================
#     export_dir = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#     os.makedirs(export_dir, exist_ok=True)
#     onnx_path = os.path.join(export_dir, "policy.onnx")

#     model = copy.deepcopy(actor_critic.actor).to("cpu")
#     model.eval()

#     # dummy input 需要带 batch 维度
#     dummy_input = torch.randn(1, env.num_propriceptive_obs)

#     input_names = ["nn_input"]
#     output_names = ["nn_output"]

#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         verbose=True,
#         input_names=input_names,
#         output_names=output_names,
#         export_params=True,
#         opset_version=13,
#     )

#     print("Exported ONNX policy to:", onnx_path)

#     # ================================
#     # 4️⃣ 转成 TensorRT engine
#     # ================================
#     engine_path = os.path.join(export_dir, "policy.engine")
#     if os.path.exists(engine_path):
#         print(f"Engine file already exists: {engine_path}")
#     else:
#         convert_onnx_to_engine(engine_path)

# def convert_onnx_to_engine(engine_path):
#     onnx_path = engine_path.replace(".engine", ".onnx")
#     trtexec_path = "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec"

#     command = [
#         trtexec_path,
#         f"--onnx={onnx_path}",
#         f"--saveEngine={engine_path}",
#         "--fp16"  # Use FP16 precision if supported
#     ]

#     try:
#         print("Converting ONNX to TensorRT engine...")
#         subprocess.run(command, check=True)
#         print("Converted TensorRT engine saved to:", engine_path)
#     except subprocess.CalledProcessError as e:
#         print("Error during ONNX to TensorRT conversion:", e)


# if __name__ == '__main__':
#     args = get_args()
#     export_policy_as_onnx(args)





























































# from global_config import ROOT_DIR
# import os
# import subprocess
# import isaacgym
# from utils import get_args, task_registry, class_to_dict
# import torch
# import copy
# from modules import ActorCriticBarlowTwins
# from envs import LeggedRobot
# from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
# from envs.no_constrains_legged_robot import Tita

# # ================================
# # ✅ 先注册任务
# # ================================
# task_registry.register(
#     "tita_constraint",
#     LeggedRobot,
#     TitaConstraintRoughCfg(),
#     TitaConstraintRoughCfgPPO()
# )

# def export_policy_as_onnx(args):

#     # ================================
#     # 1️⃣ 实例化环境，获取真实属性
#     # ================================
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     _, train_cfg = task_registry.get_cfgs(name=args.task)

#     print("env instance:", env)
#     print("train_cfg.runner:", train_cfg.runner)

#     # ================================
#     # 2️⃣ 加载训练好的模型
#     # ================================
#     resume_path = os.path.join(ROOT_DIR, 'logs', 'tita_constraint', 'tita_example_10000.pt')
#     loaded_dict = torch.load(resume_path, map_location=args.rl_device)

#     actor_critic_class = eval(train_cfg.runner.policy_class_name)
#     actor_critic = actor_critic_class(
#         env.num_propriceptive_obs,
#         getattr(env, 'num_privileged_obs', 0),
#         env.num_actions,
#         **class_to_dict(train_cfg.policy)
#     ).to(args.rl_device)
#     actor_critic.load_state_dict(loaded_dict['model_state_dict'])

#     # ================================
#     # 3️⃣ 导出 ONNX
#     # ================================
#     export_dir = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#     os.makedirs(export_dir, exist_ok=True)
#     onnx_path = os.path.join(export_dir, "policy.onnx")

#     model = copy.deepcopy(actor_critic.actor).to("cpu")
#     model.eval()

#     # dummy input 需要带 batch 维度
#     dummy_input = torch.randn(1, env.num_propriceptive_obs)

#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         verbose=True,
#         input_names=["nn_input"],
#         output_names=["nn_output"],
#         export_params=True,
#         opset_version=13,
#     )

#     print("Exported ONNX policy to:", onnx_path)

#     # ================================
#     # 4️⃣ 转成 TensorRT engine
#     # ================================
#     engine_path = os.path.join(export_dir, "policy.engine")
#     if os.path.exists(engine_path):
#         print(f"Engine file already exists: {engine_path}")
#     else:
#         convert_onnx_to_engine(engine_path)

# def convert_onnx_to_engine(engine_path):
#     onnx_path = engine_path.replace(".engine", ".onnx")
#     trtexec_path = "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec"

#     command = [
#         trtexec_path,
#         f"--onnx={onnx_path}",
#         f"--saveEngine={engine_path}",
#         "--fp16"  # Use FP16 precision if supported
#     ]

#     try:
#         print("Converting ONNX to TensorRT engine...")
#         subprocess.run(command, check=True)
#         print("Converted TensorRT engine saved to:", engine_path)
#     except subprocess.CalledProcessError as e:
#         print("Error during ONNX to TensorRT conversion:", e)

# if __name__ == '__main__':
#     args = get_args()
#     export_policy_as_onnx(args)


































































# import isaacgym
# from isaacgym import gymapi  # 如果用到 gymapi
# # 其它必要模块
# from global_config import ROOT_DIR
# import os
# import subprocess
# import copy
# # 再导入 PyTorch
# import torch
# # 工具函数
# from utils import get_args, task_registry, class_to_dict
# from modules import ActorCriticBarlowTwins
# from envs.no_constrains_legged_robot import Tita
# from global_config import ROOT_DIR
# import os
# import subprocess
# import torch
# import copy
# from utils import get_args, task_registry, class_to_dict
# from modules import ActorCriticBarlowTwins




# from envs.legged_robot import LeggedRobot
# from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
# from utils import task_registry

# # 注册任务（确保在调用 task_registry.get_cfgs 之前）
# task_registry.register(
#     "tita_constraint",
#     LeggedRobot,                    # 任务类
#     TitaConstraintRoughCfg(),       # env 配置
#     TitaConstraintRoughCfgPPO()    # 训练配置
# )


# def export_policy_as_onnx(args):
#     # ================================
#     # 1️⃣ 获取配置，不初始化 env
#     # ================================
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

#     # 从 env_cfg.env 获取属性
#     num_obs = env_cfg.env.num_observations
#     num_privileged_obs = getattr(env_cfg.env, 'num_privileged_obs', 0)
#     num_actions = env_cfg.env.num_actions

#     print("Task:", args.task)
#     print("num_observations:", num_obs)
#     print("num_privileged_obs:", num_privileged_obs)
#     print("num_actions:", num_actions)
#     print("train_cfg.runner:", train_cfg.runner)

#     # ================================
#     # 2️⃣ 加载训练好的模型
#     # ================================
#     resume_path = os.path.join(ROOT_DIR, 'logs', args.task, 'tita_example_10000.pt')
#     loaded_dict = torch.load(resume_path, map_location=args.rl_device)

#     actor_critic_class = eval(train_cfg.runner.policy_class_name)
#     actor_critic = actor_critic_class(
#         num_obs,
#         num_privileged_obs,
#         num_actions,
#         **class_to_dict(train_cfg.policy)
#     ).to(args.rl_device)
#     actor_critic.load_state_dict(loaded_dict['model_state_dict'])

#     # ================================
#     # 3️⃣ 导出 ONNX
#     # ================================
#     export_dir = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#     os.makedirs(export_dir, exist_ok=True)
#     onnx_path = os.path.join(export_dir, "policy.onnx")

#     model = copy.deepcopy(actor_critic.actor).to("cpu")
#     model.eval()

#     # dummy input 带 batch 维度
#     dummy_input = torch.randn(1, num_obs)

#     input_names = ["nn_input"]
#     output_names = ["nn_output"]

#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         verbose=True,
#         input_names=input_names,
#         output_names=output_names,
#         export_params=True,
#         opset_version=13,
#     )

#     print("✅ Exported ONNX policy to:", onnx_path)

#     # ================================
#     # 4️⃣ 转成 TensorRT engine
#     # ================================
#     engine_path = os.path.join(export_dir, "policy.engine")
#     if os.path.exists(engine_path):
#         print(f"Engine file already exists: {engine_path}")
#     else:
#         convert_onnx_to_engine(engine_path)


# def convert_onnx_to_engine(engine_path):
#     onnx_path = engine_path.replace(".engine", ".onnx")
#     trtexec_path = "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec"

#     command = [
#         trtexec_path,
#         f"--onnx={onnx_path}",
#         f"--saveEngine={engine_path}",
#         "--fp16"
#     ]

#     try:
#         print("Converting ONNX to TensorRT engine...")
#         subprocess.run(command, check=True)
#         print("✅ Converted TensorRT engine saved to:", engine_path)
#     except subprocess.CalledProcessError as e:
#         print("❌ Error during ONNX to TensorRT conversion:", e)


# if __name__ == '__main__':
#     args = get_args()
#     export_policy_as_onnx(args)

































from global_config import ROOT_DIR
import os
import subprocess

import isaacgym
from envs import *
from utils import get_args, task_registry, get_load_path, class_to_dict
from modules import ActorCritic, ActorCriticRecurrent

import numpy as np
import torch
import copy
from modules import ActorCriticRMA,ActorCriticBarlowTwins
from envs.no_constrains_legged_robot import Tita

def export_policy_as_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    print("train_cfg.runner.................:", train_cfg.runner)

    log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    loaded_dict = torch.load(resume_path)
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    actor_critic = actor_critic_class(
        env_cfg.env.num_propriceptive_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    # export policy as an onnx file
    path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    dummy_input = torch.randn(env_cfg.env.num_propriceptive_obs)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    engine_path =  os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies/policy.engine')

    if os.path.exists(engine_path):
        print(f"Engine file already exists: {engine_path}")
    else:   
        print("Exported policy as onnx script to: ", engine_path)
        convert_onnx_to_engine(engine_path)

def convert_onnx_to_engine(engine_path):
    onnx_path = engine_path.replace(".engine", ".onnx")
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"  # Use FP16 precision if supported
    ]

    try:
        print("Converting ONNX to TensorRT engine...")
        subprocess.run(command, check=True)
        print("Converted TensorRT engine saved to:", engine_path)
    except subprocess.CalledProcessError as e:
        print("Error during ONNX to TensorRT conversion:", e)


if __name__ == '__main__':
    args = get_args()
    export_policy_as_onnx(args)