import os
import subprocess
import argparse

# CRITICAL: isaacgym 必须在 torch 之前导入
import isaacgym

from global_config import ROOT_DIR
from envs import *
from utils import get_args, task_registry, get_load_path, class_to_dict
from modules import ActorCritic, ActorCriticRecurrent, ActorCriticRMA, ActorCriticBarlowTwins
import torch
import copy


def export_policy_as_onnx(args, pt_path=None, output_path=None, convert_engine=False):
    """
    导出策略为 ONNX 格式
    
    Args:
        args: 命令行参数
        pt_path: 指定的 PT 文件路径（如果提供，优先使用这个）
        output_path: 输出 ONNX 文件路径
        convert_engine: 是否转换为 TensorRT engine
    """
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    print("=" * 70)
    print(f"任务: {args.task}")
    print(f"策略类: {train_cfg.runner.policy_class_name}")
    print("=" * 70)
    
    # 确定模型路径
    if pt_path and os.path.exists(pt_path):
        resume_path = pt_path
        print(f"✓ 使用指定的模型: {resume_path}")
    else:
        log_root = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        resume_path = get_load_path(
            log_root, 
            load_run=train_cfg.runner.load_run, 
            checkpoint=train_cfg.runner.checkpoint
        )
        print(f"✓ 从日志加载模型: {resume_path}")
    
    # 加载模型
    print(f"\n📂 加载模型文件...")
    loaded_dict = torch.load(resume_path, map_location='cpu')
    
    # 获取策略类
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    
    # 设置特权观测维度
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_propriceptive_obs
    
    print(f"\n🧠 创建策略网络...")
    print(f"  • 本体观测维度: {env_cfg.env.num_propriceptive_obs}")
    print(f"  • 特权观测维度: {env_cfg.env.num_privileged_obs}")
    print(f"  • 动作维度: {env_cfg.env.num_actions}")
    
    # 创建 actor-critic
    actor_critic = actor_critic_class(
        env_cfg.env.num_propriceptive_obs,
        env_cfg.env.num_privileged_obs,
        env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy)
    ).to('cpu')
    
    # 加载权重
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    print("✓ 权重加载成功")
    
    # 提取 actor
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()
    
    print(f"\n📊 Actor 网络结构:")
    print(model)
    
    # 确定输出路径
    if output_path:
        onnx_path = output_path
    else:
        export_dir = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        os.makedirs(export_dir, exist_ok=True)
        onnx_path = os.path.join(export_dir, "policy.onnx")
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, env_cfg.env.num_propriceptive_obs)  # 添加 batch 维度
    
    # 测试前向传播
    print(f"\n🧪 测试模型...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ 输入: {list(dummy_input.shape)} -> 输出: {list(output.shape)}")
    
    # 导出 ONNX
    print(f"\n🔄 导出 ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX 导出成功: {onnx_path}")
    
    # 可选：转换为 TensorRT engine
    if convert_engine:
        engine_path = onnx_path.replace(".onnx", ".engine")
        convert_onnx_to_engine(onnx_path, engine_path)
    
    return onnx_path


def convert_onnx_to_engine(onnx_path, engine_path, trtexec_path=None):
    """转换 ONNX 为 TensorRT engine"""
    
    if os.path.exists(engine_path):
        print(f"\n⚠️  Engine 文件已存在: {engine_path}")
        return
    
    # 查找 trtexec
    if trtexec_path is None:
        common_paths = [
            "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec",
            "/usr/src/tensorrt/bin/trtexec",
            "trtexec"
        ]
        for path in common_paths:
            if os.path.exists(path) or path == "trtexec":
                trtexec_path = path
                break
    
    if trtexec_path is None:
        print("\n⚠️  未找到 trtexec，跳过 TensorRT 转换")
        print("提示: 可以稍后在 Docker 中转换")
        return
    
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]
    
    try:
        print(f"\n🔄 转换为 TensorRT engine...")
        subprocess.run(command, check=True)
        print(f"✅ Engine 保存成功: {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorRT 转换失败: {e}")
    except FileNotFoundError:
        print(f"❌ 未找到 trtexec: {trtexec_path}")


def main():
    # 先解析已有参数
    args = get_args()
    
    # 手动解析额外参数
    import sys
    pt_path = None
    output_path = None
    convert_engine = False
    trtexec_path = None
    
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--pt_path' and i + 1 < len(argv):
            pt_path = argv[i + 1]
            i += 2
        elif argv[i] == '--output' and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        elif argv[i] == '--to_engine':
            convert_engine = True
            i += 1
        elif argv[i] == '--trtexec_path' and i + 1 < len(argv):
            trtexec_path = argv[i + 1]
            i += 2
        else:
            i += 1
    
    # 显示使用的参数
    if pt_path:
        print(f"📌 指定的模型路径: {pt_path}")
    if output_path:
        print(f"📌 输出路径: {output_path}")
    if convert_engine:
        print(f"📌 将转换为 TensorRT engine")
    
    # 导出
    export_policy_as_onnx(
        args,
        pt_path=pt_path,
        output_path=output_path,
        convert_engine=convert_engine
    )
    
    print("\n" + "=" * 70)
    print("✅ 完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
