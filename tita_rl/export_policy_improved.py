import os
import subprocess
import argparse

# CRITICAL: 必须在导入 torch 之前导入 isaacgym
try:
    import isaacgym
except ImportError:
    print("警告: 未找到 isaacgym，如果不需要可以忽略")
    pass

import torch
import copy

# 延迟导入 modules，避免循环依赖
def get_actor_critic_class(policy_class_name):
    """延迟导入策略类"""
    from modules import ActorCritic, ActorCriticRecurrent, ActorCriticRMA, ActorCriticBarlowTwins
    
    class_map = {
        'ActorCritic': ActorCritic,
        'ActorCriticRecurrent': ActorCriticRecurrent,
        'ActorCriticRMA': ActorCriticRMA,
        'ActorCriticBarlowTwins': ActorCriticBarlowTwins
    }
    
    return class_map.get(policy_class_name, ActorCritic)


def export_policy_as_onnx(
    pt_path,
    output_path=None,
    num_obs=48,  # 默认观测维度
    policy_class="ActorCritic",
    convert_to_engine=False,
    trtexec_path=None
):
    """
    将 PyTorch 模型转换为 ONNX 格式
    
    参数:
        pt_path: .pt 模型文件路径
        output_path: 输出 ONNX 文件路径（默认与 pt 文件同目录）
        num_obs: 输入观测维度
        policy_class: 策略类名称
        convert_to_engine: 是否转换为 TensorRT engine
        trtexec_path: trtexec 可执行文件路径
    """
    
    # 检查输入文件
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"模型文件不存在: {pt_path}")
    
    # 设置输出路径
    if output_path is None:
        output_dir = os.path.dirname(pt_path)
        output_path = os.path.join(output_dir, "policy.onnx")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"加载模型: {pt_path}")
    loaded_dict = torch.load(pt_path, map_location='cpu')
    
    # 获取 state_dict
    state_dict = loaded_dict.get('model_state_dict', loaded_dict)
    
    # 尝试从多个来源推断模型维度
    inferred_num_obs = None
    inferred_num_actions = None
    
    # 方法1: 从 infos 字段读取
    if 'infos' in loaded_dict:
        infos = loaded_dict['infos']
        if 'num_obs' in infos:
            inferred_num_obs = infos['num_obs']
        if 'num_actions' in infos:
            inferred_num_actions = infos['num_actions']
    
    # 方法2: 从模型权重推断
    for key, value in state_dict.items():
        # 推断观测维度（从 actor 第一层输入）
        if inferred_num_obs is None:
            if 'actor.0.weight' in key or 'actor.actor.0.weight' in key:
                inferred_num_obs = value.shape[1]  # 输入维度
        
        # 推断动作维度（从 actor 最后一层输出）
        if 'mu.weight' in key or 'actor.mu.weight' in key:
            inferred_num_actions = value.shape[0]  # 输出维度
    
    # 使用推断值或默认值
    if inferred_num_obs is not None:
        num_obs = inferred_num_obs
        print(f"✓ 从模型推断观测维度: {num_obs}")
    else:
        print(f"⚠ 无法推断观测维度，使用指定值: {num_obs}")
    
    if inferred_num_actions is None:
        raise ValueError("无法自动推断动作维度，请检查模型结构")
    
    print(f"✓ 从模型推断动作维度: {inferred_num_actions}")
    print(f"\n模型配置: 观测维度={num_obs}, 动作维度={inferred_num_actions}")
    
    num_actions = inferred_num_actions
    
    # 获取策略类
    actor_critic_class = eval(policy_class)
    
    # 创建模型实例（使用简化的参数）
    try:
        # 尝试创建模型
        actor_critic = actor_critic_class(
            num_obs, 
            num_obs,  # privileged_obs 设为与 obs 相同
            num_actions
        )
    except Exception as e:
        print(f"使用默认参数创建模型失败: {e}")
        print("尝试使用完整配置...")
        # 可以在这里添加更详细的配置
        raise
    
    # 加载权重
    actor_critic.load_state_dict(state_dict)
    
    # 提取 actor 网络
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(num_obs)
    
    print(f"导出 ONNX 模型到: {output_path}")
    
    # 导出为 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX 模型已保存: {output_path}")
    
    # 可选：转换为 TensorRT engine
    if convert_to_engine:
        engine_path = output_path.replace(".onnx", ".engine")
        convert_onnx_to_engine(output_path, engine_path, trtexec_path)
    
    return output_path


def convert_onnx_to_engine(onnx_path, engine_path, trtexec_path=None):
    """
    将 ONNX 模型转换为 TensorRT engine
    """
    if trtexec_path is None:
        # 尝试常见路径
        common_paths = [
            "/usr/src/tensorrt/bin/trtexec",
            "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec",
            "trtexec"  # 系统路径
        ]
        for path in common_paths:
            if os.path.exists(path) or path == "trtexec":
                trtexec_path = path
                break
    
    if trtexec_path is None:
        print("未找到 trtexec，跳过 TensorRT 转换")
        return
    
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]
    
    try:
        print(f"转换为 TensorRT engine: {engine_path}")
        subprocess.run(command, check=True)
        print(f"✓ TensorRT engine 已保存: {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT 转换失败: {e}")
    except FileNotFoundError:
        print(f"未找到 trtexec: {trtexec_path}")


def inspect_pt_file(pt_path):
    """
    检查 PT 文件内容，显示模型信息
    """
    print(f"检查模型文件: {pt_path}")
    print("=" * 60)
    
    loaded_dict = torch.load(pt_path, map_location='cpu')
    
    # 显示顶层键
    print("顶层键:")
    for key in loaded_dict.keys():
        print(f"  - {key}")
    
    # 显示 infos 信息
    if 'infos' in loaded_dict:
        print("\nInfos 内容:")
        for key, value in loaded_dict['infos'].items():
            print(f"  - {key}: {value}")
    
    # 获取 state_dict
    state_dict = loaded_dict.get('model_state_dict', loaded_dict)
    
    # 显示模型结构
    print(f"\n模型层数: {len(state_dict)} 层")
    print("\nActor 网络结构:")
    for key, value in state_dict.items():
        if 'actor' in key:
            print(f"  - {key}: {value.shape}")
    
    # 尝试推断维度
    print("\n推断的维度:")
    for key, value in state_dict.items():
        if 'actor.0.weight' in key or 'actor.actor.0.weight' in key:
            print(f"  观测维度 (输入): {value.shape[1]}")
        if 'mu.weight' in key or 'actor.mu.weight' in key:
            print(f"  动作维度 (输出): {value.shape[0]}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='将 PyTorch 策略模型转换为 ONNX')
    parser.add_argument('--pt_path', type=str, required=True,
                        help='输入的 .pt 模型文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出的 ONNX 文件路径（默认与 pt 文件同目录）')
    parser.add_argument('--num_obs', type=int, default=None,
                        help='观测维度（如果不指定，将自动推断）')
    parser.add_argument('--policy_class', type=str, default='ActorCritic',
                        choices=['ActorCritic', 'ActorCriticRecurrent', 'ActorCriticRMA', 'ActorCriticBarlowTwins'],
                        help='策略类名称（默认: ActorCritic）')
    parser.add_argument('--to_engine', action='store_true',
                        help='是否转换为 TensorRT engine')
    parser.add_argument('--trtexec_path', type=str, default=None,
                        help='trtexec 可执行文件路径')
    parser.add_argument('--inspect', action='store_true',
                        help='只检查 PT 文件信息，不进行转换')
    
    args = parser.parse_args()
    
    # 如果只是检查文件
    if args.inspect:
        inspect_pt_file(args.pt_path)
        return
    
    export_policy_as_onnx(
        pt_path=args.pt_path,
        output_path=args.output,
        num_obs=args.num_obs,
        policy_class=args.policy_class,
        convert_to_engine=args.to_engine,
        trtexec_path=args.trtexec_path
    )


if __name__ == '__main__':
    main()
