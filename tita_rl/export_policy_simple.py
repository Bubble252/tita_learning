"""
简化版 PT 转 ONNX 工具
不依赖项目代码，只需要 PyTorch
"""
import os
import argparse
import torch
import torch.nn as nn


def inspect_pt_file(pt_path):
    """检查 PT 文件内容"""
    print(f"\n检查模型文件: {pt_path}")
    print("=" * 70)
    
    loaded_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # 显示顶层键
    print("\n📦 顶层键:")
    for key in loaded_dict.keys():
        value = loaded_dict[key]
        if isinstance(value, dict):
            print(f"  • {key} (字典, {len(value)} 个条目)")
        elif isinstance(value, torch.Tensor):
            print(f"  • {key} (张量, shape={value.shape})")
        else:
            print(f"  • {key} ({type(value).__name__})")
    
    # 显示 infos 信息
    if 'infos' in loaded_dict and loaded_dict['infos'] is not None:
        print("\n📋 Infos 内容:")
        for key, value in loaded_dict['infos'].items():
            print(f"  • {key}: {value}")
    elif 'infos' in loaded_dict and loaded_dict['infos'] is None:
        print("\n📋 Infos 内容: None (未保存配置信息)")
    
    # 获取 state_dict
    if 'model_state_dict' in loaded_dict:
        state_dict = loaded_dict['model_state_dict']
    else:
        state_dict = loaded_dict
    
    # 显示模型结构
    print(f"\n🧠 模型层数: {len(state_dict)} 层")
    
    actor_layers = {k: v for k, v in state_dict.items() if 'actor' in k}
    if actor_layers:
        print("\n🎯 Actor 网络结构:")
        for key, value in actor_layers.items():
            print(f"  • {key}: {list(value.shape)}")
    
    # 推断维度
    print("\n📊 推断的维度:")
    num_obs = None
    num_actions = None
    
    for key, value in state_dict.items():
        # 查找第一层输入
        if num_obs is None and 'actor' in key and 'weight' in key:
            if '.0.weight' in key or 'actor.weight' in key:
                num_obs = value.shape[1]
                print(f"  ✓ 观测维度 (输入): {num_obs}")
        
        # 查找输出层
        if 'mu.weight' in key or ('actor' in key and key.endswith('.weight')):
            potential_actions = value.shape[0]
            if num_actions is None or potential_actions < 100:  # 动作数通常不会很大
                num_actions = potential_actions
    
    if num_actions:
        print(f"  ✓ 动作维度 (输出): {num_actions}")
    
    if num_obs is None or num_actions is None:
        print("\n⚠️  警告: 无法完全推断模型维度")
        print("建议手动指定 --num_obs 和 --num_actions 参数")
    
    print("=" * 70 + "\n")
    
    return num_obs, num_actions


def extract_actor_from_pt(pt_path, num_obs=None, num_actions=None):
    """
    从 PT 文件中提取 actor 网络
    返回: (actor_model, num_obs, num_actions)
    """
    loaded_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # 获取 state_dict
    if 'model_state_dict' in loaded_dict:
        state_dict = loaded_dict['model_state_dict']
    else:
        state_dict = loaded_dict
    
    # 自动推断维度
    if num_obs is None or num_actions is None:
        for key, value in state_dict.items():
            if num_obs is None and 'actor' in key and '.0.weight' in key:
                num_obs = value.shape[1]
            if num_actions is None and 'mu.weight' in key:
                num_actions = value.shape[0]
    
    if num_obs is None or num_actions is None:
        raise ValueError(f"无法推断模型维度。请手动指定: --num_obs 和 --num_actions")
    
    # 提取 actor 相关的权重
    actor_dict = {}
    for key, value in state_dict.items():
        if 'actor' in key:
            # 移除 'actor.' 前缀
            new_key = key.replace('actor.', '')
            actor_dict[new_key] = value
    
    # 创建一个简单的 Sequential 模型来包装
    class ActorWrapper(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            # 动态创建网络结构
            self.layers = nn.ModuleDict()
            
            # 按键排序，重建网络
            keys = sorted([k for k in state_dict.keys() if 'weight' in k])
            
            for key in keys:
                if '.weight' in key:
                    layer_name = key.replace('.weight', '')
                    weight = state_dict[key]
                    bias_key = key.replace('weight', 'bias')
                    
                    if 'mu' in key:  # 输出层
                        layer = nn.Linear(weight.shape[1], weight.shape[0])
                    else:
                        layer = nn.Linear(weight.shape[1], weight.shape[0])
                    
                    layer.weight.data = weight
                    if bias_key in state_dict:
                        layer.bias.data = state_dict[bias_key]
                    
                    self.layers[layer_name] = layer
        
        def forward(self, x):
            # 简单前向传播
            for name, layer in self.layers.items():
                x = layer(x)
                if 'mu' not in name:  # 除了最后一层，都加激活函数
                    x = torch.elu(x)
            return x
    
    model = ActorWrapper(actor_dict)
    
    return model, num_obs, num_actions


def export_to_onnx(pt_path, output_path=None, num_obs=None, num_actions=None):
    """将 PT 文件转换为 ONNX"""
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"文件不存在: {pt_path}")
    
    # 设置输出路径
    if output_path is None:
        output_dir = os.path.dirname(pt_path) or '.'
        output_path = os.path.join(output_dir, "policy.onnx")
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print(f"\n🔄 开始转换...")
    print(f"输入: {pt_path}")
    print(f"输出: {output_path}")
    
    # 提取 actor 网络
    try:
        model, num_obs, num_actions = extract_actor_from_pt(pt_path, num_obs, num_actions)
    except Exception as e:
        print(f"\n❌ 提取模型失败: {e}")
        print("\n💡 提示: 尝试使用 --inspect 查看模型结构")
        print("     然后手动指定 --num_obs 和 --num_actions")
        return None
    
    model.eval()
    
    print(f"\n✓ 模型配置: 输入={num_obs}, 输出={num_actions}")
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, num_obs)
    
    # 导出 ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['obs'],
            output_names=['actions'],
            dynamic_axes={
                'obs': {0: 'batch_size'},
                'actions': {0: 'batch_size'}
            }
        )
        print(f"\n✅ 成功! ONNX 文件已保存到: {output_path}")
        
        # 验证文件
        import onnx
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX 模型验证通过")
        except:
            print("⚠️  ONNX 验证跳过 (未安装 onnx 包)")
        
        return output_path
    
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='PT 转 ONNX 工具 (简化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查模型信息
  python export_policy_simple.py --pt_path model.pt --inspect
  
  # 自动转换 (推荐)
  python export_policy_simple.py --pt_path model.pt
  
  # 手动指定维度
  python export_policy_simple.py --pt_path model.pt --num_obs 235 --num_actions 12
  
  # 指定输出路径
  python export_policy_simple.py --pt_path model.pt --output my_policy.onnx
        """
    )
    
    parser.add_argument('--pt_path', type=str, required=True,
                        help='PT 模型文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 ONNX 文件路径')
    parser.add_argument('--num_obs', type=int, default=None,
                        help='观测维度 (自动推断)')
    parser.add_argument('--num_actions', type=int, default=None,
                        help='动作维度 (自动推断)')
    parser.add_argument('--inspect', action='store_true',
                        help='只检查文件，不转换')
    
    args = parser.parse_args()
    
    # 检查模式
    if args.inspect:
        inspect_pt_file(args.pt_path)
        return
    
    # 转换模式
    num_obs, num_actions = inspect_pt_file(args.pt_path)
    
    # 使用命令行参数覆盖推断值
    if args.num_obs:
        num_obs = args.num_obs
    if args.num_actions:
        num_actions = args.num_actions
    
    export_to_onnx(args.pt_path, args.output, num_obs, num_actions)


if __name__ == '__main__':
    main()
