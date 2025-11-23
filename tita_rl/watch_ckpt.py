import torch
import os
from collections import OrderedDict
from graphviz import Digraph

def print_tree(state_dict):
    """以层级结构打印模型参数"""
    print("\n📦 模型层级结构:")
    total_params = 0
    for name, param in state_dict.items():
        indent_level = name.count(".")
        indent = "  " * indent_level
        param_count = param.numel()
        total_params += param_count
        print(f"{indent}├─ {name:<55} {tuple(param.shape)} ({param_count} params)")
    print(f"└─ 参数总量: {total_params:,}")
    return total_params


def visualize_tree(state_dict, output_path="model_structure"):
    """
    使用 graphviz 生成层级结构图（PNG）
    """
    dot = Digraph(comment="Model Structure", format='png')
    dot.attr(rankdir='LR', bgcolor='white')

    # 添加节点与连接（按层次）
    for name in state_dict.keys():
        parts = name.split('.')
        for i in range(1, len(parts)+1):
            prefix = '.'.join(parts[:i])
            parent = '.'.join(parts[:i-1]) if i > 1 else None

            if prefix not in dot.body:
                dot.node(prefix, label=prefix.split('.')[-1])

            if parent and parent != "":
                dot.edge(parent, prefix)

    out_file = dot.render(output_path, cleanup=True)
    print(f"\n🖼️ 模型结构图已保存: {out_file}")


def analyze_pt(path, visualize=False):
    print(f"📂 正在分析文件: {path}")
    if not os.path.exists(path):
        print("❌ 文件不存在！")
        return

    try:
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"\n📄 文件类型: {type(data)}")

    # 如果是纯 state_dict
    if isinstance(data, OrderedDict):
        print("✅ 检测到纯模型参数 (state_dict)")
        total_params = print_tree(data)
        if visualize:
            visualize_tree(data)

    # 如果是 checkpoint 格式
    elif isinstance(data, dict):
        print(f"🧩 包含键: {list(data.keys())}")

        if 'model_state_dict' in data:
            model_state = data['model_state_dict']
            print("✅ 检测到模型参数部分。")
            total_params = print_tree(model_state)
            if visualize:
                visualize_tree(model_state)
        else:
            total_params = 0

        if 'optimizer_state_dict' in data:
            print("\n⚙️ 检测到优化器状态。")
            opt_state = data['optimizer_state_dict']
            print(f"  包含键: {list(opt_state.keys())}")
            if 'param_groups' in opt_state:
                print(f"  参数组数量: {len(opt_state['param_groups'])}")

        if 'iter' in data:
            print(f"\n⏱️ 训练迭代次数: {data['iter']}")

        if 'infos' in data:
            print(f"🧠 附加信息: {data['infos']}")

    else:
        print("⚠️ 未知文件结构。")

    print("\n✅ 文件分析完成。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze and visualize a PyTorch .pt file")
    parser.add_argument("path", type=str, help="Path to .pt file")
    parser.add_argument("--viz", action="store_true", help="Enable Graphviz visualization (save as PNG)")
    args = parser.parse_args()

    analyze_pt(args.path, visualize=args.viz)

