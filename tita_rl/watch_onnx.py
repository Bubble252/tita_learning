# watch_onnx.py
import onnx
import argparse
from onnx import numpy_helper
import numpy as np

def print_onnx_info(onnx_path):
    print(f"📂 正在分析文件: {onnx_path}")
    
    # 加载模型
    model = onnx.load(onnx_path)
    
    # 检查模型合法性
    onnx.checker.check_model(model)
    print("✅ ONNX 模型合法\n")
    
    # 模型基本信息
    print("=== 模型基本信息 ===")
    print(f"模型名称: {model.graph.name}")
    print(f"输入数: {len(model.graph.input)}")
    for inp in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in inp.type.tensor_type.shape.dim]
        print(f"  输入: {inp.name}, shape={shape}")

    print(f"输出数: {len(model.graph.output)}")
    for out in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in out.type.tensor_type.shape.dim]
        print(f"  输出: {out.name}, shape={shape}")

    # 节点信息
    print(f"\n节点数: {len(model.graph.node)}")
    print(f"初始化参数数: {len(model.graph.initializer)}\n")
    
    # 每个初始化参数信息
    print("=== 初始化参数列表 ===")
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        print(f"  {init.name}: shape={arr.shape}, dtype={arr.dtype}, size={arr.size}")
        
    # 可选：打印每个节点类型
    print("\n=== 节点类型统计 ===")
    node_types = {}
    for node in model.graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
    for op, count in node_types.items():
        print(f"  {op}: {count}")

    print("\n✅ ONNX 文件分析完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=str, help="ONNX 模型路径")
    args = parser.parse_args()
    print_onnx_info(args.onnx_path)

