import onnx
from onnx import checker
import argparse
import os
import sys

def main(onnx_path, export_png=True, open_netron=True):
    # 1. 加载模型
    print(f"📂 加载模型: {onnx_path}")
    model = onnx.load(onnx_path)

    # 2. 检查模型合法性
    try:
        checker.check_model(model)
        print("✅ ONNX 模型检查通过 (结构合法)")
    except Exception as e:
        print("❌ ONNX 模型检查失败:", e)
        sys.exit(1)

    # 3. 打印模型基本信息
    print("\n=== 模型基本信息 ===")
    print(f"模型名称: {model.graph.name}")
    print(f"输入数: {len(model.graph.input)}")
    for i, inp in enumerate(model.graph.input):
        print(f"  Input[{i}]: {inp.name}")

    print(f"输出数: {len(model.graph.output)}")
    for i, out in enumerate(model.graph.output):
        print(f"  Output[{i}]: {out.name}")

    print(f"节点数: {len(model.graph.node)}")

    # 4. 生成 PNG (需要 pydot + graphviz)
    if export_png:
        try:
            from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
            pydot_graph = GetPydotGraph(
                model.graph,
                name=model.graph.name,
                rankdir="TB",
                node_producer=GetOpNodeProducer("docstring"),
            )
            png_path = os.path.splitext(onnx_path)[0] + ".png"
            pydot_graph.write_png(png_path)
            print(f"📸 已生成模型结构图: {png_path}")
        except Exception as e:
            print("⚠️ 无法生成 PNG，可能缺少 graphviz/pydot:", e)

    # 5. 可选：打开 Netron
    if open_netron:
        try:
            import netron
            print("🌐 使用 Netron 打开模型，可在浏览器中查看")
            netron.start(onnx_path)
        except ImportError:
            print("⚠️ 未安装 netron，可通过 `pip install netron` 使用")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查并可视化 ONNX 模型")
    parser.add_argument("onnx_path", type=str, help="ONNX 文件路径")
    parser.add_argument("--no-png", action="store_true", help="不生成 PNG")
    parser.add_argument("--no-netron", action="store_true", help="不自动打开 Netron")
    args = parser.parse_args()

    main(
        args.onnx_path,
        export_png=not args.no_png,
        open_netron=not args.no_netron
    )

