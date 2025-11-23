# =====================================================
# ✅ 必须最先导入 IsaacGym
# =====================================================
from isaacgym import gymapi

import torch
import onnxruntime as ort
import numpy as np
from modules.actor_critic import ActorCriticBarlowTwins  # 根据你的路径调整

def compare_models(pt_path, onnx_path, input_shape):
    print(f"📦 加载 PyTorch 模型: {pt_path}")

    # === 1. 实例化模型结构 ===
    model = ActorCriticBarlowTwins(
        obs_shape=input_shape[-1],
        action_size=8,
        num_priv_latent=36,
        num_hist=10,
        num_prop=33,
        num_scan=187,
        activation="elu"
    )

    # === 2. 载入参数 ===
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # === 3. 创建输入 ===
    dummy_input = torch.randn(input_shape, dtype=torch.float32)

    # === 4. PyTorch 推理 ===
    with torch.no_grad():
        torch_out = model(dummy_input).detach().cpu().numpy()

    # === 5. ONNX 推理 ===
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: dummy_input.numpy()})[0]

    # === 6. 对比结果 ===
    diff = np.abs(torch_out - onnx_out)
    print(f"\n✅ 模型对比完成")
    print(f"最大误差: {diff.max():.6f}")
    print(f"平均误差: {diff.mean():.6f}")

    if diff.max() < 1e-4:
        print("🎯 完美匹配（ONNX 转换成功）")
    else:
        print("⚠️ 存在数值差异，请检查导出时是否使用相同参数")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--input_shape", nargs="+", type=int, required=True)
    args = parser.parse_args()

    compare_models(args.pt_path, args.onnx_path, tuple(args.input_shape))

