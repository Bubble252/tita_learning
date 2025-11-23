"""
自动导出 ActorCritic/ActorCriticBarlowTwins 为 ONNX
通过精细的 import hook 阻断循环导入
"""

import os
import sys
import copy
import argparse
import subprocess
from isaacgym import gymapi  # 必须最先导入
import torch
import types


class SmartModuleMocker:
    """智能模块mock，提供所需的属性但阻止深层导入"""
    
    def __init__(self):
        self.original_import = None
        # 需要完全阻止的模块
        self.blocked_imports = {
            'utils.task_registry',
            'runner',
            'runner.on_policy_runner',
            'runner.on_constraint_policy_runner',
        }
        
    def __enter__(self):
        self.original_import = __builtins__.__import__
        
        # 预先注入mock模块到sys.modules
        print("预先注入mock模块...")
        
        # Mock runner相关模块
        for module_name in ['runner', 'runner.on_policy_runner', 'runner.on_constraint_policy_runner']:
            mock_module = types.ModuleType(module_name)
            mock_module.__dict__.update({
                '__file__': f'<mock {module_name}>',
                '__package__': module_name.rsplit('.', 1)[0] if '.' in module_name else '',
                'OnConstraintPolicyRunner': type('OnConstraintPolicyRunner', (), {}),
                'OnPolicyRunner': type('OnPolicyRunner', (), {}),
            })
            sys.modules[module_name] = mock_module
            print(f"  ✓ Mock {module_name}")
        
        # Mock task_registry
        task_registry_mock = types.ModuleType('utils.task_registry')
        task_registry_mock.__dict__.update({
            '__file__': '<mock utils.task_registry>',
            '__package__': 'utils',
            'task_registry': types.SimpleNamespace(
                register=lambda *args, **kwargs: None,
                get_task_class=lambda *args: None,
            ),
        })
        sys.modules['utils.task_registry'] = task_registry_mock
        print("  ✓ Mock utils.task_registry")
        
        __builtins__.__import__ = self._custom_import
        return self
    
    def __exit__(self, *args):
        __builtins__.__import__ = self.original_import
    
    def _custom_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """自定义import函数，拦截特定模块"""
        
        # 如果是被阻止的模块，直接返回已经创建的mock
        for blocked in self.blocked_imports:
            if name == blocked or name.startswith(blocked + '.'):
                if name in sys.modules:
                    return sys.modules[name]
        
        # 对于 utils 模块的导入，需要特殊处理
        if name == 'utils' and fromlist:
            # 检查fromlist中是否包含task_registry
            if 'task_registry' in fromlist:
                # 先正常导入utils模块（但会跳过task_registry的实际导入）
                try:
                    # 临时替换utils.task_registry，避免导入错误
                    return self.original_import(name, globals, locals, fromlist, level)
                except ImportError as e:
                    if 'task_registry' in str(e) or 'runner' in str(e):
                        # 如果是task_registry相关的错误，返回一个修改过的utils模块
                        utils_module = sys.modules.get('utils')
                        if utils_module is None:
                            utils_module = types.ModuleType('utils')
                            sys.modules['utils'] = utils_module
                        
                        # 添加task_registry属性
                        if not hasattr(utils_module, 'task_registry'):
                            utils_module.task_registry = types.SimpleNamespace(
                                register=lambda *args, **kwargs: None,
                                get_task_class=lambda *args: None,
                            )
                        return utils_module
                    raise
        
        # 其他模块正常导入
        return self.original_import(name, globals, locals, fromlist, level)


def export_policy_as_onnx(
    pt_path,
    actor_class_name="ActorCritic",
    export_engine=False,
    obs_size=None,
    priv_obs_size=None,
    action_size=None,
    num_priv_latent=None,
    num_hist=None,
    num_prop=None,
    num_scan=None,
    num_critic_obs=None,
    priv_encoder_dims=None,
    scan_encoder_dims=None,
    actor_hidden_dims=None,
    critic_hidden_dims=None,
    activation='elu',
    num_costs=6,
    teacher_act=True,
    imi_flag=True
):
    """
    导出 ActorCritic 或 ActorCriticBarlowTwins 为 ONNX
    """

    print(f"加载权重: {pt_path}")
    loaded_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    # 获取维度信息
    if obs_size is not None and action_size is not None:
        print(f"使用手动指定维度 obs_size={obs_size}, priv_obs_size={priv_obs_size}, action_size={action_size}")
    else:
        try:
            info = loaded_dict.get('infos', {})
            obs_size_auto = info.get('obs_size', None)
            priv_obs_size_auto = info.get('priv_obs_size', None)
            action_size_auto = info.get('action_size', None)
            obs_size = obs_size or obs_size_auto
            priv_obs_size = priv_obs_size or priv_obs_size_auto
            action_size = action_size or action_size_auto
        except:
            pass

        if obs_size is None or action_size is None:
            for k, v in loaded_dict['model_state_dict'].items():
                if "actor.fc1.weight" in k and obs_size is None:
                    obs_size = v.shape[1]
                if "actor.fc_out.weight" in k and action_size is None:
                    action_size = v.shape[0]
            if priv_obs_size is None:
                priv_obs_size = obs_size

    if obs_size is None or action_size is None:
        raise ValueError("无法识别 obs_size 或 action_size，请手动指定。")

    print(f"最终使用 obs_size={obs_size}, priv_obs_size={priv_obs_size}, action_size={action_size}")

    # 使用智能模块mocker阻断循环导入
    print("\n准备导入 actor_critic 模块（使用智能import hook）...")
    with SmartModuleMocker():
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 先导入必要的依赖（绕过循环导入）
        try:
            from modules.actor_critic import ActorCritic, ActorCriticBarlowTwins
        except Exception as e:
            print(f"导入出错: {e}")
            print("尝试直接导入...")
            # 如果还是失败，尝试更激进的方法
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "modules.actor_critic",
                os.path.join(project_root, "modules", "actor_critic.py")
            )
            actor_critic_module = importlib.util.module_from_spec(spec)
            sys.modules['modules.actor_critic'] = actor_critic_module
            spec.loader.exec_module(actor_critic_module)
            ActorCritic = actor_critic_module.ActorCritic
            ActorCriticBarlowTwins = actor_critic_module.ActorCriticBarlowTwins

    print(f"✅ 成功导入 {actor_class_name}")

    actor_critic_class = ActorCriticBarlowTwins if actor_class_name == "ActorCriticBarlowTwins" else ActorCritic

    # 初始化 ActorCritic 或 ActorCriticBarlowTwins
    if actor_class_name == "ActorCriticBarlowTwins":
        info = loaded_dict.get('infos', {}) or {}
        num_priv_latent = num_priv_latent or info.get('num_priv_latent', 36)
        num_hist = num_hist or info.get('num_hist', 10)
        num_prop = num_prop or info.get('num_prop', 33)
        num_scan = num_scan or info.get('num_scan', 187)
        num_critic_obs = num_critic_obs or info.get('num_critic_obs', priv_obs_size)

        priv_encoder_dims = priv_encoder_dims if priv_encoder_dims is not None else []
        scan_encoder_dims = scan_encoder_dims or [128, 64, 32]
        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]

        print(f"\nBarlowTwins 参数:")
        print(f"  num_priv_latent={num_priv_latent}, num_hist={num_hist}")
        print(f"  num_prop={num_prop}, num_scan={num_scan}, num_critic_obs={num_critic_obs}")
        print(f"  priv_encoder_dims={priv_encoder_dims}, scan_encoder_dims={scan_encoder_dims}")
        print(f"  actor_hidden_dims={actor_hidden_dims}, activation={activation}")

        actor_critic = actor_critic_class(
            num_prop=num_prop,
            num_scan=num_scan,
            num_critic_obs=num_critic_obs,
            num_priv_latent=num_priv_latent,
            num_hist=num_hist,
            num_actions=action_size,
            scan_encoder_dims=scan_encoder_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            priv_encoder_dims=priv_encoder_dims,
            num_costs=num_costs,
            teacher_act=teacher_act,
            imi_flag=imi_flag
        )
    else:
        actor_critic = actor_critic_class(
            obs_size,
            priv_obs_size,
            action_size
        )

    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.eval()

    # # ----------------------------
    # # ONNX 导出
    # # ----------------------------
    # path = os.path.join(os.path.dirname(pt_path), "exported")
    # os.makedirs(path, exist_ok=True)
    # onnx_path = os.path.join(path, "policy.onnx")

    # print(f"\n开始导出 ONNX...")

    # if actor_class_name == "ActorCriticBarlowTwins":
    #     model = copy.deepcopy(actor_critic.actor_teacher_backbone).to("cpu")
    #     model.eval()

    #     class BarlowTwinsWrapper(torch.nn.Module):
    #         def __init__(self, model, num_hist, num_prop):
    #             super().__init__()
    #             self.model = model
    #             self.num_hist = num_hist
    #             self.num_prop = num_prop

    #         def forward(self, x):
    #             obs = x[:, :self.num_prop]
    #             obs_hist = x[:, self.num_prop:].reshape(-1, self.num_hist, self.num_prop)
    #             return self.model(obs, obs_hist)

    #     model = BarlowTwinsWrapper(model, num_hist=num_hist, num_prop=num_prop)
    #     dummy_input = torch.randn(1, num_prop + num_hist * num_prop)
    #     print(f"输入维度: {dummy_input.shape}")

    # else:
    #     model = copy.deepcopy(actor_critic.actor).to("cpu")
    #     model.eval()
    #     dummy_input = torch.randn(1, obs_size)
    #     print(f"输入维度: {dummy_input.shape}")

    # print(f"导出到: {onnx_path}")
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     onnx_path,
    #     verbose=False,
    #     input_names=["nn_input"],
    #     output_names=["nn_output"],
    #     export_params=True,
    #     opset_version=13,
    #     dynamic_axes={"nn_input": {0: "batch_size"}, "nn_output": {0: "batch_size"}},
    # )
    # print("✅ 成功导出 ONNX")

    # # TensorRT engine 可选导出
    # if export_engine:
    #     engine_path = onnx_path.replace(".onnx", ".engine")
    #     convert_onnx_to_engine(onnx_path, engine_path)
    
    
    
    
    # ----------------------------
    # ONNX 导出
    # ----------------------------
    path = os.path.join(os.path.dirname(pt_path), "exported")
    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, "policy.onnx")

    print(f"\n开始导出 ONNX...")

    if actor_class_name == "ActorCriticBarlowTwins":
        model = copy.deepcopy(actor_critic.actor_teacher_backbone).to("cpu")
        model.eval()

        class BarlowTwinsWrapper(torch.nn.Module):
            def __init__(self, model, num_hist, num_prop):
                super().__init__()
                self.model = model
                self.num_hist = num_hist
                self.num_prop = num_prop

            def forward(self, x):
                obs = x[:, :self.num_prop]
                obs_hist = x[:, self.num_prop:].reshape(-1, self.num_hist, self.num_prop)
                return self.model(obs, obs_hist)

        model = BarlowTwinsWrapper(model, num_hist=num_hist, num_prop=num_prop)
        dummy_input = torch.randn(1, num_prop + num_hist * num_prop)
        print(f"输入维度: {dummy_input.shape}")

    else:
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        model.eval()
        dummy_input = torch.randn(1, obs_size)
        print(f"输入维度: {dummy_input.shape}")

    print(f"导出到: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=["nn_input"],
        output_names=["nn_output"],
        export_params=True,
        opset_version=13,
        dynamic_axes={"nn_input": {0: "batch_size"}, "nn_output": {0: "batch_size"}},
    )
    print("✅ 成功导出 ONNX")

    # TensorRT engine 可选导出
    if export_engine:
        engine_path = onnx_path.replace(".onnx", ".engine")
        convert_onnx_to_engine(onnx_path, engine_path)


def convert_onnx_to_engine(onnx_path, engine_path):
    trtexec_path = "/home/bubble/下载/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec"

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]

    try:
        print("\nConverting ONNX to TensorRT engine...")
        subprocess.run(command, check=True)
        print("✅ TensorRT engine saved to:", engine_path)
    except subprocess.CalledProcessError as e:
        print("❌ Error during ONNX to TensorRT conversion:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, required=True, help="训练好的 .pt 权重文件")
    parser.add_argument("--actor_class", type=str, default="ActorCritic", help="actor 类型")
    parser.add_argument("--export_engine", action="store_true", help="是否生成 TensorRT engine")
    parser.add_argument('--obs_size', type=int, default=None, help="手动指定 obs_size")
    parser.add_argument('--priv_obs_size', type=int, default=None, help="手动指定 priv_obs_size")
    parser.add_argument('--action_size', type=int, default=None, help="手动指定 action_size")
    parser.add_argument('--num_priv_latent', type=int, default=None, help="BarlowTwins: 特权信息潜在维度")
    parser.add_argument('--num_hist', type=int, default=None, help="BarlowTwins: 历史步数")
    parser.add_argument('--num_prop', type=int, default=None, help="BarlowTwins: 本体感知观测维度")
    parser.add_argument('--num_scan', type=int, default=None, help="BarlowTwins: 扫描观测维度")
    parser.add_argument('--num_critic_obs', type=int, default=None, help="BarlowTwins: Critic观测维度")
    parser.add_argument('--activation', type=str, default='elu', help="激活函数类型")
    args = parser.parse_args()

    export_policy_as_onnx(
        pt_path=args.pt_path,
        actor_class_name=args.actor_class,
        export_engine=args.export_engine,
        obs_size=args.obs_size,
        priv_obs_size=args.priv_obs_size,
        action_size=args.action_size,
        num_priv_latent=args.num_priv_latent,
        num_hist=args.num_hist,
        num_prop=args.num_prop,
        num_scan=args.num_scan,
        num_critic_obs=args.num_critic_obs,
        activation=args.activation
    )
