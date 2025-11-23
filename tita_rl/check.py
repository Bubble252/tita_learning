import os
from isaacgym import gymapi  # 必须最先导入
import torch

pt_path = "~/桌面/tita/tita_rl/tita_example_10000.pt"
pt_path = os.path.expanduser(pt_path)  # 展开 ~
loaded = torch.load(pt_path, map_location="cpu")
print(loaded.get('infos', {}))

