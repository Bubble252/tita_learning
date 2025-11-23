import isaacgym
import torch

import train   # 这一步会触发 __main__ 以外的部分，但不会触发 register
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from envs import LeggedRobot
from utils.task_registry import task_registry

# 手动注册一次
#task_registry.register("tita_constraint", LeggedRobot, TitaConstraintRoughCfg(), TitaConstraintRoughCfgPPO())

print("当前已注册的任务：\n")
for name in task_registry.task_classes.keys():
    print(" -", name)

