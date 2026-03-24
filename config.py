# config.py
# Configuration for Dynamic Retrocausal Simulator mini project
# 3 agents, 5x5 grid, 7 steps, Mesa 2.3.0, TCN-based

import torch

# ──────────────────────────────────────────────
# Grid & Simulation Settings
# ──────────────────────────────────────────────
GRID_WIDTH = 5
GRID_HEIGHT = 5
GRID_SIZE = (GRID_WIDTH, GRID_HEIGHT)          # 方便统一使用
MAX_VAL = 5.0                                  # 坐标归一化时的最大值（可选）

NUM_AGENTS = 3
STEPS = 7                                      # 每轮模拟步数 ← 这里补上 STEPS

ALLOW_COLLISIONS_DEFAULT = False               # 默认值，创建 model 时可覆盖

# ──────────────────────────────────────────────
# TCN & Data Parameters
# ──────────────────────────────────────────────
SEQ_LEN = 5                                    # TCN 输入序列长度（过去几步）
FEATURE_DIM = 12                               # 单步特征维度（你设的 2 pos + 10 rel）

TCN_NUM_CHANNELS = [64, 64, 32]
TCN_KERNEL_SIZE = 5
TCN_DROPOUT = 0.1                              # 建议加上 dropout 防过拟合
TCN_OUTPUT_SIZE = 1                            # 二分类：碰撞概率

# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10                                  # early stopping 耐心值
LEARNING_RATE = 0.001
VAL_TEST_SIZE = 0.2
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

MIXED_DATA_PATH = f'{DATA_DIR}/abm_mixed_3agents_5x5_collisions.pkl'
TCN_MODEL_PATH = f'{MODELS_DIR}/tcn_best.pt'          # 建议用更有意义的命名

# 可选：不同实验的模型版本
# TCN_MODEL_PATH_LATEST = f'{MODELS_DIR}/tcn_latest.pt'
# TCN_MODEL_PATH_BEST = f'{MODELS_DIR}/tcn_best.pt'

# ──────────────────────────────────────────────
# Device (自动检测)
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")  # 可选：初始化时打印确认

# ──────────────────────────────────────────────
# Convenience constants
# ──────────────────────────────────────────────
NUM_SIMS_SMALL = 60          # 小规模调试用（30+30）
NUM_SIMS_MEDIUM = 400        # 中等规模验证
NUM_SIMS_LARGE = 2000        # 正式训练/评估用