import torch
from pathlib import Path

# ================== Core Simulation ==================
GRID_WIDTH = 5
GRID_HEIGHT = 5
NUM_AGENTS = 3
STEPS = 7
SEQ_LEN = 5
MAX_VAL = 5.0
ALLOW_COLLISIONS = True

# ================== TCN Hyperparams ==================
TCN_INPUT_SIZE = 12
TCN_NUM_CHANNELS = [64, 64, 32]
TCN_KERNEL_SIZE = 5
TCN_OUTPUT_SIZE = 1

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001
VAL_TEST_SIZE = 0.2
RANDOM_STATE = 42

# ================== Paths - 超级健壮版 ==================
def get_base_dir():
    """多重保险获取项目根目录"""
    cwd = Path.cwd().resolve()
    
    # 情况1：当前工作目录就是项目根目录（最常见）
    if (cwd / "config.py").exists() or (cwd / "src").exists():
        return cwd
    
    # 情况2：尝试用 __file__
    try:
        file_based = Path(__file__).parent.parent.resolve()
        if (file_based / "config.py").exists() or (file_based / "src").exists():
            return file_based
    except NameError:
        pass
    
    # 情况3：硬编码兜底（根据你的实际路径）
    hard_coded = Path(r"E:\projects\Dynamic-Retrocausal-Simulator").resolve()
    if hard_coded.exists():
        return hard_coded
    
    # 如果都没找到，返回当前工作目录
    return cwd

BASE_DIR = get_base_dir()

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MIXED_DATA_PATH = DATA_DIR / 'abm_mixed_3agents_5x5_collisions.pkl'
TCN_MODEL_PATH = MODELS_DIR / 'tcn_best.pt'

# ================== Device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 调试信息 ==================
print(f"BASE_DIR resolved to: {BASE_DIR}")
print(f"DATA_DIR       : {DATA_DIR}")
print(f"MIXED_DATA_PATH: {MIXED_DATA_PATH}")
print(f"TCN_MODEL_PATH : {TCN_MODEL_PATH}")
print(f"Current working dir: {Path.cwd()}")