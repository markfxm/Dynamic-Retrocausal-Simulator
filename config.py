# Configuration file for Dynamic Retrocausal Simulator

# Grid settings
GRID_WIDTH = 5
GRID_HEIGHT = 5
MAX_VAL = 5.0

# Agent settings
NUM_AGENTS = 3
ALLOW_COLLISIONS = False

# Data paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
MIXED_DATA_PATH = f'{DATA_DIR}/abm_mixed_3agents_5x5_collisions.pkl'
TCN_MODEL_PATH = f'{MODELS_DIR}/tcn_model.pt'

# TCN Model parameters
TCN_INPUT_SIZE = 12  # 2 pos + 10 rel
TCN_NUM_CHANNELS = [64, 64, 32]
TCN_KERNEL_SIZE = 5
TCN_OUTPUT_SIZE = 1

# Training parameters
SEQ_LEN = 5
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001
VAL_TEST_SIZE = 0.2
RANDOM_STATE = 42