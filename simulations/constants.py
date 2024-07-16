# File name constants
TRAINING_DATA_COLLECTOR_JSON = 'training_data_collector.json'
STATE_FILE = 'state_data.csv.gz'
ACTION_REWARD_POLICY_FILE = 'action_reward_policy_data.csv.gz'
NEXT_STATE_FILE = 'next_state_data.csv.gz'


# These are defaults.
NW_LATENCY_BASE = 0.960
NW_LATENCY_MU = 0.040
NW_LATENCY_SIGMA = 0.0
NUMBER_OF_CLIENTS = 1

ARS_MAPPING = {f'ARS_EXPLR_{i}': (i / 100.0) for i in range(101)}

ARS_SETTINGS = [item for i in range(101) for item in [f'ARS_EXPLR_{i}', f'ARS_DUPL_{i}']]

DQN_EXPLR_SETTINGS = [item for i in range(
    101) for item in [f'DQN_EXPLR_{i}_TRAIN', f'DQN_EXPLR_{i}', f'OFFLINE_DQN_EXPLR_{i}_TRAIN', f'OFFLINE_DQN_EXPLR_{i}']]

DQN_DUPL_SETTINGS = [item for i in range(
    101) for item in [f'DQN_DUPL_{i}_TRAIN', f'DQN_DUPL_{i}', f'OFFLINE_DQN_DUPL_{i}_TRAIN', f'OFFLINE_DQN_DUPL_{i}']]

POLICY_ORDER = ["DQN", 'OFFLINE_DQN', 'DQN_OPTIMIZED', "DQN_DUPL_TRAIN", "DQN_DUPL", "OFFLINE_DQN_DUPL_TRAIN", "OFFLINE_DQN_DUPL",
                "DQN_EXPLR", "OFFLINE_DQN_EXPLR"] + DQN_DUPL_SETTINGS + DQN_EXPLR_SETTINGS + ARS_SETTINGS + ["random", "ARS", "round_robin"]

# POLICY_COLORS = {
#     "ARS": "C0",
#     "random": "C1",
#     "DQN": "C2",
#     "DQN_OPTIMIZED": "C2",
#     "round_robin": "C3",
#     'DQN_EXPLR': "C4",
#     "DQN_DUPL": 'C5',
#     "DQN_DUPL_TRAIN": 'C5',
# } | {f'DQN_EXPLR_{i}': 'C4' for i in range(101)} | {f'DQN_EXPLR_{i}_TRAIN': 'C4' for i in range(101)} | {f'DQN_DUPL_{i}': f'C{5 + i}' for i in range(101)} | {f'DQN_DUPL_{i}_TRAIN': f'C{5 + i}' for i in range(101)} | {f'ARS_{i}': 'C0' for i in range(101)}

# POLICY_COLORS = POLICY_COLORS | {f'OFFLINE_{policy}': color for policy, color in POLICY_COLORS.items()}

POLICY_COLORS = {
    "ARS": "C0",
    "random": "C1",
    "DQN": "C5",
    "DQN_OPTIMIZED": "C2",
    "round_robin": "C3",
    'DQN_EXPLR': "C4",
    "DQN_DUPL": 'C5',
    "DQN_DUPL_TRAIN": 'C5',
}

# Extend POLICY_COLORS with additional keys
POLICY_COLORS.update({f'DQN_EXPLR_{i}': 'C2' for i in range(101)})
POLICY_COLORS.update({f'DQN_EXPLR_{i}_TRAIN': 'C2' for i in range(101)})
POLICY_COLORS.update({f'DQN_DUPL_{i}': f'C4' for i in range(101)})
POLICY_COLORS.update({f'DQN_DUPL_{i}_TRAIN': f'C4' for i in range(101)})
POLICY_COLORS.update({f'ARS_EXPLR_{i}': 'C6' for i in range(101)})
POLICY_COLORS.update({f'ARS_DUPL_{i}': 'C7' for i in range(101)})

# Add 'OFFLINE_' prefix variations
POLICY_COLORS.update({f'OFFLINE_{policy}': color for policy, color in POLICY_COLORS.items()})


# Pareto distribution alpha
ALPHA = 1.1

NUM_TEST_EPSIODES = 3


TRAIN_POLICIES_TO_RUN = [
    # 'round_robin',
    'ARS',
    # 'response_time',
    # 'weighted_response_time',
    # 'random',
    'DQN'
]


EVAL_POLICIES_TO_RUN = [
    # 'round_robin',
    'ARS',
    'DQN',
    'random',
    # 'DQN_EXPLR',
    # 'DQN_DUPL'
] + ['DQN_DUPL_10', 'DQN_DUPL_15', 'DQN_DUPL_20', 'DQN_DUPL_25'] + ['DQN_EXPLR_0', 'DQN_EXPLR_10', 'DQN_EXPLR_15', 'DQN_EXPLR_20', 'DQN_EXPLR_25']
