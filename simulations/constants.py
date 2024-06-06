# These are defaults.
NW_LATENCY_BASE = 0.960
NW_LATENCY_MU = 0.040
NW_LATENCY_SIGMA = 0.0
NUMBER_OF_CLIENTS = 1


DQN_EXPLR_SETTINGS = [f'DQN_EXPLR_{i}' for i in range(101)]

POLICY_ORDER = ["DQN", "DQN_DUPL", "DQN_EXPLR"] + DQN_EXPLR_SETTINGS + ["random", "ARS", "round_robin"]

POLICY_COLORS = {
    "ARS": "C0",
    "random": "C1",
    "DQN": "C2",
    "round_robin": "C3",
    'DQN_EXPLR': "C4",
    "DQN_DUPL": 'C5',
} | {f'DQN_EXPLR_{i}': 'C4' for i in range(101)}

# Pareto distribution alpha
ALPHA = 1.1
