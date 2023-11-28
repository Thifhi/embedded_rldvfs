import os
from collections import OrderedDict

DEEP_Q_ALG = "Deep-Q"
TABLE_Q_ALG = "Table-Q"

# Used by Trainer
AGENT_NAME = "dq-test1"
ALGORITHM = DEEP_Q_ALG
TEST_EPOCH = 4
TRAIN_UNTIL = 48

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_FOLDER = os.path.join(HERE, "Agents")
AGENT_LOG_PATH = "Logs"

USE_PRETRAINED = False
PRETRAINED_PATH = AGENTS_FOLDER

# Used by Agent
AGENT_CONFIG_PATH = "agent_config.json"
POLICY_NET_PATH = "policy_net.pt"
TARGET_NET_PATH = "target_net.pt"
REPLAY_BUFFER_PATH = "replay-buffer.pickle"
AGENT_RUNNING_STATS_PATH = "agent_running_stats.json"
Q_TABLE_PATH = "q_table.npy"

DEFAULT_DEEP_Q_CONFIG = OrderedDict([
    # Network Config
    ("STATE_ITEMS", ["cpi", "temperature", "frequency"]),
    ("ACTIONS", [1000 + i * 250 for i in range(13)]),

    ("LEARNING_RATE", 0.0005),
    ("GAMMA", 0.9),
    ("EPSILON", 1.0),
    ("EPSILON_MIN", 0.01),
    ("EPSILON_DECAY_PER_RUN", 0.04),

    ("FC1_OUT", 48),
    ("FC2_OUT", 48),

    ("BATCH_SIZE", 100),
    ("REPLAY_BUFFER_CAPACITY", 100000),

    ("TARGET_UPDATE_FREQUENCY", 1000),

    # Reward Constant
    ("K_Frequency", 0.001),
    ("K_Temperature", 5),
    ("TEMPERATURE_CONSTRAINT", 70)
])

DEFAULT_TABLE_Q_CONFIG = OrderedDict([
    # Network Config
    ("STATE_ITEMS", ["cpi", "temperature", "max_neighbor_temp", "average_neighbor_temp", "frequency", "power"]),
    ("ACTIONS", [1000 + i * 250 for i in range(13)]),

    ("CPI_STEPS", [0 + i * 0.2 for i in range(10)]),
    ("TEMPERATURE_STEPS", [62 + i for i in range(18)]),
    ("FREQUENCY_STEPS", [1000 + i * 250 for i in range(13)]),
    ("POWER_STEPS", [1 + 0.2 * i for i in range(16)]),

    ("LEARNING_RATE", 0.1),
    ("GAMMA", 0.9),
    ("EPSILON", 1.0),
    ("EPSILON_MIN", 0.01),
    ("EPSILON_DECAY_PER_RUN", 0.04),

    # Reward Constant
    ("K_Frequency", 0.001),
    ("K_Temperature", 5),
    ("TEMPERATURE_CONSTRAINT", 70)
])

DEFAULT_RUNNING_STATS = OrderedDict([
    ("RUN_COUNTER", 0),
    ("LEARN_COUNTER", 0),
    ("REWARDS", OrderedDict([("TRAIN", OrderedDict()), ("TEST", OrderedDict())]))
])