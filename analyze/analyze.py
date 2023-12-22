import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pathlib import Path
import json

AGENTS_PATH = Path("../agents")
AGENT_STATS_FILE = "agent_running_stats.json"
AGENT_PLOTS = "Plots"

def graph_training_q_losses(agent):
    agent_running_stats_path = AGENTS_PATH / agent / AGENT_STATS_FILE
    with open(agent_running_stats_path) as handle:
        stats = json.load(handle)
    x = []
    y = []
    q_losses = stats["Q_LOSSES"]
    for key in q_losses:
        x.append(int(key))
        y.append(sum(q_losses[key]) / len(q_losses[key]))
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.bar(x, y)
    fig.savefig(AGENTS_PATH, agent, AGENT_PLOTS, "Training-Q-Loss")

if __name__ == "__main__":
    graph_training_q_losses("testing_agent_1")