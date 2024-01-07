import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pathlib import Path
import json

AGENTS_PATH = Path(__file__).resolve().parent.parent / "agents"
AGENT_STATS_FILE = "agent_running_stats.json"
AGENT_PLOTS = "Plots"

def graph_training_q_losses(agent, save_path):
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
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / "Traininig-Q-Loss.png")

if __name__ == "__main__":
    agent = "testing_agent"
    save_path = AGENTS_PATH / agent / AGENT_PLOTS
    graph_training_q_losses(agent, save_path)