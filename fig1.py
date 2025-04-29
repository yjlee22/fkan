import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8
})

data = [
    {"architecture": "KAN-1", "method": "FedAvg",
        "accuracy": 79.1874, "round": 498, "std": 0.6423},
    {"architecture": "KAN-1", "method": "FedDyn",
        "accuracy": 79.0997, "round": 479, "std": 1.0024},
    {"architecture": "KAN-1", "method": "FedGamma",
        "accuracy": 79.7428, "round": 498, "std": 0.9892},
    {"architecture": "KAN-1", "method": "FedSAM",
        "accuracy": 79.1231, "round": 493, "std": 1.1864},
    {"architecture": "KAN-1", "method": "FedSMOO",
        "accuracy": 79.4154, "round": 366, "std": 0.9862},
    {"architecture": "KAN-1", "method": "FedSpeed",
        "accuracy": 79.2809, "round": 352, "std": 1.1338},
    {"architecture": "MLP-3", "method": "FedAvg",
        "accuracy": 75.3756, "round": 475, "std": 2.1617},
    {"architecture": "MLP-3", "method": "FedDyn",
        "accuracy": 76.7144, "round": 430, "std": 2.3543},
    {"architecture": "MLP-3", "method": "FedGamma",
        "accuracy": 76.9073, "round": 494, "std": 0.6724},
    {"architecture": "MLP-3", "method": "FedSAM",
        "accuracy": 76.0772, "round": 493, "std": 1.9568},
    {"architecture": "MLP-3", "method": "FedSMOO",
        "accuracy": 76.7904, "round": 448, "std": 2.5853},
    {"architecture": "MLP-3", "method": "FedSpeed",
        "accuracy": 75.9778, "round": 450, "std": 2.3086},
]

color_map = {
    "FedAvg": "#ec8e5a",
    "FedDyn": "#03C75A",
    "FedGamma": "#f7cd5d",
    "FedSAM": "#00C7FD",
    "FedSMOO": "#0068B5",
    "FedSpeed": "#cd5df7"
}

std_values = [d["std"] for d in data]
min_std, max_std = min(std_values), max(std_values)


def normalize(val):
    return (val - min_std) / (max_std - min_std)

fig, ax = plt.subplots(figsize=(3.5, 3.5 * 0.8))

for d in data:
    norm = normalize(d["std"])
    size = 100 + norm * 400   
    alpha = 0.3 + norm * 0.7  
    color = color_map[d["method"]]

    if d["architecture"] == "KAN-1":
        ax.scatter(d["round"], d["accuracy"],
                   s=size, marker='o',
                   facecolors=color, edgecolors='black',
                   alpha=alpha, clip_on=False)
    else:
        ax.scatter(d["round"], d["accuracy"],
                   s=size, marker='o',
                   facecolors='none', edgecolors=color,
                   linewidths=1.5, alpha=alpha, clip_on=False)

ax.margins(x=0.05, y=0.05)

ax.set_ylim(72, 82)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

# Legends
method_handles = [
    Line2D([0], [0], color=color_map[m], linewidth=2)
    for m in color_map
]
arch_handles = [
    Line2D([0], [0], marker='o', color='k',
           markerfacecolor='k', linestyle='None', markersize=6),
    Line2D([0], [0], marker='o', color='k',
           markerfacecolor='none', linestyle='None', markersize=6)
]

leg1 = ax.legend(method_handles, list(color_map.keys()),
                 loc='lower left', frameon=False)
ax.add_artist(leg1)

ax.set_xlabel("Convergence Round")
ax.set_ylabel("Top-1 Test Accuracy (%)")

plt.tight_layout(pad=1.5)

if not os.path.exists('result'):
    os.makedirs('result')

plt.savefig('result/intro.pdf', format='pdf', dpi=300)
plt.savefig('result/intro.svg', format='svg')

fig.set_size_inches(3.5, 3.5 * 0.8)
plt.savefig('result/intro.pdf', format='pdf', dpi=300)
plt.savefig('result/intro.svg', format='svg')
