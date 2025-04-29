import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 8
rcParams['axes.titlesize'] = 8
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8

base_dirs = ["./seed0", "./seed1", "./seed2", "./seed3", "./seed4"]
data_subdirs = ["BloodMNIST_Dirichlet_1.0"]

fed_methods = ["FedAvg", "FedDyn", "FedGamma", "FedSAM", "FedSMOO", "FedSpeed"]

def parse_filename(filename):
    parts = os.path.basename(filename).split('_')
    dataset = parts[0]
    model = parts[1]
    try:
        layer_idx = parts.index('layer')
        layer_num = int(parts[layer_idx + 1])
    except ValueError:
        print(f"Warning: 'layer' not found in {filename}, skipping")
        return None

    grid_value = None
    if 'grid' in parts:
        gi = parts.index('grid')
        if gi + 1 < len(parts):
            grid_value = parts[gi + 1]

    method = parts[-1].split('.')[0]
    return dataset, model, layer_num, grid_value, method


all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))))

for seed_idx, base_dir in enumerate(base_dirs):
    for data_subdir in data_subdirs:
        if "_" in data_subdir:
            dataset, distribution = data_subdir.split('_', 1)
        else:
            dataset, distribution = data_subdir, ""
        dir_path = os.path.join(base_dir, data_subdir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue
        for filename in os.listdir(dir_path):
            if not filename.endswith('.npy'):
                continue
            parsed = parse_filename(filename)
            if parsed is None:
                continue
            _, model, layer_num, grid_value, method = parsed
            if grid_value is not None and grid_value != "5":
                continue
            arr = np.load(os.path.join(dir_path, filename))
            all_results[seed_idx][dataset][distribution][model][layer_num][grid_value][method] = arr

out_dir = "result"
os.makedirs(out_dir, exist_ok=True)

kan_style = {
    'linestyle': '-',
    'color': '#0068B5',
    'marker': '^',
    'markersize': 3,
    'linewidth': 0.5
}
mlp_styles = {
    1: {'linestyle': '--', 'color': '#7a7b78', 'marker': '^', 'markersize': 3, 'linewidth': 0.5},
    2: {'linestyle': '-.', 'color': '#03C75A', 'marker': 's', 'markersize': 3, 'linewidth': 0.5},
    3: {'linestyle': ':', 'color': '#f7cd5d', 'marker': 'd', 'markersize': 3, 'linewidth': 0.5}
}

fixed_grid_value = "5"
dataset = "BloodMNIST"
distribution = "Dirichlet_1.0"

kan_acc_mean_all = {}
kan_acc_std_all = {}
mlp_acc_mean_all = {}
mlp_acc_std_all = {}
max_rounds_all = {}
global_rounds_all = {}

for method in fed_methods:
    has_data = False
    for seed_idx in all_results:
        d = all_results[seed_idx]
        if (dataset in d and distribution in d[dataset] and
            "kan" in d[dataset][distribution] and
            1 in d[dataset][distribution]["kan"] and
            fixed_grid_value in d[dataset][distribution]["kan"][1] and
                method in d[dataset][distribution]["kan"][1][fixed_grid_value]):
            has_data = True
            break
    if not has_data:
        continue

    max_r = 0
    for seed_idx in all_results:
        d = all_results[seed_idx]
        if (dataset in d and distribution in d[dataset] and
            "kan" in d[dataset][distribution] and
            1 in d[dataset][distribution]["kan"] and
            fixed_grid_value in d[dataset][distribution]["kan"][1] and
                method in d[dataset][distribution]["kan"][1][fixed_grid_value]):
            length = len(d[dataset][distribution]["kan"]
                         [1][fixed_grid_value][method])
            max_r = max(max_r, length)
    if max_r == 0:
        continue

    max_rounds_all[method] = max_r
    global_rounds_all[method] = np.arange(1, max_r + 1)

    kan_data = np.zeros((len(all_results), max_r))
    kan_count = np.zeros(max_r)
    mlp_data = {L: np.zeros((len(all_results), max_r)) for L in [1, 2, 3]}
    mlp_count = {L: np.zeros(max_r) for L in [1, 2, 3]}

    for seed_idx in all_results:
        d = all_results[seed_idx]
        if (dataset in d and distribution in d[dataset]):
            if ("kan" in d[dataset][distribution] and
                1 in d[dataset][distribution]["kan"] and
                fixed_grid_value in d[dataset][distribution]["kan"][1] and
                    method in d[dataset][distribution]["kan"][1][fixed_grid_value]):
                arr = d[dataset][distribution]["kan"][1][fixed_grid_value][method]
                rounds = len(arr)
                kan_data[seed_idx, :rounds] = arr[:, 1]
                kan_count[:rounds] += 1
            for L in [1, 2, 3]:
                if ("mlp" in d[dataset][distribution] and
                    L in d[dataset][distribution]["mlp"] and
                    fixed_grid_value in d[dataset][distribution]["mlp"][L] and
                        method in d[dataset][distribution]["mlp"][L][fixed_grid_value]):
                    arr = d[dataset][distribution]["mlp"][L][fixed_grid_value][method]
                    rounds = len(arr)
                    mlp_data[L][seed_idx, :rounds] = arr[:, 1]
                    mlp_count[L][:rounds] += 1

    kan_mean = np.zeros(max_r)
    kan_std = np.zeros(max_r)
    for i in range(max_r):
        if kan_count[i] > 0:
            vals = kan_data[kan_data[:, i] != 0, i]
            kan_mean[i] = np.mean(vals)
            kan_std[i] = np.std(vals)
    kan_acc_mean_all[method] = kan_mean
    kan_acc_std_all[method] = kan_std

    mlp_acc_mean_all[method] = {L: np.zeros(max_r) for L in [1, 2, 3]}
    mlp_acc_std_all[method] = {L: np.zeros(max_r) for L in [1, 2, 3]}
    for L in [1, 2, 3]:
        for i in range(max_r):
            if mlp_count[L][i] > 0:
                vals = mlp_data[L][mlp_data[L][:, i] != 0, i]
                mlp_acc_mean_all[method][L][i] = np.mean(vals)
                mlp_acc_std_all[method][L][i] = np.std(vals)

nrows, ncols = 2, 3
acc_fig, acc_axs = plt.subplots(nrows, ncols, figsize=(7.16, 4.8), sharey=True)
acc_axs = acc_axs.flatten()

for idx, method in enumerate(fed_methods):
    ax = acc_axs[idx]
    if method not in max_rounds_all:
        ax.text(0.5, 0.5, f"No data for\n{method}",
                ha='center', va='center', transform=ax.transAxes)
        continue

    max_r = max_rounds_all[method]
    rounds = global_rounds_all[method]

    tick_positions = np.linspace(1, max_r, 5, dtype=int)
    marker_positions = tick_positions - 1  

    ax.set_xticks(tick_positions)

    kan_style['markevery'] = marker_positions
    for L in mlp_styles:
        mlp_styles[L]['markevery'] = marker_positions

    ax.plot(rounds, kan_acc_mean_all[method], **kan_style, label="KAN-1 layer")
    ax.fill_between(rounds,
                    kan_acc_mean_all[method] - kan_acc_std_all[method],
                    kan_acc_mean_all[method] + kan_acc_std_all[method],
                    color=kan_style['color'], alpha=0.1)

    for L in [1, 2, 3]:
        if np.any(mlp_acc_mean_all[method][L]):
            style = mlp_styles[L]
            ax.plot(rounds,
                    mlp_acc_mean_all[method][L],
                    **style,
                    label=f"MLP-{L} layer")
            ax.fill_between(rounds,
                            mlp_acc_mean_all[method][L] -
                            mlp_acc_std_all[method][L],
                            mlp_acc_mean_all[method][L] +
                            mlp_acc_std_all[method][L],
                            color=style['color'], alpha=0.1)

    ax.set_title(method)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Global Round")
    if idx % ncols == 0:
        ax.set_ylabel("Test Accuracy (%)")

total = len(fed_methods)
for i in range(total, nrows*ncols):
    acc_axs[i].set_visible(False)

handles, labels = acc_axs[0].get_legend_handles_labels()
acc_axs[0].legend(handles, labels, loc='lower right',
                  frameon=False, fontsize=6)

acc_fig.subplots_adjust(wspace=0.15, hspace=0.4)

acc_fig.savefig(os.path.join(out_dir, "result1.pdf"),
                format='pdf', bbox_inches='tight')