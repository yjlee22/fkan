import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from matplotlib import rcParams

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'text.usetex': True,
})

base_dirs = ["./seed0", "./seed1", "./seed2", "./seed3", "./seed4"]

data_subdirs = [
    "BloodMNIST_Dirichlet_0.001",
    "BloodMNIST_Dirichlet_0.01",
    "BloodMNIST_Dirichlet_0.1",
    "BloodMNIST_Dirichlet_1.0"
]

dirichlet_degrees = []
for subdir in data_subdirs:
    match = re.search(r'Dirichlet_(\d+\.\d+)', subdir)
    if match:
        dirichlet_degrees.append(float(match.group(1)))

sorted_indices = np.argsort(dirichlet_degrees)
data_subdirs = [data_subdirs[i] for i in sorted_indices]
dirichlet_degrees = [dirichlet_degrees[i] for i in sorted_indices]

fed_methods = [
    "FedAvg", "FedDyn", "FedGamma",
    "FedSAM", "FedSMOO", "FedSpeed"
]

selected_grid_values = ["3", "5", "10"]

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
    try:
        grid_idx = parts.index('grid')
        if grid_idx + 1 < len(parts):
            grid_value = parts[grid_idx + 1]
    except ValueError:
        pass

    method = parts[-1].split('.')[0]

    return dataset, model, layer_num, grid_value, method

all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))))

for seed_idx, base_dir in enumerate(base_dirs):
    for data_subdir in data_subdirs:
        if "_" in data_subdir:
            parts = data_subdir.split('_')
            dataset = parts[0]
            distribution = "_".join(parts[1:])
        else:
            dataset = data_subdir
            distribution = ""

        dir_path = os.path.join(base_dir, data_subdir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        parsed_result = parse_filename(filename)
                        if parsed_result is None:
                            continue

                        dataset_from_file, model, layer_num, grid_value, method = parsed_result

                        if model != "kan" or (grid_value not in selected_grid_values and grid_value is not None):
                            continue

                        data = np.load(file_path)

                        all_results[seed_idx][dataset][distribution][model][layer_num][grid_value][method] = data
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        else:
            print(f"Warning: Directory not found: {dir_path}")

out_dir = "result"
os.makedirs(out_dir, exist_ok=True)

grid_styles = {
    "3": {"color": "#03C75A", "label": "$g=3$", "hatch": None},
    "5": {"color": "#f7cd5d", "label": "$g=5$", "hatch": None},
    "10": {"color": "#0068B5", "label": "$g=10$", "hatch": None}
}

fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.0), sharey=True)
axes = axes.flatten()

model = "kan"
layer_num = 1

for i, method in enumerate(fed_methods):
    ax = axes[i]

    results_by_grid = {grid_value: {} for grid_value in selected_grid_values}

    all_dirichlet_degrees = set()

    for data_subdir in data_subdirs:
        if "_" in data_subdir:
            parts = data_subdir.split('_')
            dataset = parts[0]
            distribution = "_".join(parts[1:])
        else:
            dataset = data_subdir
            distribution = ""

        match = re.search(r'Dirichlet_(\d+\.\d+)', distribution)
        if match:
            dirichlet_degree = float(match.group(1))
            all_dirichlet_degrees.add(dirichlet_degree)
        else:
            continue

        for grid_value in selected_grid_values:
            final_accs = []

            for seed_idx in all_results:
                if (dataset in all_results[seed_idx] and
                    distribution in all_results[seed_idx][dataset] and
                    model in all_results[seed_idx][dataset][distribution] and
                    layer_num in all_results[seed_idx][dataset][distribution][model] and
                    grid_value in all_results[seed_idx][dataset][distribution][model][layer_num] and
                        method in all_results[seed_idx][dataset][distribution][model][layer_num][grid_value]):

                    data = all_results[seed_idx][dataset][distribution][model][layer_num][grid_value][method]
                    if len(data) > 0:
                        final_accs.append(data[-1][1])

            if final_accs:
                results_by_grid[grid_value][dirichlet_degree] = {
                    'mean': np.mean(final_accs),
                    'std': np.std(final_accs)
                }

    all_dirichlet_degrees = sorted(list(all_dirichlet_degrees))

    bar_width = 0.25  
    index = np.arange(len(all_dirichlet_degrees))  

    for j, grid_value in enumerate(selected_grid_values):
        means = []
        stds = []

        for degree in all_dirichlet_degrees:
            if degree in results_by_grid[grid_value]:
                means.append(results_by_grid[grid_value][degree]['mean'])
                stds.append(results_by_grid[grid_value][degree]['std'])
            else:
                means.append(0)
                stds.append(0)

        offset = (j - 1) * bar_width

        style = grid_styles[grid_value]
        bars = ax.bar(index + offset, means, bar_width,
                      yerr=stds, capsize=2,
                      color=style["color"],
                      label=style["label"],
                      hatch=style["hatch"],
                      alpha=0.7)

    ax.set_xticks(index)
    ax.set_xticklabels([str(d) for d in all_dirichlet_degrees])

    ax.set_title(f"{method}")
    ax.set_xlabel("Dirichlet Degree ($\\alpha$)")

    if i % 3 == 0:
        ax.set_ylabel("Top-1 Test Accuracy (\%)")
    else:
        ax.set_ylabel("")

    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    if i == 0:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),   
            fontsize=6,
            frameon=False,
            borderaxespad=0              
        )

plt.tight_layout()
plt.savefig(os.path.join(
    out_dir, "result2.pdf"))
plt.close()