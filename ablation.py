from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

fed_method = "FedAvg"

def parse_filename(filename):
    parts = os.path.basename(filename).split('_')
    dataset = parts[0]
    model = parts[1]
    if 'layer' not in parts:
        print(f"Warning: 'layer' not found in {filename}, skipping")
        return None
    layer_idx = parts.index('layer')

    is_width_model = False
    width_value = None
    layer_num = None
    layer_part = parts[layer_idx + 1] if layer_idx + 1 < len(parts) else None

    if layer_part and layer_part.startswith('w') and layer_part[1:].isdigit():
        is_width_model = True
        width_value = int(layer_part[1:])
        layer_num = 1  
    else:
        try:
            layer_num = int(layer_part)
            width_value = 1 if layer_num == 1 else None
        except:
            print(f"Warning: Invalid layer number in {filename}, skipping")
            return None

    grid_value = None
    if 'grid' in parts:
        gi = parts.index('grid')
        if gi + 1 < len(parts):
            grid_value = parts[gi + 1]

    method = parts[-1].split('.')[0]

    return dataset, model, layer_num, grid_value, method, is_width_model, width_value


depth_results = defaultdict(lambda: defaultdict(lambda: defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))))
width_results = defaultdict(lambda: defaultdict(lambda: defaultdict(
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
            _, model, layer_num, grid_value, method, is_width_model, width_value = parsed
            if grid_value and grid_value != "5":
                continue
            if method != fed_method:
                continue

            arr = np.load(os.path.join(dir_path, filename))
            if is_width_model:
                width_results[seed_idx][dataset][distribution][model][width_value][grid_value][method] = arr
            else:
                depth_results[seed_idx][dataset][distribution][model][layer_num][grid_value][method] = arr
                if layer_num == 1:
                    width_results[seed_idx][dataset][distribution][model][1][grid_value][method] = arr

out_dir = "result"
os.makedirs(out_dir, exist_ok=True)

depth_styles = {
    1: {'linestyle': '-',  'color': '#03C75A', 'marker': '^', 'markersize': 3, 'linewidth': 0.5},
    3: {'linestyle': '--', 'color': '#f7cd5d', 'marker': 'd', 'markersize': 3, 'linewidth': 0.5},
    5: {'linestyle': '-.', 'color': '#0068B5', 'marker': 'p', 'markersize': 3, 'linewidth': 0.5},
}
width_styles = {
    1: {'linestyle': '-',  'color': '#03C75A', 'marker': '^', 'markersize': 3, 'linewidth': 0.5},
    3: {'linestyle': '--', 'color': '#f7cd5d', 'marker': 'd', 'markersize': 3, 'linewidth': 0.5},
    5: {'linestyle': '-.', 'color': '#0068B5', 'marker': 'p', 'markersize': 3, 'linewidth': 0.5},
}

fixed_grid_value = "5"
dataset = "BloodMNIST"
distribution = "Dirichlet_1.0"
method = fed_method

depth_acc_mean_all = {}
depth_acc_std_all = {}
max_rounds = 0

for layer_num in [1, 3, 5]:
    layer_max = 0
    for seed_idx in depth_results:
        d = depth_results[seed_idx]
        if (dataset in d and distribution in d[dataset] and
            'kan' in d[dataset][distribution] and
            layer_num in d[dataset][distribution]['kan'] and
            fixed_grid_value in d[dataset][distribution]['kan'][layer_num] and
                method in d[dataset][distribution]['kan'][layer_num][fixed_grid_value]):
            length = len(d[dataset][distribution]['kan']
                         [layer_num][fixed_grid_value][method])
            layer_max = max(layer_max, length)
    if layer_max == 0:
        continue
    max_rounds = max(max_rounds, layer_max)

    acc_data = np.zeros((len(depth_results), layer_max))
    seed_count = np.zeros(layer_max)
    for seed_idx in depth_results:
        d = depth_results[seed_idx]
        if (dataset in d and distribution in d[dataset] and
            'kan' in d[dataset][distribution] and
            layer_num in d[dataset][distribution]['kan'] and
            fixed_grid_value in d[dataset][distribution]['kan'][layer_num] and
                method in d[dataset][distribution]['kan'][layer_num][fixed_grid_value]):
            arr = d[dataset][distribution]['kan'][layer_num][fixed_grid_value][method]
            rounds = len(arr)
            acc_data[seed_idx, :rounds] = arr[:, 1]
            seed_count[:rounds] += 1

    mean = np.zeros(layer_max)
    std = np.zeros(layer_max)
    for r in range(layer_max):
        if seed_count[r] > 0:
            vals = acc_data[acc_data[:, r] != 0, r]
            mean[r] = np.mean(vals)
            std[r] = np.std(vals)
    depth_acc_mean_all[layer_num] = mean
    depth_acc_std_all[layer_num] = std

width_acc_mean_all = {}
width_acc_std_all = {}

for width_num in [1, 3, 5]:
    width_max = 0
    for seed_idx in width_results:
        w = width_results[seed_idx]
        if (dataset in w and distribution in w[dataset] and
            'kan' in w[dataset][distribution] and
            width_num in w[dataset][distribution]['kan'] and
            fixed_grid_value in w[dataset][distribution]['kan'][width_num] and
                method in w[dataset][distribution]['kan'][width_num][fixed_grid_value]):
            length = len(w[dataset][distribution]['kan']
                         [width_num][fixed_grid_value][method])
            width_max = max(width_max, length)
    if width_max == 0:
        continue
    max_rounds = max(max_rounds, width_max)

    acc_data = np.zeros((len(width_results), width_max))
    seed_count = np.zeros(width_max)
    for seed_idx in width_results:
        w = width_results[seed_idx]
        if (dataset in w and distribution in w[dataset] and
            'kan' in w[dataset][distribution] and
            width_num in w[dataset][distribution]['kan'] and
            fixed_grid_value in w[dataset][distribution]['kan'][width_num] and
                method in w[dataset][distribution]['kan'][width_num][fixed_grid_value]):
            arr = w[dataset][distribution]['kan'][width_num][fixed_grid_value][method]
            rounds = len(arr)
            acc_data[seed_idx, :rounds] = arr[:, 1]
            seed_count[:rounds] += 1

    mean = np.zeros(width_max)
    std = np.zeros(width_max)
    for r in range(width_max):
        if seed_count[r] > 0:
            vals = acc_data[acc_data[:, r] != 0, r]
            mean[r] = np.mean(vals)
            std[r] = np.std(vals)
    width_acc_mean_all[width_num] = mean
    width_acc_std_all[width_num] = std

global_rounds = np.arange(1, max_rounds + 1)

tick_positions = np.linspace(1, max_rounds, 5, dtype=int)
marker_positions = tick_positions - 1  

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=False)

for layer_num in [1, 3, 5]:
    if layer_num not in depth_acc_mean_all:
        continue
    depth_styles[layer_num]['markevery'] = marker_positions
    data = depth_acc_mean_all[layer_num]
    std = depth_acc_std_all[layer_num]
    rounds = len(data)

    ax1.plot(
        global_rounds[:rounds],
        data,
        **depth_styles[layer_num],
        label=f"{layer_num} layer"
    )
    ax1.fill_between(
        global_rounds[:rounds],
        data - std,
        data + std,
        color=depth_styles[layer_num]['color'],
        alpha=0.1
    )

axins1 = inset_axes(ax1, width="50%", height="50%",
                    bbox_to_anchor=(0.15, 0.25, 0.5, 0.5),
                    bbox_transform=ax1.transAxes, loc='center')

zoom_start = max(int(0.9 * max_rounds), max_rounds - 5)
zoom_end = max_rounds

for layer_num in [1, 3, 5]:
    if layer_num not in depth_acc_mean_all:
        continue
    data = depth_acc_mean_all[layer_num]
    std = depth_acc_std_all[layer_num]

    if len(data) >= zoom_start:
        inset_style = depth_styles[layer_num].copy()
        inset_style['markersize'] = 2
        inset_style['linewidth'] = 0.4
        inset_style['markevery'] = 2

        axins1.plot(
            global_rounds[zoom_start-1:len(data)],
            data[zoom_start-1:],
            **inset_style
        )

axins1.set_xlim(global_rounds[zoom_start-1],
                global_rounds[min(zoom_end-1, len(global_rounds)-1)])
visible_data = []
for layer_num in [1, 3, 5]:
    if layer_num in depth_acc_mean_all:
        data = depth_acc_mean_all[layer_num]
        if len(data) >= zoom_start:
            visible_data.extend(data[zoom_start-1:])

if visible_data:
    y_min = max(min(visible_data) - 2, 0)  
    y_max = min(max(visible_data) + 2, 100)  
    axins1.set_ylim(y_min, y_max)

axins1.set_xticklabels([])
axins1.set_yticklabels([])
axins1.tick_params(axis='both', which='both', length=2)
axins1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

mark_inset(ax1, axins1, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.5, ls="--")

for width_num in [1, 3, 5]:
    if width_num not in width_acc_mean_all:
        continue
    width_styles[width_num]['markevery'] = marker_positions
    data = width_acc_mean_all[width_num]
    std = width_acc_std_all[width_num]
    rounds = len(data)
    param_count = 5 ** width_num
    label = f"{param_count} parameters"

    ax2.plot(
        global_rounds[:rounds],
        data,
        **width_styles[width_num],
        label=label
    )
    ax2.fill_between(
        global_rounds[:rounds],
        data - std,
        data + std,
        color=width_styles[width_num]['color'],
        alpha=0.1
    )

axins2 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(0.15, 0.25, 0.5, 0.5),
                    bbox_transform=ax2.transAxes, loc='center')

for width_num in [1, 3, 5]:
    if width_num not in width_acc_mean_all:
        continue
    data = width_acc_mean_all[width_num]
    std = width_acc_std_all[width_num]

    if len(data) >= zoom_start:
        inset_style = width_styles[width_num].copy()
        inset_style['markersize'] = 2
        inset_style['linewidth'] = 0.4
        inset_style['markevery'] = 2 

        axins2.plot(
            global_rounds[zoom_start-1:len(data)],
            data[zoom_start-1:],
            **inset_style
        )

axins2.set_xlim(global_rounds[zoom_start-1],
                global_rounds[min(zoom_end-1, len(global_rounds)-1)])
visible_data = []
for width_num in [1, 3, 5]:
    if width_num in width_acc_mean_all:
        data = width_acc_mean_all[width_num]
        if len(data) >= zoom_start:
            visible_data.extend(data[zoom_start-1:])

if visible_data:
    y_min = max(min(visible_data) - 2, 0) 
    y_max = min(max(visible_data) + 2, 100)  
    axins2.set_ylim(y_min, y_max)

axins2.set_xticklabels([])
axins2.set_yticklabels([])
axins2.tick_params(axis='both', which='both', length=2)
axins2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.5, ls="--")

ax1.set_xlabel("Global Round")
ax2.set_xlabel("Global Round")
ax1.set_ylabel("Test Accuracy (%)")
ax2.set_ylabel("Test Accuracy (%)")
ax1.set_title("Depth Comparison", fontsize=8)
ax2.set_title("Width Comparison", fontsize=8)
ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

ax1.set_yticks(np.arange(0, 101, 25))
ax2.set_yticks(np.arange(0, 101, 25))
ax1.set_xticks(tick_positions)
ax2.set_xticks(tick_positions)

ax1.legend(loc='lower right', frameon=False, fontsize=8)
ax2.legend(loc='lower right', frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "discuss.pdf"),
    format='pdf',
    bbox_inches='tight'
)