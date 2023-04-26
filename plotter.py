import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.font_manager
import pickle 

from matplotlib import font_manager

# font_dirs = ['fonts/']
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# for font_file in font_files:
#     print(font_file)
#     font_manager.fontManager.addfont(font_file)



def aggregate_runs(csv_files):
    runs = {}
    for file in csv_files:
        df = pd.read_csv(file)
        
        _, _, tm, env_id, model_type, nodes, edges, parenting = file.split('_')
        nodes = int(nodes[1:])
        edges = int(edges[1:])
        parenting = int(parenting[-5])
        # if parenting == 3:
        #     parenting = 2
        if parenting == 2 and float(tm) < 2260000:
            continue
        key = f'Parenting L{parenting}'
        
        # if env_id == "LongestPath-v0" and model_type == 'GNN' and nodes == 30:
        if env_id == "DensestSubgraph-v0" and model_type == 'GNN' and nodes == 30:
            if key not in runs:
                runs[key] = []
            runs[key].append(df)
    return runs


def plot_aggregated_runs(aggregated_runs):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20})
    # plt.rcParams['font.family'] = 'times new roman bold'

    colors = [ '#648FFF', '#DC267F', '#FE6100', '#785EF0', '#FFB000', "#31467D"]

    for idx, (key, runs) in enumerate(aggregated_runs.items()):
        aggregated_df = pd.concat(runs).groupby('Step', as_index=False).agg({'Value': ['mean', 'sem']})
        aggregated_df.columns = ['Step', 'Mean', 'SEM']
        x = aggregated_df['Step']
        y = aggregated_df['Mean']
        yerr = aggregated_df['SEM'] * 1.96  # 95% confidence interval
        color = colors[(5 + idx) % len(colors)]
        
        plt.plot(x, y, label=key, color=color)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.3, facecolor=color)


    plt.xlabel('Iteration')
    plt.ylabel('Graph Density')
    plt.legend()
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0,2,1]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')
    
    plt.savefig('DensestSubgraph_GNN_30_density.pdf', format='pdf')
    
csv_folder = 'csv_output'
csv_files = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

aggregated_runs = aggregate_runs(csv_files)
for run in aggregated_runs.keys():
    print(run, len(aggregated_runs[run]))
plot_aggregated_runs(aggregated_runs)
