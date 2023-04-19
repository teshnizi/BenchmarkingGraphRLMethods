import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def aggregate_runs(csv_files):
    runs = {}
    for file in csv_files:
        df = pd.read_csv(file)
        
        _, _, _, env_id, model_type, nodes, edges, parenting = file.split('_')
        nodes = int(nodes[1:])
        edges = int(edges[1:])
        parenting = int(parenting[-5])
        
        key = f'Parenting L{parenting}'
        
        if model_type == 'Transformer' and nodes == 30:
            if key not in runs:
                runs[key] = []
            runs[key].append(df)
        
    return runs


def plot_aggregated_runs(aggregated_runs):
    plt.figure(figsize=(10, 6))

    colors = [ '#648FFF', '#DC267F', '#FE6100', '#785EF0', '#FFB000', "#31467D"]

    for idx, (key, runs) in enumerate(aggregated_runs.items()):
        aggregated_df = pd.concat(runs).groupby('Step', as_index=False).agg({'Value': ['mean', 'sem']})
        aggregated_df.columns = ['Step', 'Mean', 'SEM']
        x = aggregated_df['Step']
        y = aggregated_df['Mean']
        yerr = aggregated_df['SEM'] * 1.96  # 95% confidence interval
        color = colors[idx % len(colors)]
        plt.plot(x, y, label=key, color=color)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.3, facecolor=color)

    
    plt.xlabel('Iteration')
    plt.ylabel('Fraction Solved')
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')
    plt.savefig('Transformer_30.pdf', format='pdf')

csv_folder = 'csv_output'
csv_files = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

aggregated_runs = aggregate_runs(csv_files)
plot_aggregated_runs(aggregated_runs)
