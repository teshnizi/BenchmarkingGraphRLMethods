import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def event_to_csv(event_path, csv_path):
    ea = EventAccumulator(event_path)
    ea.Reload()

    if "Eval/mean_sol_cost" in ea.Tags()["scalars"]:
        data = ea.Scalars("Eval/mean_sol_cost")
        df = pd.DataFrame(data, columns=['Wall time', 'Step', 'Value'])
        df.to_csv(csv_path, index=False)

runs_folder = 'runs'
output_folder = 'csv_output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for run_folder in os.listdir(runs_folder):
    run_path = os.path.join(runs_folder, run_folder)
    if "DensestSubgraph-v0" in run_folder:
        if os.path.isdir(run_path):
            for event_file in os.listdir(run_path):
                if event_file.startswith('events.out.tfevents'):
                        event_path = os.path.join(run_path, event_file)
                        csv_file = f'{run_folder}.csv'
                        csv_path = os.path.join(output_folder, csv_file)
                        event_to_csv(event_path, csv_path)
                        break
