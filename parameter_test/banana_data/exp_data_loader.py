import numpy as np
import pandas as pd
import os
class ExpDataLoader():
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        self.inputs = []
        self.labels = []
        self.exp_cnt = 0
        for exp_folder in os.listdir(dir):
            if exp_folder.startswith('_'):
                continue # Skip hidden folders
            if not os.path.isdir(os.path.join(dir,exp_folder)):
                continue
            exp_time = exp_folder
            exp_folder = os.path.join(dir,exp_folder)
            file = f"ForceData_{exp_time}_refined.csv"
            file = os.path.join(exp_folder,file)
            if not os.path.exists(file):
                print(f"File {file} not found in {exp_folder}")
                continue
            break_force_path = os.path.join(exp_folder,'break_force.npy')
            if not os.path.exists(break_force_path):
                print(f"Break force file {break_force_path} not found in {exp_folder}")
                continue

            # Read data
            data = pd.read_csv(file)
            break_force = np.load(break_force_path).item()

            self.inputs.append({
                'time': data['Time'].to_numpy(),
                'now_force': data['Force'].to_numpy(),
                'goal_force': data['GoalForce'].to_numpy(),
                'now_pos': data['Pos'].to_numpy()
            })
            self.labels.append(break_force)
            self.exp_cnt += 1
        # print(f"Loaded {self.exp_cnt} experiments from {dir}")
    
    def load_data(self):
        return zip(self.inputs, self.labels)
            

if __name__ == "__main__":
    loader = ExpDataLoader()
    