import numpy as np
import pandas as pd
import os
class ExpDataLoader():
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        self.inputs = []
        self.labels = []
        self.exp_cnt = 0
        for filename in os.listdir(dir):
            if not filename.endswith('.csv'):
                continue
            labelname = filename.replace('.csv', '.txt')
            if not os.path.exists(os.path.join(dir, labelname)):
                print(f"Label file {labelname} not found for {filename}")
                continue
            file_path = os.path.join(dir, filename)
            break_force_path = os.path.join(dir, labelname)

            # Read data
            data = pd.read_csv(file_path)
            with open(break_force_path, 'r') as f:
                str = f.readline().strip()
                break_force = float(str.split(' ')[-1])

            self.inputs.append({
                'time': data['time'][100:].to_numpy(),
                'now_force': data['now_force'][100:].to_numpy(),
                'goal_force': data['goal_force'][100:].to_numpy(),
                'now_pos': data['now_pos'][100:].to_numpy()
            })
            self.labels.append(break_force)
            self.exp_cnt += 1
        # print(f"Loaded {self.exp_cnt} experiments from {dir}")
    
    def load_data(self):
        return zip(self.inputs, self.labels)
            

if __name__ == "__main__":
    loader = ExpDataLoader()
    