import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

def refine_pos(pos,force):
    # this set of parameters are from DexHand-v1, the fingertip may be of different shape
    # a = 4.27550573
    # b = 2.40134727

    # this set of parameters are from DexHand-v2, the fingertip is more flat
    a = 1.56212941
    b = 4.44046467
    delta_pos = (np.sqrt(b*b+4*a*force) - b)/2/a
    return pos - delta_pos

    # f = a p**2 + b * p
    # a p^2 + bp - f = 0
    # p = -b + sqrt(b^2 + 4af) / 2a

args = ArgumentParser(description="Label data for breaking force estimation.")
# when fb(breaking force) is given, label the data with fb
args.add_argument("--fb", type=float, default=None, help="Breaking force to label the data with.")


data_dir = "data"
data_name = "data_20250630_155851"  # Replace with your actual data file name
data_file = f"{data_dir}/{data_name}.csv"  
fit_range_st = 200
fit_range_ed = 600
break_force = args.parse_args().fb

labeled_data_path = f"labeled_data/{data_name}.txt"
# if os.path.exists(labeled_data_path):
#     # read data from labeled_data.txt
#     print(f"Loading labeled data from {labeled_data_path}")
#     with open(labeled_data_path, "r") as f:
#         line = f.readline().strip()
#         fit_range_st, fit_range_ed, stiffness, stiffness_refine, break_force = map(float, line.split())
#         fit_range_st = int(fit_range_st)
#         fit_range_ed = int(fit_range_ed)



data = pd.read_csv(data_file)

now_force = np.array(data["now_force"])
goal_force = np.array(data["goal_force"])
now_pos = np.array(data["now_pos"])
refined_pos = refine_pos(now_pos, now_force)
k = np.array(data["k"])
time = np.array(data["time"])
time = time - time[0]  # Normalize time to start from 0


stiffness = np.polyfit(now_pos[fit_range_st:fit_range_ed], now_force[fit_range_st:fit_range_ed], 1)[0]
stiffness_refine = np.polyfit(refined_pos[fit_range_st:fit_range_ed], now_force[fit_range_st:fit_range_ed], 1)[0]

plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2,2,1)
plt.scatter(now_pos,now_force, label='Force vs Position', color='blue', s=1)
plt.scatter(now_pos[fit_range_st:fit_range_ed], now_force[fit_range_st:fit_range_ed], color='red', s=2, label='Fitting Range')
# draw a horizontal line at break_force if it is given
if break_force is not None:
    plt.axhline(break_force, color='green', linestyle='--', label='Breaking Force')
plt.xlabel('Position')
plt.ylabel('Force')
plt.title(f'Force vs Position k = {stiffness:.3f}, refine k = {stiffness_refine:.3f}')
plt.grid()
plt.legend()

ax2 = plt.subplot(2,2,2, sharex=ax1)
plt.scatter(refined_pos,now_force, label='Force vs Position', color='blue', s=1)
plt.scatter(refined_pos[fit_range_st:fit_range_ed], now_force[fit_range_st:fit_range_ed], color='red', s=2, label='Fitting Range')
# draw a horizontal line at break_force if it is given
if break_force is not None:
    plt.axhline(break_force, color='green', linestyle='--', label='Breaking Force')
plt.xlabel('Position')
plt.ylabel('Force')
plt.title(f'Force vs Position k = {stiffness:.3f}, refine k = {stiffness_refine:.3f}')
plt.grid()
plt.legend()

ax3 = plt.subplot(2,2,3)
plt.scatter(time, now_force, label='Force over Time', color=[0.4,0.4,1], s=1)
plt.plot(time, goal_force, label='Goal Force', color='orange', linewidth=1)
if break_force is not None:
    plt.axhline(break_force, color='green', linestyle='--', label='Breaking Force')
plt.xlabel('Time (s)')
plt.ylabel('Force')
plt.title('Force over Time')

plt.grid()
plt.legend()

plt.subplot(2,2,4, sharex = ax3)
plt.scatter(time, now_pos, label='Position over Time', color='purple', s=1)
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Position over Time')
plt.grid()
plt.legend()

plt.tight_layout()

plt.show()

if break_force is not None:
    print(f"Estimated Breaking Force: {break_force:.3f} N")
    ret = input("Label this data? [Y/n]:")
    if ret.lower() == 'y':
        with open(f"labeled_data/{data_name}.txt", "w") as f:
            f.write(f"{fit_range_st} {fit_range_ed} {stiffness:.3f} {stiffness_refine:.3f} {break_force:.3f}\n")
        print("Data labeled successfully.")
    else:
        print("Data labeling skipped.")