import os
import matplotlib.pyplot as plt
import numpy as np

labeled_data_folder = "labels"
fb_list = []
k_list = []
refine_k_list = []
for file_name in os.listdir(labeled_data_folder):
    if not file_name.endswith(".txt"):
        continue
    file_path = os.path.join(labeled_data_folder, file_name)
    with open(file_path, "r") as f:
        line = f.readline().strip()
        fit_range_st, fit_range_ed, stiffness, stiffness_refine, break_force = map(float, line.split())
        fit_range_st = int(fit_range_st)
        fit_range_ed = int(fit_range_ed)
        fb_list.append(break_force)
        k_list.append(stiffness)
        refine_k_list.append(stiffness_refine)
    

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
linear_fit = np.polyfit(k_list, fb_list, 1)
x_fit = np.linspace(min(k_list), max(k_list), 100)
y_fit = np.polyval(linear_fit, x_fit)
plt.plot(x_fit, y_fit, label='Linear Fit', color='orange', linewidth=2, alpha=0.7)
plt.scatter(k_list, fb_list, label='Original k vs Break Force', color='blue', s=10)
plt.xlabel('Stiffness (k)')
plt.ylabel('Breaking Force (Fb)')
plt.title('Breaking Force vs Stiffness')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(k_list, fb_list - np.polyval(linear_fit, k_list), 'o', label='Residuals', color='red', markersize=5)
plt.grid()
plt.legend()
plt.tight_layout()



# calculate the correlation coefficient
corr_k_fb = np.corrcoef(k_list, fb_list)[0, 1]
corr_refine_k_fb = np.corrcoef(refine_k_list, fb_list)[0, 1]
print(f"Correlation coefficient between k and Fb: {corr_k_fb:.3f}")
print(f"Correlation coefficient between refined k and Fb: {corr_refine_k_fb:.3f}")

print(f"Data points: {len(fb_list)}")
err = fb_list - np.polyval(linear_fit, k_list)
print(f"MAE error: {np.mean(np.abs(err)):.3f}, Std error: {np.std(err):.3f}")
print(f"Samples mean: {np.mean(fb_list):.3f}, Std: {np.std(fb_list):.3f}")
print(f"fitted slope: {linear_fit[0]:.3f}, fitted intercept: {linear_fit[1]:.3f}")
used_fit_data = {
    "k_list": k_list,
    "fb_list": fb_list,
}
import pandas as pd
df = pd.DataFrame(used_fit_data)
df.to_csv("used_fit_data.csv", index=False)
plt.show()