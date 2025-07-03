import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("alpha_results.csv")
result_mean = data['Mean MAE'].values
result_std = data['Std MAE'].values

# 绘制结果
plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(len(result_mean)), result_mean, yerr=result_std, fmt='o', capsize=5, label='Mean MAE with Std Dev')
plt.title('Mean MAE with Standard Deviation for Different Alpha Values')
plt.xlabel('Alpha Value Index')
plt.ylabel('Mean Absolute Error (MAE)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()