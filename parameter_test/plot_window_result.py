import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('result_matrix_5.csv',header=None).values.reshape(20,-1)
std = pd.read_csv('std_matrix_5.csv',header=None).values.reshape(20,-1)
min_point_nums = np.arange(2, 41,2)  
window_sizes = np.arange(2, 133,5)  
x = min_point_nums
y = window_sizes
print(x,y)
X, Y = np.meshgrid(x, y, indexing='ij')
print(X.shape, Y.shape, data.shape)
Z = data

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# ax.scatter(X, Y, Z, cmap='viridis')
ax.set_xlabel('Min Point Numbers')
ax.set_ylabel('Window Size')
ax.set_zlabel('Mean Absolute Error')
ax.set_title('Mean Absolute Error vs Min Point Numbers and Window Size')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, std, cmap='viridis', edgecolor='none')
ax.set_xlabel('Min Point Numbers')
ax.set_ylabel('Window Size')
ax.set_zlabel('Standard Deviation')
ax.set_title('Standard Deviation vs Min Point Numbers and Window Size')

plt.show()