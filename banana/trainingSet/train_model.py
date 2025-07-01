import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def refine_pos(pos,force):
    a = 4.27550573101652
    b = 2.40134727398760
    delta_pos = (np.sqrt(b*b+4*a*force) - b)/2/a    
    return pos - delta_pos

max_length = 60

data_cnt = 69

new_label_list = np.zeros(data_cnt)

# 确认异常数据
error_dict = {
    5: (0,60),
    8 :(30,180),
    16 : (120,250),
    9: (40,85),
    12: (40,90),
    13 : (120,180),
    24: (30,70),
    29: (0,45),
    30: (0,60),
    31: (0,30),
    32: (70, 110),
    33: (0,60)
}

# for idx in [32]:
#     data_file = f'data_{idx}.npy'
#     label_file = f'label_{idx}.npy'  
#     data = np.load(data_file)

#     plot_pos = (data[:,0] )/95.4929
#     plot_force = data[:,1]  

#     plt.subplot(2,1,1)
#     plt.plot(plot_pos,plot_force)
#     plt.subplot(2,1,2)
#     plt.scatter(plot_pos[70:110],plot_force[70:110])
#     plt.show()

# exit(0)

# 遍历 dataset 文件夹中的每个文件
model = LinearRegression()
max_length = 60
x_list = []
y_list = []
for idx in range(1, data_cnt+1):  
    # 读取 npy 数据
    data_file = f'data_{idx}.npy'
    label_file = f'label_{idx}.npy'

    data = np.load(data_file)
    label_data = np.load(label_file)
    new_label = np.nanmean(data[:,1] / label_data)

    if idx in error_dict:
        interval = error_dict[idx]
        li = interval[0]
        ri = interval[1]
        plot_pos = (data[li:ri,0] )/95.4929
        plot_force = data[li:ri,1]
    else:
        plot_pos = (data[30:30+max_length,0] )/95.4929
        plot_force = data[30:30+max_length,1]
    
    plot_pos = refine_pos(plot_pos,plot_force)

    model.fit(plot_pos.reshape(-1,1),plot_force)
    if abs(model.coef_[0]) >10:
        plt.subplot(2,1,1)
        plt.plot(data[:,0]/95.4929,data[:,1])
        plt.subplot(2,1,2)
        plt.scatter(plot_pos,plot_force)
        plt.title(f'{idx}')
        plt.show()
        continue 
    
    x_list.append(model.coef_[0])
    y_list.append(new_label)

 


plt.scatter(x_list,y_list,color=[1,0,0],s=3,label='data point')
x_data = np.array(x_list)
y_data = np.array(y_list)
model.fit(x_data.reshape(-1,1),y_data)
fit_k = model.coef_[0]
fit_b = model.intercept_
error = np.abs(x_data * fit_k + fit_b - y_data)
print(np.linalg.norm(error,1)/error.size)
print(np.sqrt(np.mean(error**2)))
print(fit_k,fit_b)
df = pd.DataFrame()
df['x'] = x_list
df['y'] = y_list
df.to_csv('train_xy.csv')
print(df.corr('pearson'))
plt.axline((0,fit_b),(1,fit_k+fit_b),linestyle='--')
plt.legend(loc='best')
plt.xlabel('stiffness')
plt.ylabel('break force')

plt.show()