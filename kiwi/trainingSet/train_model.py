import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

def refine_pos(pos,force):
    a = 4.27550573101652
    b = 2.40134727398760
    delta_pos = (np.sqrt(b*b+4*a*force) - b)/2/a    
    return pos - delta_pos


def get_hardness(fb,data : pd.DataFrame):
    pos = data['Pos'].to_numpy(dtype=np.float32)
    force = data['Force'].to_numpy(dtype=np.float32)
    goal = data['GoalForce'].to_numpy(dtype=np.float32)


    pos = refine_pos(pos,force)

    fit_idx = np.where(goal < (fb-0.2))[0]
    if len(fit_idx) > 400:
        fit_idx = fit_idx[:400]
    if abs(fb -5.965) < 0.01:
        fit_idx = np.arange(100,200)
    elif abs(fb-5.481) < 0.01:
        fit_idx = np.arange(100,200)
        plt.plot(pos,force)
        plt.plot(pos[fit_idx],force[fit_idx])
        plt.show()
    model = LinearRegression()
    model.fit(pos[fit_idx].reshape(-1,1),force[fit_idx])
    return  model.coef_[0]

fb_list = []
k_list = []
for dir in os.listdir("."):
    if not os.path.isdir(dir):
        continue
    date = dir
    if 'break_force.npy' not in os.listdir(dir):
        continue
    fb = np.load(os.path.join(dir,'break_force.npy')).item()
    data_file = f'{dir}/ForceData_{date}.csv'
    data = pd.read_csv(data_file)
    hardness = get_hardness(fb,data)
    fb_list.append(fb)
    k_list.append(hardness)

plt.scatter(k_list,fb_list,s=3,color=[1,0,0],label='data point')

x_data = np.array(k_list)
y_data = np.array(fb_list)
model = LinearRegression()
model.fit(x_data.reshape(-1,1),y_data)
fit_k = model.coef_[0]
fit_b = model.intercept_
error = np.abs(x_data * fit_k + fit_b - y_data)
print(np.mean(error))
print(np.sqrt(np.mean(error**2)))
print(fit_k,fit_b)
df = pd.DataFrame()

df['x'] = k_list
df['y'] = fb_list
df.to_csv('train_xy.csv')

print(df.corr('pearson'))
plt.axline((0.1,0.1*fit_k+fit_b),(1,fit_k+fit_b),linestyle='--')
plt.legend(loc='best')
plt.xlabel('stiffness')
plt.ylabel('break force')

plt.show()