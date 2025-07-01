
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
    MIN_POINT_NUMS = 10
    def __init__(self,bias = True) -> None:
        self.coff = np.array([0,0])
        self.has_bias = bias
    def fit(self,X,y):
        X = X.reshape(-1,1)
        if self.has_bias:
            X = np.hstack((X,np.ones_like(X)))
        self.coff = np.linalg.inv(X.T @ X) @ (X.T @ y)
    
    def predict(self,X):
        return X * self.k + self.bias

    @property
    def k(self):
        return self.coff[0]
    
    @property
    def bias(self):
        return self.coff[-1]

class NaiveFbEstimator():
    def __init__(self):
        self.slope = 2.408
        self.bias = 1.666
    def fit(self,k):
        return self.slope * k + self.bias

def refine_pos(pos,force):
    a = 4.27550573101652
    b = 2.40134727398760
    delta_pos = (np.sqrt(b*b+4*a*force) - b)/2/a    
    return pos - delta_pos

class FbEstimator():

    '''
    Breaking Force estimator.
    '''
    MAX_POINT_NUMS = 60
    DEFAULT_FB = 11.03
    def __init__(self):
        self.finished = False

        self._Fb = None
        self._last_pos = None
        self._pos_list = np.empty(FbEstimator.MAX_POINT_NUMS)
        self._force_list = np.empty(FbEstimator.MAX_POINT_NUMS)
        self._point_cnt = 0
        self.list_ptr = 0
        self.property_est = LinearRegression()
        self.fb_est = NaiveFbEstimator()

    def data_push_in(self, pos, force):
        ret = self._try_data_push_in(pos, force)
        if self._point_cnt >= self.property_est.MIN_POINT_NUMS:
            if self._point_cnt < FbEstimator.MAX_POINT_NUMS:
                self.property_est.fit(self._pos_list[:self._point_cnt],self._force_list[:self._point_cnt])
                self._pred = self.property_est.predict(self._pos_list[:self._point_cnt])
                self._sigma = 1 - np.sum(np.square(self._pred - self._force_list[:self._point_cnt])) / np.square(np.std(self._force_list[:self._point_cnt]))
            else:
                self.property_est.fit(self._pos_list,self._force_list)
                self._pred = self.property_est.predict(self._pos_list[:self._point_cnt])
                self._sigma = 1 - np.sum(np.square(self._pred - self._force_list)) / np.square(np.std(self._force_list))
            # print(self.property_est.k,end=' ')
            self._Fb = self.fb_est.fit(self.property_est.k)
            
            self.finished = True

    def _try_data_push_in(self,pos,force):
        if self._last_pos is not None and abs(self._last_pos - pos) < 0.001:
            return False
        self._last_pos = pos
        self._pos_list[self.list_ptr] = pos
        self._force_list[self.list_ptr] = force
        self._point_cnt +=1
        self.list_ptr = (self.list_ptr + 1) % FbEstimator.MAX_POINT_NUMS
        return True
    
    @property
    def break_force(self):
        return self._Fb if self.finished else FbEstimator.DEFAULT_FB

    @property
    def sigma(self):
        return self._sigma if self.finished else 1


if __name__ == "__main__":
    fb_estimator = FbEstimator()
    file_time = "data_20250630_155851"
    file_name = f'data/{file_time}.csv'
    result_name = f"data/{file_time}.txt"
    data = pd.read_csv(file_name)
    now_force = data['now_force'].to_numpy()
    goal_force = data['goal_force'].to_numpy()
    now_pos = data['now_pos'] .to_numpy()
    time_list = data['time'].to_numpy()
  
    st_point = 100

    now_force = now_force[st_point:]
    goal_force = goal_force[st_point:]
    now_pos = now_pos[st_point:]
    time_list = time_list[st_point:] - time_list[0]

    with open(result_name, 'r') as f:
        str = f.readline().strip()
        gt_fb = float(str.split(' ')[-1])


    fb_list = []
    sigma_list = []
    has_print = False
    st,idx = 0,0
    for force,pos in zip(now_force,now_pos):
        # pos = refine_pos(pos,force)
        fb_estimator.data_push_in(pos,force)
        fb_list.append(fb_estimator.break_force)
        if fb_estimator.break_force > 1 and st == 0:
            st = idx
        sigma_list.append(fb_estimator.sigma)
        if not has_print and fb_estimator.break_force < force * 2:
            print(fb_estimator.break_force,gt_fb)
            has_print = True
        idx += 1

    plt.figure(1)
    bt = np.min(time_list[now_force > gt_fb])
    bf = gt_fb
    plt.scatter(time_list,fb_list,label='Break Force',s=1,c='r')
    plt.fill([bt,bt+1,bt+1,bt],[0,0,15,15],color=[0.8,0.2,0.2],alpha=0.6)
    plt.axhline(y=bf,xmax=bt+1,linestyle=":",color='k')
    plt.plot(time_list,now_force,label='Now Force')
    plt.plot(time_list,goal_force,label="Goal Force")
    plt.ylim([0,15])
    plt.xlim([0,bt+1])
    plt.xlabel('Time(s)')
    plt.ylabel('Force(N)')
    plt.legend(loc='best')

    plt.show()


    
