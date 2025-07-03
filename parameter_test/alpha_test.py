from banana_data import BananaExpDataLoader
from kiwi_data import KiwiExpDataLoader
from mango_data import MangoExpDataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tqdm
import concurrent.futures
from functools import partial
import threading


class LinearRegression():
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
    def __init__(self, slope, bias):
        self.slope = slope
        self.bias = bias
    def fit(self,k):
        return self.slope * k + self.bias


class FbEstimator:
    """
    Breaking Force estimator.
    """

    FRUIT_PARAM_DICT = {
        "banana": {
            "fb0": 7.008,
            "std0": 2.916,
            "gamma": 3.33,
            "b": 0.254,
            "std": 1.618,
        },
        "kiwi": {
            "fb0": 3.652,
            "std0": 1.750,
            "gamma": 2.78,
            "b": 0.345,
            "std": 0.959,
        },
        "mango": {
            "fb0": 11.033,
            "std0": 2.346,
            "gamma": 2.408,
            "b": 1.666,
            "std": 1.280,
        },
    }

    def __init__(self, fruit_type, max_point_nums=60, min_point_nums=10):
        self.filled = False
        self.fruit_type = fruit_type
        self.fruit_param_dict = FbEstimator.FRUIT_PARAM_DICT[fruit_type]
        self.default_fb = self.fruit_param_dict["fb0"]
        self.slope = self.fruit_param_dict["gamma"]
        self.bias = self.fruit_param_dict["b"]
        self.max_point_nums = max_point_nums
        self.min_point_nums = min_point_nums

        self._Fb = None
        self._last_pos = None
        self._pos_list = np.empty(max_point_nums)
        self._force_list = np.empty(max_point_nums)
        self._point_cnt = 0
        self.list_ptr = 0
        self.property_est = LinearRegression()
        self.fb_est = NaiveFbEstimator(self.slope, self.bias)

    def data_push_in(self, pos, force):
        ret = self._try_data_push_in(pos, force)
        if self._point_cnt >= self.min_point_nums:
            if self._point_cnt < self.max_point_nums:
                self.property_est.fit(self._pos_list[: self._point_cnt], self._force_list[: self._point_cnt])
                self._pred = self.property_est.predict(self._pos_list[: self._point_cnt])
            else:
                self.property_est.fit(self._pos_list, self._force_list)
                self._pred = self.property_est.predict(self._pos_list[: self._point_cnt])
            self._Fb = self.fb_est.fit(self.property_est.k)

            self.filled = True

    def _try_data_push_in(self, pos, force):

        self._last_pos = pos
        self._pos_list[self.list_ptr] = pos
        self._force_list[self.list_ptr] = force
        self._point_cnt += 1
        self.list_ptr = (self.list_ptr + 1) % self.max_point_nums
        return True

    @property
    def break_force(self):
        return self._Fb if self.filled else self.default_fb

class ExpExecuter():
    def __init__(self):
        self.fruit_data_loader = {
            "banana": BananaExpDataLoader(),
            "kiwi": KiwiExpDataLoader(),
            "mango": MangoExpDataLoader(),
        }
        self.gt = []
        self.pred = []

    def run_one_config(self, min_point_nums, max_point_nums, alpha_l):
        if min_point_nums > max_point_nums:
            return None,None
        total_loss = []
        for fruit_type, data_loader in self.fruit_data_loader.items():
            for data,label in data_loader.load_data():
                mae = self.run_one_data(data, label, fruit_type, min_point_nums, max_point_nums, alpha_l)
                if mae is not None:
                    total_loss.append(mae)
        if len(total_loss) == 0:
            # print(f"No valid data found.")
            return None, None
        else:
            return np.mean(total_loss), np.std(total_loss)


    def run_one_data(self, data, label, fruit_type, min_point_nums, max_point_nums, alpha_l):
        fb_est = FbEstimator(fruit_type, max_point_nums, min_point_nums)
        for pos, force in zip(data['now_pos'], data['now_force']):
            fb_est.data_push_in(pos, force)
            break_force = fb_est.break_force
            if break_force * alpha_l < force:
                # print(f"{break_force:.3f} {label:.3f}")
                self.gt.append(label)
                self.pred.append(break_force)
                return abs(break_force - label)
        return None
    
    def run_point_num_matrix_config(self, min_p_range, win_size_range, alpha_l, n_jobs=4):
        
        # 准备参数网格
        # min_p_range = range(min_point_nums[0], min_point_nums[1] + 1)
        # win_size_range = range(window_size[0], window_size[1] + 1)
        total_tasks = len(min_p_range) * len(win_size_range)
        
        # 初始化结果矩阵
        result_matrix = np.zeros((len(min_p_range), len(win_size_range)))
        result_matrix[:] = np.nan  # 初始化为NaN，表示未计算
        std_matrix = np.zeros((len(min_p_range), len(win_size_range)))
        std_matrix[:] = np.nan  # 初始化为NaN，表示未计算
        
        # 创建线程池执行器
        with tqdm.tqdm(total=total_tasks, desc="Running configurations") as pbar:
            # 使用锁保证进度条更新的线程安全
            pbar_lock = threading.Lock()
            
            # 任务处理函数
            def process_task(min_p, max_p,i,j):
                ret,std = self.run_one_config(min_point_nums=min_p, max_point_nums=max_p, alpha_l=alpha_l)
                with pbar_lock:
                    pbar.update(1)
                    pbar.set_postfix(min_points=min_p, max_points=max_p, mean_mae=ret)
                return i, j, ret,std
            
            # 使用线程池并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # 创建所有任务
                tasks = []
                for i, min_p in enumerate(min_p_range):
                    for j, win_size in enumerate(win_size_range):
                        tasks.append(executor.submit(process_task, min_p, win_size,i,j))
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(tasks):
                    try:
                        i, j, ret,std = future.result()
                        result_matrix[i, j] = ret if ret is not None else np.nan
                        std_matrix[i, j] = std if std is not None else np.nan
                        
                    except Exception as e:
                        print(f"Task failed: {e}")
        
        return result_matrix, std_matrix
    
    def run_alpha_matrix_config(self, min_point_num, max_point_num, alpha_ls, n_jobs=4):
        """
        Run the experiment with a matrix of alpha values.
        """
        total_tasks = len(alpha_ls)
        result_mean = np.zeros((len(alpha_ls),))
        result_mean[:] = np.nan  # 初始化为NaN，表示未计算
        result_std = np.zeros((len(alpha_ls),))
        result_std[:] = np.nan  # 初始化为NaN，表示未计算
        idx = 0
        plt.figure(figsize=(8, 8))

        for alpha_l in tqdm.tqdm(alpha_ls, desc="Running alpha configurations"):
            ret, std = self.run_one_config(min_point_nums=min_point_num, max_point_nums=max_point_num, alpha_l=alpha_l)
            
            result_mean[idx] = ret if ret is not None else np.nan
            result_std[idx] = std if std is not None else np.nan
            idx += 1
            plt.plot(self.gt, self.pred, 'o', label=f'Alpha: {alpha_l:.3f}', color=plt.cm.viridis(idx / len(alpha_ls)))
            plt.plot([0,14],[0,14], 'k--', color='gray')  # 画y=x的参考线
            plt.xlabel('Ground Truth Breaking Force')
            plt.ylabel('Predicted Breaking Force')
            self.gt.clear()
            self.pred.clear()
            plt.xlim(0,14)
            plt.ylim(0,14)
            plt.legend()
            plt.show(block=False)
            # show 1s
            plt.pause(0.5)
            plt.cla()
        return result_mean, result_std



if __name__ == "__main__":
    exp_executer = ExpExecuter()

    alpha_ls = np.arange(0.1, 0.9+0.025, 0.025)
    result_mean, result_std = exp_executer.run_alpha_matrix_config(10,60,alpha_ls)

    # 将结果转换为DataFrame
    df = pd.DataFrame(result_mean, columns=['Mean MAE'])
    df['Std MAE'] = result_std
    # 保存到CSV文件
    df.to_csv('alpha_results.csv', index=False)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.errorbar(alpha_ls, result_mean, yerr=result_std, fmt='-o', capsize=5, label='Mean MAE with Std')
    plt.xlabel('Alpha Value')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE vs Alpha Value')
    plt.grid()
    plt.legend()
    plt.show()
