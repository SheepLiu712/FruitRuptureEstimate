This dataset provides 2 kinds of python script, break_force_est_offline.py and train_model.py.

train_model.py uses all the data under the same directory to train a linear relationship between rupture force and stiffness.

break_force_est_offline.py use ONE specified data under the same directory to simulate the online estimation process and output the final estimation of rupture force based on the trained model.

To run these two kinds of script, below required packages should be installed:
- numpy
- matplotlib
- pandas