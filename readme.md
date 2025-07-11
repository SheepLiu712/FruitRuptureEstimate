# README
This repository is data and scripts of paper "A Grip Force Planning Method for Reliable and Rapid Fruit Grasping Based on Rupture Force Estimation". This repository consists of two parts:
- folders `banana`,`kiwi`,`mango`: data and scripts used for training and validating the rupture force prediction model.
- folder `parameter_test`: data and scripts used for analyzing the sensitivity of key hyperparameters involved in the online estimator

## PART1: training and validating

Folders in this part provide 2 kinds of python script, break_force_est_offline.py and train_model.py.

train_model.py uses all the data under the same directory to train a linear relationship between rupture force and stiffness.

break_force_est_offline.py use ONE specified data under the same directory to simulate the online estimation process and output the final estimation of rupture force based on the trained model.

To run these two kinds of script, below required packages should be installed:
- numpy
- matplotlib
- pandas

## PART2: sensitivity analysis
It should be noticed that the data used in this part is the same as the validation data in PART1, so the rupture force prediction model is not trained again.

In folder `parameter_test`, there are two kinds of python script, `<parameter>_test.py` and `plot_<parameter>_result.py`. The former is used to test the sensitivity of the specified parameter, and the latter is used to plot the result of the sensitivity analysis. There are also some subfolders in this part, which are used to store the data for testing and are the same as the validation data in PART1.

To run these scripts, the same packages as PART1 are required, and additionally:
- tqdm