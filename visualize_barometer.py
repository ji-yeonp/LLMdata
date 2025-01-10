import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


EXP_PATH = 'experiments'

if __name__ == "__main__":
    exp_list = os.listdir(EXP_PATH)
    for exp in exp_list:
        with open(os.path.join(EXP_PATH, exp, 'barometer.pkl'), mode='rb') as f:
            train_barometer, valid_barometer = pickle.load(f)
            f.close()
        print(train_barometer.keys())
        print(valid_barometer.keys())