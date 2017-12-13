import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
import csv
import pandas as pd
import numpy as np


with open ('nasdaq100_padding.csv', 'r') as f:
    data = pd.read_csv('./nasdaq100_padding.csv')
    data = np.array(data)
    train_day = 90
    val_day = 7
    test_day = 7
    minutes = 390
    train = data[:train_day * minutes, :]
    val = data[train_day * minutes:(train_day + val_day) * minutes,:]
    test = data[(train_day + val_day) * minutes:(train_day + val_day + test_day) * minutes,:]
    fig = plt.figure()
    plt.plot(test[81])
    plt.ylabel("NASDAQ")
    plt.xlabel("distance")
    plt.show()
    plt.savefig('NASDAQ.png')