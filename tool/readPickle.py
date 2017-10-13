from datetime import date, timedelta
import pandas as pd
import pickle
import os

df = pd.read_pickle("../eikon_sample_data/timeseries_sample1.pickle")
df.head()

# if __name__ == '__main__':
#     main()