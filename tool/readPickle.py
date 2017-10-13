from datetime import date, timedelta
import pandas as pd
import pickle
import os

df = pd.read_pickle("../eikon_sample_data/timeseries_sample1.pickle")
print ("time series:")
print (df.head())

df = pd.read_pickle('../eikon_sample_data/fs_sample.pickle')
print ("finance statement:")
print (df)

df = pd.read_pickle('../eikon_sample_data/news_sample.pickle')
print ("story ID:")
print (df.storyId.iloc[0])
print ("story_head:")
print (df.text.iloc[0])
print ("story content with html tag")
print (df.story_content.iloc[0])


# if __name__ == '__main__':
#     main()