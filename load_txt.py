import numpy as np
import pandas as pd

df=pd.read_table("D:\\data_toyuanrui.txt")
del df['label']
del df['date']
df1=df.describe(percentiles=[.01,.05,.25,.75,.95,.99])
def zerocount(df):
    zero = (df == 0).sum(axis=0)
    zcount = zero / (df.count() + df.isnull().sum())
    return zcount
def nancount(df):
    nancount=df.isna().sum() / (df.isna().sum() + df.count())
    return nancount
df1.loc['0_percentage']=df.apply(zerocount)
df1.loc['nan_percentage']=df.apply(nancount)
df2=df1.T
# df2.to_csv("D:\\result.csv")
print("3:",df2)
df1=df.copy()
df1.fillna(value=df.mean(),inplace=True)
print(df1)