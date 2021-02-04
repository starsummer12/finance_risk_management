import numpy as np
import pandas as pd
import  math

df=pd.read_table("D:\\data_toyuanrui.txt")
collist=list(df.columns[2:])
# collist=collist[:50]
def woe_result(df,collist,bin=10):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-9999999999)
    pos=df['label'].sum()
    neg=df['label'].count() - pos
    rst=pd.DataFrame()
    for i in collist:
        quantiles = list(set(df[i].quantile(np.arange(0, 1.000001, 1 / bin))))
        quantiles.sort()
        lowerL,UpperL,posL,negL,woeL,ivL=[],[],[],[],[],[]
        
        for j in range(len(quantiles)):
            if j==0:
                if(quantiles[0]==-9999999999):
                    tmp = df[df[i] == -9999999999]
                    lowerBound = -9999999999
                    upperBound = -9999999999
            else:
                if (j == 1) & (quantiles[0] != -9999999999):
                    tmp = df[(df[i] >= quantiles[j-1]) & (df[i] <= quantiles[j])]
                    lowerBound = -9999999999
                    upperBound = quantiles[j]
                else:
                    tmp = df[(df[i] > quantiles[j-1]) & (df[i] <= quantiles[j])]
                    lowerBound = quantiles[j-1]
                    if j == len(quantiles)-1:
                        upperBound = 9999999999
                    else:
                        upperBound = quantiles[j]
            bin_pos=tmp['label'].sum()
            bin_neg=tmp['label'].count() - bin_pos
            if (bin_pos==0) | (bin_neg==0):
                bin_woe=0
                bin_iv=0 
            else:
                null_pos_per = bin_pos/pos
                null_neg_per = bin_neg/neg
                bin_woe=np.log(null_pos_per/null_neg_per)
                bin_iv=bin_woe*(null_pos_per-null_neg_per)
            lowerL.append(lowerBound)
            UpperL.append(upperBound)
            posL.append(bin_pos)
            negL.append(bin_neg)
            woeL.append(bin_woe)
            ivL.append(bin_iv)

        rst_tmp = pd.DataFrame({"feature": i, "lowerBound": lowerL, "upperBound": UpperL, 
                                "positivenum": posL,"negtivenum": negL, 
                                "bin woe": woeL,"bin iv": ivL})
        rst_tmp['feature iv'] = sum(rst_tmp['bin iv'])
        rst = rst.append(rst_tmp)
    return rst

woe=woe_result(df,collist)