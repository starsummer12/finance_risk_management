import numpy as np
import pandas as pd
import  math

df=pd.read_table("D:\\data_toyuanrui.txt")
df1=df.T
collist=list(df.columns[2:])
df.to_csv("D:\\df.csv")

def woe_result(df,collist,bin=10):
    minlist=[]
    maxlist=[]
    posnumlist=[]
    negnumlist=[]
    woelist=[]
    ivlist=[]
    featurelist=[]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-99999)
    for i in collist:
        # print(i)
        # i='feature1'
        pos=df[df['label'] == 1][i].count()
        neg=df[df['label'] == 0][i].count()
        # print(pos,neg)
        quantilest=list(set(df[i].quantile(np.arange(0, 1.000001, 1 / bin))))
        quantilest.sort()
        print(quantilest)
        woe=0
        iv=0
        for j in range(len(quantilest)):
            if(j==0):
                if(quantilest[0]==-99999):
                    range1= (df[i] == -99999)
                    range_min = df[range1][i].min()
                    range_max = df[range1][i].max()
                    posnum=df[range1 & (df['label'] ==1)][i].count()
                    negnum=df[range1 & (df['label'] ==0)][i].count()
                    # print("1",negnum)
                    # negnum=neg-posnum
                    null_pos_per = posnum/pos
                    null_neg_per = negnum/neg
                    if (null_pos_per == 0) | (null_neg_per == 0) | (null_neg_per == null_pos_per):
                                woe_value = 0
                    else:
                        woe_value=np.log(null_pos_per/null_neg_per)
                        iv_value=woe_value*(null_pos_per-null_neg_per)
                    woe=woe_value
                    iv+=iv_value

                    featurelist.append(i)
                    minlist.append(range_min)
                    maxlist.append(range_max)
                    posnumlist.append(posnum)
                    negnumlist.append(negnum)
                    woelist.append(woe)
                    ivlist.append(iv)

            else:
                if (j == 1) & (quantilest[0] != -99999):
                    min=(df[i] >= quantilest[j-1])
                    max=(df[i] <= quantilest[j])
                    range_min=df[min][i].min()
                    range_max=df[max][i].max()
                    range1=(df[i] >= quantilest[j-1])&(df[i] <= quantilest[j])
                    posnum = df[range1 & (df['label'] == 1)][i].count()
                    negnum = df[range1 & (df['label'] == 0)][i].count()

                    # print("2", negnum)
                    # negnum = neg - posnum
                    null_pos_per = posnum/pos
                    null_neg_per = negnum/neg

                else:
                    min = (df[i] > quantilest[j - 1])
                    max = (df[i] <= quantilest[j])
                    range_min = df[min][i].min()
                    range_max = df[max][i].max()
                    range1 = (df[i] > quantilest[j-1]) & (df[i] <= quantilest[j])
                    posnum= df[range1 & (df['label'] == 1)][i].count()
                    negnum = df[range1 & (df['label'] == 0)][i].count()

                    # print("3", negnum)
                    # negnum = neg - posnum
                    null_pos_per = posnum / pos
                    null_neg_per = negnum / neg
                if (null_pos_per == 0) | (null_neg_per == 0) | (null_neg_per == null_pos_per):
                    woe_value = 0
                else:
                    woe_value = np.log(null_pos_per / null_neg_per)
                    iv_value = woe_value * (null_pos_per - null_neg_per)
                    woe = woe_value
                    iv += iv_value
                featurelist.append(i)
                minlist.append(range_min)
                maxlist.append(range_max)
                posnumlist.append(posnum)
                negnumlist.append(negnum)
                woelist.append(woe)
                ivlist.append(iv)
        # num_psi = pd.DataFrame({'featurelist': featurelist, 'minlist': minlist,
        #                         'maxlist': maxlist, 'posnumlist': posnumlist,
        #                         'negnumlist': negnumlist, 'woelist': woelist,
        #                         'ivlist': ivlist})
        # print(num_psi)

    #feature:not match
    return featurelist,minlist,maxlist,posnumlist,negnumlist,woelist,ivlist
fealist,minlist,maxlist,postivenum,negtivenum,woe_list,iv_list=woe_result(df,collist)

result=pd.DataFrame()
result['feature']=fealist
result['min']=minlist
result['max']=maxlist
# result['range']=scope
result['posnum']=postivenum
result['negnum']=negtivenum
result['woe']=woe_list
result['iv']=iv_list


result.to_csv("D:\\woe_iv.csv")
print(result)

