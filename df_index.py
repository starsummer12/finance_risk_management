import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

df=pd.read_table("D:\\data_toyuanrui.txt")
df_copy=df.copy()
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

print(df1)
# df.fillna(value=df.mean(),inplace=True)

aucvalue=[]
ksv=[]
def auc(dfcopy):
    for i in range(2,796):
        l1 = dfcopy.iloc[:, [1, i]].dropna(how='any')
        label=l1.iloc[:, 0]
        features=l1.iloc[:,1]
        fpr,tpr,thresholds=roc_curve(label,features)
        value=metrics.auc(fpr, tpr)
        aucvalue.append(value)
    return aucvalue

def ks(dfcopy):
    for i in range(2, 796):
        l1 = dfcopy.iloc[:, [1, i]].dropna(how='any')
        label = l1.iloc[:, 0]
        features = l1.iloc[:, 1]
        fpr, tpr, thresholds =roc_curve(label, features)
        # aucvalue = metrics.auc(fpr, tpr)
        ksvalue = max(tpr - fpr)
        ksv.append(ksvalue)
    return ksv
dfauc=auc(df_copy)
dfks=ks(df_copy)


# df2['auc']=dfauc
# df2['ks']=dfks
print(df2)


def PSi_calc(indsn1,indsn2,numVarlist,binNum=10):
    #将df1和df2的nan进行处理
    #replace():将np.inf的范围值替换成nan
    #fillna:将np.nan数据用-99999填充
    #输出df1,df2用-999999(-1.0e+10)代替的dataframe
    indsn1 = indsn1.replace([np.inf,-np.inf],np.nan).fillna(-9999999999)
    indsn2 = indsn2.replace([np.inf,-np.inf],np.nan).fillna(-9999999999)

    psi_list=[]
    for i in numVarlist:
        #对于每一个feature去进行数值去重并转换为list
        #生成每一个feature的uniqueValue list
        uniqueValue = indsn1[i].unique().tolist()
        #这里list的长度有可能大于或小于binnum
        if len(uniqueValue)>binNum:
            #将每个feature分成10个分位点（分箱）去重后转换为list并进行排序
            #生成1个包含10个元素的list并根据快速排序方法进行排序
            quantailList = list(set(indsn1[i].quantile(np.arange(0,1.000001,1/binNum))))
            quantailList.sort()
            psi_var=0
            for j in range(len(quantailList)):
                if j == 0:
                    if quantailList[0] == -9999999999:#判断最小值是否为空值
                        #将空值求出df1的预期占比和df2的实际占比
                        pct1 = indsn1[indsn1[i] == -9999999999][i].count()/len(indsn1)
                        pct2 = indsn2[indsn2[i] == -9999999999][i].count()/len(indsn2)
                        #pct1或pct2=0或相等表示psi指标很稳定，否则就用PSi计算公式进行计算
                        if (pct1==0) | (pct2==0) | (pct1==pct2):
                                            psi = 0
                        else:
                            psi = (pct1-pct2)*np.log(pct1/pct2)
                        psi_var += psi
                elif j>0:
                    #将range的范围(indsn1[i] >= quantailList[j-1]) & (indsn1[i] <= quantailList[j])进行统计在
                    #此范围的个数算出预期占比和实际占比
                    #这里判断j==1是要去判断范围的取值
                    if (j == 1) & (quantailList[0] != -9999999999):
                        pct1 = indsn1[(indsn1[i] >= quantailList[j-1]) & (indsn1[i] <= quantailList[j])][i].count()/len(indsn1)
                        pct2 = indsn2[(indsn2[i] >= quantailList[j-1]) & (indsn2[i] <= quantailList[j])][i].count()/len(indsn2)
                    else:
                        pct1 = indsn1[(indsn1[i] > quantailList[j-1]) & (indsn1[i] <= quantailList[j])][i].count()/len(indsn1)
                        pct2 = indsn2[(indsn2[i] > quantailList[j-1]) & (indsn2[i] <= quantailList[j])][i].count()/len(indsn2)
                        # pct1或pct2=0或相等表示psi指标很稳定，否则就用PSi计算公式进行计算
                    if (pct1==0) | (pct2==0) | (pct1==pct2):
                        psi = 0
                    else:
                        psi = (pct1-pct2)*np.log(pct1/pct2)
                        psi_var += psi
        else:
            psi_var=0
            for j in uniqueValue:
                #如果uniquevalue的个数小于分箱数就对他们单个计算占比
                pct1 = indsn1[indsn1[i] == j][i].count()/len(indsn1)
                pct2 = indsn2[indsn2[i] == j][i].count()/len(indsn2)
                # pct1或pct2=0或相等表示psi指标很稳定，否则就用PSi计算公式进行计算
                if (pct1==0) | (pct2==0) | (pct1==pct2):
                    psi = 0
                else:
                    psi = (pct1-pct2)*np.log(pct1/pct2)
                    psi_var += psi
        #将每一个psi值添加到list中并生成dataframe
        psi_list.append(psi_var)
    # num_psi = pd.DataFrame({'Variable':numVarlist,'PSi':psi_list})

    return psi_list

df=pd.read_table("D:\\data_toyuanrui.txt")
#对dataframe用日期进行划分,reset_index为将划分的两个dataframe进行索引的重置
dfm3=df[(df['date']>='2019-03-01')&(df['date']<='2019-03-31')].reset_index(drop=True)
dfm4=df[(df['date']>='2019-04-01')&(df['date']<='2019-04-30')].reset_index(drop=True)
#这里采用了一个set集合的操作，将df的列求差集得到每一个df的feature列并转化为list
# varlist = list(set(df1.columns)-set(['date','label']))
varlist = list(df.columns[2:])
# psi_result = PSi_calc(dfm3,dfm4,varlist)
# df2['psi']=psi_result


#woe_iv


# df2.to_csv("D:\\result_index.csv")