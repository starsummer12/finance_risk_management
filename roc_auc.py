import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import  metrics
import warnings
warnings.filterwarnings('ignore')
import math

dfa=pd.read_table("D:\\data_toyuanrui.txt")

dfa.fillna(value=dfa.mean(),inplace=True)

# df1=df1.copy()
aucvalue=[]
ks=[]
def auc(df1):
    for i in range(2,796):
        l1=df1.iloc[:,[1,i]]
        # print("l1:",l1)
        label=l1.iloc[:, 0]
        features=l1.iloc[:,1]
        fpr,tpr,thresholds=roc_curve(label,features)
        value=metrics.auc(fpr, tpr)
        aucvalue.append(value)
    return aucvalue

def ks(df1):
    for i in range(2, 796):
        l1 = df1.iloc[:, [1, i]]
        label = l1.iloc[:, 0]
        features = l1.iloc[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(label, features)
        # aucvalue = metrics.auc(fpr, tpr)
        ksvalue = max(tpr - fpr)
        # ks.append(ksvalue)
    return ksvalue
dfauc=auc(dfa)
dfks=ks(dfa)

# df1.loc[len(df)]=dfks
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
            # [0.1657933384, -9999999999.0, 0.5002044383999998, 0.40267773220000014, 0.29876359399999997, 0.6045822358,
            #  0.7294833250000001, 1.0]
            psi_var=0
            for j in range(len(quantailList)):
                if j == 0:
                    if quantailList[0] == -9999999999:#判断最小值是否为空值
                        #将空值求出df1的预期占比和df2的实际占比
                        pct1 = indsn1[indsn1[i] == -9999999999][i].count()/len(indsn1)
                        # print("pct1_:",indsn1[indsn1[i] == -9999999999][i].count())
                        # print("pct1:",len(indsn1))
                        pct2 = indsn2[indsn2[i] == -9999999999][i].count()/len(indsn2)
                        # print("pct2_:",indsn2[indsn2[i] == -9999999999][i].count())
                        # print("pct2:", len(indsn2))
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
    num_psi = pd.DataFrame({'Variable':numVarlist,'PSi':psi_list})

    return num_psi

df=pd.read_table("D:\\data_toyuanrui.txt")
#对dataframe用日期进行划分,reset_index为将划分的两个dataframe进行索引的重置
df1=df[(df['date']>='2019-03-01')&(df['date']<='2019-03-31')].reset_index(drop=True)
df2=df[(df['date']>='2019-04-01')&(df['date']<='2019-04-30')].reset_index(drop=True)
#这里采用了一个set集合的操作，将df的列求差集得到每一个df的feature列并转化为list
varlist = list(set(df1.columns)-set(['date','label']))
psi_result = PSi_calc(df1,df2,varlist)

psi_result['auc']=dfauc
psi_result['ks']=dfks
psi_result_t=psi_result.T
print(psi_result_t)
psi_result_t.to_csv("D:\\psi_ks_auc.csv")