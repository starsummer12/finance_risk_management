import pandas as pd
import numpy as np

result=0
sum=0
for i in range(1,5):
    if(i%2==0):
        sum+=i
        sum += 1
    else:
        sum+=2*i
        sum += 1
    # sum+=1

print(sum)