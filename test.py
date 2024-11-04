import numpy as np
import pandas as pd

#合并小方块
df=pd.read_excel("p4.xlsx").values
print(np.mean(df))
df=df[1:,1:]
df1=np.zeros([5,200])
df2=np.zeros(([5,4]))
for i in range(0,250,50):
    df1[int(i/50),:]=np.mean(df[i:i+49,:],axis=0)

for j in range(0,200,50):
    df2[:,int(j/50)]=np.mean(df1[:,j:j+49],axis=1)

nn=pd.DataFrame(df2[::-1,:])
writer_into_32 = pd.ExcelWriter('p44.xlsx')
nn.to_excel(writer_into_32)
writer_into_32.close()