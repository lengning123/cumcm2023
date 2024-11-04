import numpy as np
import cumcm2 as cum2
import cumcm1 as cum1
import pandas as pd

df_a=pd.read_excel('slope1.xlsx').values
df_depth=pd.read_excel('p44.xlsx').values
df_angle=pd.read_excel('arf1.xlsx').values
nmi=1852

print(df_angle.shape)

def span(depth,a,angle):
    hang=[]
    for i in range(1,26):
        d=(i-13)*0.02*nmi
        h=depth+d*np.cos(angle)*np.tan(a)
        hang.append(h)
    dot=[]
    for i in hang:
        lie=[]
        for j in range(1,26):
            d=(j-13)*0.02*nmi
            ll=i+d*np.sin(angle)*np.tan(a)
            lie.append(ll)
        dot.append(lie)
    return np.asarray(dot)

for i in range(5):
    for j in range(1):
        zz=span(df_depth[i,j],df_a[i,j],df_angle[i,j])
        df=pd.DataFrame(zz)
        name=str(i)+str(j)
        writer_into_32 = pd.ExcelWriter('{}.xlsx'.format(name))
        df.to_excel(writer_into_32)
        writer_into_32.close()