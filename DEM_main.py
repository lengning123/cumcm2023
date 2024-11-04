import DEM as dem
from DEM import Drawgrid
import numpy as np
import pandas as pd


if __name__=='__main__':
    nmi = 1852
    npgrid=pd.read_excel("p44.xlsx").values
    pre=npgrid
    print(npgrid.shape)
    npgrid=dem.AddRound(npgrid)
    dx,dy=dem.Cacdxdy(npgrid,0.02*50*nmi,0.02*50*nmi)
    slope,arf=dem.CacSlopAsp(dx,dy)
    print(arf.shape)
    print(slope.shape)
    df=pd.DataFrame(slope)
    df1=pd.DataFrame(arf)
    writer_into_3 = pd.ExcelWriter('slope1.xlsx')
    df.to_excel(writer_into_3)
    writer_into_3.close()

    writer_into_32 = pd.ExcelWriter('arf1.xlsx')
    df1.to_excel(writer_into_32)
    writer_into_32.close()
