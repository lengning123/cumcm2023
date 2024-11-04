import numpy as np
import pandas as pd
#导入第一问的模块
import cumcm1 as cum

#计算测线的法线方向上坡度
def alpha(a,b):
    return np.arctan(np.cos(b-np.pi/2)*np.tan(a))

if __name__ == '__main__':
    #问题常数
    a = 1.5/180*np.pi
    print(a)
    b = [0, 45, 90, 135, 180, 225, 270, 315]
    d = 0.3*1852
    theta = 120/180*np.pi
    len_list=[]
    #求解
    for i in range(len(b)):
        #获取坡度
        alpha1 = alpha(a,b[i]/180*np.pi)
        alpha2 = -1*np.arctan(np.cos(b[i]/180*np.pi) * np.tan(a))
        print(alpha2)
        #print(alpha1)
        l=[]
        depth = 0
        for j in range(8):
            #求深度
            depth = 120 - cum.dh(j, d, alpha2)
            l.append(cum.length(depth, theta, alpha1))
        len_list.append(l)
    print(len_list)
    df=pd.DataFrame(len_list)

    #写入文件
    writer_into_2 = pd.ExcelWriter('问题二.xlsx')
    df.to_excel(writer_into_2)
    writer_into_2.close()