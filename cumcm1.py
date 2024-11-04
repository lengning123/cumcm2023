import numpy as np
import pandas as pd
#转化为弧度
theta = 120/180*np.pi
alpha = 1.5/180*np.pi
d = 200 #单位（米）
D = 70
#求覆盖率
def percent(D,d=d,theta=theta,alpha=alpha):
    eta = (D - d/2/np.tan(theta/2) - d/2*np.tan(alpha))/D
    return eta

#求覆盖长度
def length(D,theta=theta,alpha=alpha):
    beta = np.pi/2 - theta/2 - alpha
    gama = np.pi - theta - beta
    x1 = D/np.sin(beta)*np.sin(theta/2)
    x2 = D/np.sin(gama)*np.sin(theta/2)
    return x1+x2
#求海水变化深度
def dh(n,d=d,alpha=alpha):
    return n*d*np.tan(alpha)


if __name__ == '__main__':
    #记录海水深度、覆盖率和覆盖长度
    dep_list=[]
    per_list = []
    len_list = []

    for i in range(9):
        n = i-4
        depth = D - dh(n)
        per_list.append(percent(depth))
        len_list.append(length(depth))
        dep_list.append(depth)
    #写入文件
    df=pd.DataFrame([dep_list,len_list,per_list])
    writer_into_1 = pd.ExcelWriter('问题一.xlsx')
    df.to_excel(writer_into_1)
    writer_into_1.close()
    print(dep_list)
    print( )
    print(per_list)
    print( )
    print(len_list)