import numpy as np
import cumcm2 as cum2
import cumcm1 as cum1
import pandas as pd

def de_percent(p,D_true,theta_true,alpha):
    return D_true*(1-p)/(1/np.tan(theta_true/2)+np.tan(alpha))*2

def De_p(p,d,theta_true,alpha):
    return d/2/(1-p)*(1/np.tan(theta_true/2)+np.tan(alpha))

def depth_all(a0,length0,center_depth0):
    return center_depth0+length0/2*np.tan(a0)

#顺着
def percent1(d_list1,d1,d2,a1,a2,theta_true):
    dx=sum(d_list1)-1*nmi
    beta = np.pi / 2 - a1 - theta_true / 2
    nita = np.pi / 2 + a2 - theta_true / 2
    if dx>0:
        d_a1=d1-np.tan(a1)*(0.5*nmi+dx)
        d_a2=d2+np.tan(a2)*(0.5*nmi-dx)
        x1 = d_a1/np.sin(beta)*np.sin(theta_true/2) - dx/np.cos(a1)
        x2 = d_a2 / np.sin(nita) * np.sin(theta_true / 2) + dx/np.cos(a2)
        p = x2/(x1+x2)
        if x1+x2 < 0:
            p=0
    if dx<=0:
        d_a1 = d1 - np.tan(a1) * (0.5 * nmi + dx)
        d_a2 = d2 + np.tan(a2) * (0.5 * nmi - dx)
        x1 = d_a1 / np.sin(beta) * np.sin(theta_true / 2) - dx / np.cos(a1)
        x2 = d_a2 / np.sin(nita) * np.sin(theta_true / 2) + dx / np.cos(a2)
        p = x2 / (x1 + x2)
    return p

#逆着
def percent2(d_list1,d1,a1,d2,a2,theta_true):
    dx = sum(d_list1) - 1 * nmi
    d_a1 = d1 - np.tan(a1) * (0.5 * nmi + dx)
    d_a2 = d2 - np.tan(a2) * (0.5 * nmi - dx)
    beta = np.pi / 2 + a1 - theta_true / 2
    nita = np.pi / 2 - a2 - theta_true / 2
    x1 = d_a1 / np.sin(beta) * np.sin(theta_true / 2) - dx / np.cos(a1)
    x2 = d_a2 / np.sin(nita) * np.sin(theta_true / 2) + dx / np.cos(a2)
    p = x2 / (x1 + x2)
    return p



#定义海里
nmi = 1852
#定义常量变量
theta = 120/180*np.pi
df_a=pd.read_excel('slope1.xlsx')
df_depth=pd.read_excel('p44.xlsx')
df_angle=pd.read_excel('arf1.xlsx')
num=np.zeros(df_a.shape)
d_all=[]
row,column=df_a.shape
l_all=0
for i in range(row):
    dd_all=[]
    for j in range(column):
        a = df_a.iloc[i,j]/180*np.pi
        angle = df_angle.iloc[i,j]
        if angle>90:
            angle = angle - 90
        angle = angle/180*np.pi
        length = 0.02 * 50 * nmi / np.cos(a)
        width = 0.02 * 50 * nmi
        center_depth=df_depth.iloc[i,j]
        D=depth_all(a,length,center_depth)
        all_width = D*np.tan(theta/2)
        d_list=[all_width/2]
        D=D-d_list[-1]*np.tan(a)
        w=cum1.length(D,theta,a)
        while w <= length:
            d=de_percent(0,D,theta,a)
            d_list.append(d*np.cos(a))
            D=D-d*np.tan(a)
            w += cum1.length(D, theta, a)
        num[i,j]=len(d_list)
        dd_all.append(np.array(d_list))
    d_all.append(dd_all)
print(np.sum(num))
print(len(d_all[4][0]))
print(percent1(d_all[2][1],df_depth.iloc[2,1],df_depth.iloc[2,2],df_a.iloc[2,2],df_a.iloc[2,2],theta))