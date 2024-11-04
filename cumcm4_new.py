import numpy as np
import cumcm2 as cum2
import cumcm1 as cum1
import pandas as pd
from tqdm import tqdm#进度条设置
import matplotlib.pylab as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def de_percent(p,D_true,theta_true,alpha):
    return D_true*(1-p)/(1/np.tan(theta_true/2)+np.tan(alpha))*2

def De_p(p,d,theta_true,alpha):
    return d/2/(1-p)*(1/np.tan(theta_true/2)+np.tan(alpha))

def depth_all(a0,length0,center_depth0):
    return center_depth0+length0/2*np.tan(a0)

#定义海里
nmi = 1852
#定义常量变量
theta = 120/180*np.pi
df_a=pd.read_excel('slope1.xlsx')
df_depth=pd.read_excel('p44.xlsx')
df_angle=pd.read_excel('arf1.xlsx')
num=np.zeros(df_a.shape)
d_all=[]
w_all=[]
row,column=df_a.shape
l_all=0
for i in range(row):
    dd_all=[]
    ww_all=[]
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
        w_list=[]
        while w <= length:
            d=de_percent(0,D,theta,a)
            D=D-d*np.tan(a)
            w += cum1.length(D, theta, a)
            d_list.append(d * np.cos(a))
            w_list.append(w)
        num[i,j]=len(d_list)
        dd_all.append(np.array(d_list))
        w_all.append(w_list)
    d_all.append(dd_all)

print(num)

def percent(w_all0):
    s=w_all0/nmi - 1
    return s

m=10                   #蚂蚁个数
rounds=2500              #最大迭代次数
Rho=0.4                #信息素蒸发系数
p0=0.2                 #转移概率常数
AMAX= 20            #搜索变量x最大值
AMIN= 0             #搜索变量x最小值
step=1             #局部搜索步长
n=20                  #变量个数
P=np.zeros(shape=(rounds,m)) #状态转移矩阵
fitneess_value_list=[] #迭代记录最优目标函数值

#适应度函数
def calc_fitness(X):
    f=226-len(X)
    return f

def calc_e(X):
    p=0
    all_n=0
    for i in range(len(X)):
        n = len(w_all[i])-int(X[i])
        if n < 0 :
            n=1
        p+=percent(w_all[i][n-1])
        all_n+=n
    return np.abs(p*100)/all_n

def dcalc_e(X):
    all_n=0
    loss=0
    over=0
    for i in range(len(X)):
        n = len(w_all[i]) - int(X[i])
        if n < 0:
            n = 1
        p = percent(w_all[i][n - 1])
        if p >0.2:
            over+=p
        else:
            loss+=p
        all_n += n
    return [over/all_n*100,loss/all_n*100]


#子代和父辈之间的选择操作
def update_best(pa,pa_fitness,pa_e,kid,kid_fitness,kid_e):

    # 规则1，如果 pa 和 kid 都没有违反约束，则取适应度小的
    if pa_e <= 0.0001 and kid_e <= 0.0001:
        if pa_fitness <= kid_fitness:
            return pa,pa_fitness,pa_e
        else:
            return kid,kid_fitness,kid_e
    # 规则2，如果kid违反约束而pa没有违反约束，则取pa
    if pa_e < 0.0001 and kid_e  >= 0.0001:
        return pa,pa_fitness,pa_e
    # 规则3，如果pa违反约束而kid没有违反约束，则取kid
    if pa_e >= 0.0001 and kid_e < 0.0001:
        return kid,kid_fitness,kid_e
    # 规则4，如果两个都违反约束，则取适应度值小的
    if pa_fitness <= kid_fitness:
        return pa,pa_fitness,pa_e
    else:
        return kid,kid_fitness,kid_e

def initialization():
    X = np.zeros(shape=(m, n))  # 蚁群 shape=(20, 2)
    Tau = np.zeros(shape=(m,))  # 信息素
    for i in range(m):  # 遍历每一个蚂蚁
        for j in range(n):
            X[i,j] = np.random.uniform(0, AMAX, 1)[0]
        Tau[i] = calc_fitness(X[i])  # 计算信息素
    return X,Tau

def position_update(NC,P,X):

    lamda = 1 / (NC + 1)
    # 位置更新
    for i in range(m):  # 遍历每一个蚂蚁
        # 局部搜索
        new_x=[]
        for j in range(n):
            if P[NC, i] < p0:
                new_x.append(X[i, 0] + (2 * np.random.randint(100) - 100)/100 * step * lamda)
        # 全局搜索
            else:
                new_x.append(X[i, 0] + (AMAX - AMIN) * (np.random.random() - 0.5))

        # 边界处理
            if (new_x[-1] < AMIN) or (new_x[-1] > AMAX):
                new_x[-1] = np.random.uniform(AMIN, AMAX, 1)[0]  # 初始化x

    # 子代个体蚂蚁
    kids=np.array(new_x)
    # 子代惩罚项
    kids_e=calc_e(kids)
    # 子代目标函数值
    kids_fit = calc_fitness(kids)
    # 父辈个体蚂蚁
    pa=X[i]
    # 父辈惩罚项
    pa_e=calc_e(pa)
    # 父辈目标函数值
    pa_fit = calc_fitness(pa)
    pbesti, pbest_fitness, pbest_e = update_best(pa, pa_fit, pa_e, kids, kids_fit,kids_e)
    X[i]=pbesti
    return X

def Update_information(Tau,X):

    for i in range(m):  # 遍历每一个蚂蚁
        Tau[i] = (1 - Rho) * Tau[i] + calc_fitness(X[i])  # (1 - Rho) * Tau[i] 信息蒸发后保留的
    return Tau

def main():
    X,Tau=initialization() #初始化蚂蚁群X 和信息素 Tau
    for NC in tqdm(range(rounds)):  # 遍历每一代
        BestIndex = np.argmin(Tau)  # 最优索引
        Tau_best = Tau[BestIndex]   # 最优信息素
        # 计算状态转移概率
        for i in range(m):  # 遍历每一个蚂蚁
            P[NC, i] = np.abs((Tau_best - Tau[i])) / np.abs(Tau_best) + 0.01  # 即离最优信息素的距离
        # 位置更新
        X=position_update(NC,P,X) #X.shape=(20, 2)

        # 更新信息素
        Tau=Update_information(Tau, X)

        # 记录最优目标函数值
        index = np.argmin(Tau)  # 最小值索引
        value = Tau[index]  # 最小值
        fitneess_value_list.append(calc_fitness(X[index]))  # 记录最优目标函数值

    # 打印结果
    min_index = np.argmin(Tau)  # 最优值索引
    minValue = calc_fitness(X[min_index])  # 最优目标函数值

    print('最优变量', [float('{:.4f}'.format(i)) for i in X[min_index,:]])
    print('最优目标函数值', minValue)
    print('最优变量对应的惩罚项', calc_e(X[min_index]))
    print(dcalc_e(X[min_index]))


    plt.plot(fitneess_value_list, label='迭代曲线')
    plt.legend()
    plt.show()



if __name__=='__main__':
    main()
