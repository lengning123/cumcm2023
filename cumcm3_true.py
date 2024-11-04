import numpy as np
import cumcm2 as cum2
import cumcm1 as cum1
from tqdm import tqdm#进度条设置
import matplotlib.pylab as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#求间距
def de_percent(p,D_true,theta_true,alpha):
    return D_true*(1-p)/(1/np.tan(theta_true/2)+np.tan(alpha))*2

#求对应间距和覆盖率的深度
def De_p(p,d,theta_true,alpha):
    return d/2/(1-p)*(1/np.tan(theta_true/2)+np.tan(alpha))

#求深度
def depth_all(a,length,center_depth):
    return center_depth+length/2*np.tan(a)

#单个区块下的长度和
def square_length(l,depth,theta_true,alpha,angle_true,width_true):
    #开始位置
    star=width_true/2/np.cos(angle_true)
    #结束位置
    finish_l=star-l/np.tan(angle_true)
    #剩余长度
    length_all=width-star-finish_l
    #测线长度
    s_l=l/np.sin(angle_true)
    #间距
    d0=de_percent(0.1,depth,theta_true,alpha)
    #全长
    all_length=int(length_all/d0+2)*s_l
    #测线出现在区域外侧
    if star+finish_l>width:
        star=l*np.tan(angle_true)
        s_l=l/np.cos(angle_true)
        all_length=int((width-star)/d0+2)*s_l
        if star>width:
            all_length=width+(np.pi-angle_true)*100
    return all_length

#定义海里
nmi = 1852
#定义常量变量
theta = 120/180*np.pi
length = 4*nmi
width = 2*nmi
a = 1.5/180*np.pi
center_depth=110
#测线夹角
angle = 170/180*np.pi
#测线切线夹角
band_angle = angle - np.pi/2
alpha0 = cum2.alpha(a, band_angle)
#计算区间长度
d_list=[]
D_list=[center_depth + length / 2 * np.tan(a)]
d=0
D=depth_all(a,length,center_depth)
#计算所有区间长度和区间起始深度
while sum(d_list)< length:
    d=D/9/np.tan(a)
    d_list.append(d)
    D = D *8/9
    D_list.append(D)
d_list[-1]=length-sum(d_list[:-1])
#总深度和总宽度
all_d = depth_all(a,length,center_depth)
all_width = all_d*np.tan(theta/2)
#print(D_list)
#print(d_list)
#print(all_width/2)

'''mm=0
for i in range(24):
    mm+=square_length(d_list[13],D_list[13],theta,alpha0,band_angle,all_width)
print(mm)'''


#=======================定义参数==========================
m=10                   #蚂蚁个数
rounds=2500              #最大迭代次数
Rho=0.4                #信息素蒸发系数
p0=0.2                 #转移概率常数
AMAX= 180            #搜索变量x最大值
AMIN= 91             #搜索变量x最小值
step=0.1             #局部搜索步长
n=24                  #变量个数
P=np.zeros(shape=(rounds,m)) #状态转移矩阵
fitneess_value_list=[] #迭代记录最优目标函数值

#适应度函数
def calc_fitness(X):
    f=0
    for i in range(len(X)):
        band_angle = X[i]/180*np.pi- np.pi / 2
        f+=square_length(d_list[i],D_list[i],theta, alpha0, band_angle, all_width)
    return f

def calc_e(X):
    return 0

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
            X[i,j] = np.random.uniform(170, AMAX, 1)[0]
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


    plt.plot(fitneess_value_list, label='迭代曲线')
    plt.legend()
    plt.show()



if __name__=='__main__':
    main()
