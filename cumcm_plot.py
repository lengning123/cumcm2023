import matplotlib.pylab as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x=[]
y=[]
nmi=1582
for i in range(10):
    y.append(204081.159846225/nmi)
for i in range(90):
    y.append(144811.128917223235/nmi)
for i in range(2400):
    y.append(125936/nmi)
for j in range(1,2501):
    x.append(j)
plt.plot(x,y,label='迭代曲线')
plt.legend()
plt.xlabel('训练次数')
plt.ylabel('最短测迹距离')
plt.savefig('1.png')
plt.show()