from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
#####在原栅格图像周围加一圈并返回
def AddRound(npgrid):
    ny, nx = npgrid.shape  # ny:行数，nx:列数
    square=np.zeros((ny+2,nx+2))
    square[1:-1,1:-1]=npgrid
    #四边
    square[0,1:-1]=npgrid[0,:]
    square[-1,1:-1]=npgrid[-1,:]
    square[1:-1,0]=npgrid[:,0]
    square[1:-1,-1]=npgrid[:,-1]
    #角点
    square[0,0]=npgrid[0,0]
    square[0,-1]=npgrid[0,-1]
    square[-1,0]=npgrid[-1,0]
    square[-1,-1]=npgrid[-1,0]
    return square
#####计算xy方向的梯度
def Cacdxdy(npgrid,sizex,sizey):
    square=AddRound(npgrid)
    dx=((square[1:-1,:-2])-(square[1:-1,2:]))/sizex/2
    dy=((square[2:,1:-1])-(square[:-2,1:-1]))/sizey/2
    dx=dx[1:-1,1:-1]
    dy=dy[1:-1,1:-1]
    return dx,dy
####计算坡度\坡向
def CacSlopAsp(dx,dy):
    slope=(np.arctan(np.sqrt(dx*dx+dy*dy)))*57.29578  #转换成°
    #坡向
    a=np.zeros([dx.shape[0],dx.shape[1]]).astype(np.float32)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            x=float(dx[i,j])
            y=float(dy[i,j])
            if (x==0.)& (y==0.):
                a[i,j]=-1
            elif x==0.:
                if y>0.:
                    a[i,j]=0.
                else:
                    a[i,j]=180.
            elif y==0.:
                if x>0:
                    a[i,j]=90.
                else:
                    a[i,j]=270.
            else:
                a[i,j]=float(math.atan(y/x))*57.29578
                if a[i,j]<0.:
                    a[i,j]=90.-a[i,j]
                elif a[i,j]>90.:
                    a[i,j]=450.-a[i,j]
                else:
                    a[i,j]=90.-a[i,j]
    return slope,a
####绘制平面栅格图
def Drawgrid(judge,pre=[],A=[],strs=""):
    if judge==0:
        if strs == "":
            plt.imshow(A, interpolation='nearest', cmap=plt.cm.hot, origin='lower')  # cmap='bone'  cmap=plt.cm.hot
            plt.colorbar(shrink=0.8)
            plt.xticks(())
            plt.yticks(())
            plt.show()
        else:
            plt.imshow(A, interpolation='nearest', cmap=strs, origin='lower')  # cmap='bone'  cmap=plt.cm.hot
            plt.colorbar(shrink=0.8)
            xt=np.arange(0, 4.02, 0.02)
            yt=np.arange(0, 5.02, 0.02)
            plt.xticks(())
            plt.yticks(())
            plt.show()
    elif judge==1:
        fig = plt.figure()
        ax = Axes3D(fig)
        # X = np.arange(1,482,1)
        # Y = np.arange(1,322,1)
        X = np.arange(0, 4.02, 0.02)
        Y = np.arange(0, 5.02, 0.02)
        X, Y = np.meshgrid(X, Y)
        Z = pre
        ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('rainbow'))  # cmap=plt.get_cmap('rainbow')
        ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
        plt.show()
