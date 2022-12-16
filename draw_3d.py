'''
Author: Zhou Hao
Date: 2022-10-27 20:31:21
LastEditors: Zhou Hao
LastEditTime: 2022-12-16 16:27:38
Description: file content
E-mail: 2294776770@qq.com
'''
from collections import Counter
from matplotlib import projections
from sklearn.datasets import make_swiss_roll,make_blobs,make_gaussian_quantiles
import numpy as np,pandas as pd 
import matplotlib.pyplot as plt
import INGB
import time,random


def motivation(is_save:bool=False,is_show:bool=False,noise_rate:float=0)->None:
    # get dataset ***********************************************************
    X, y = make_blobs(n_samples=2000, centers=3, n_features=3,
                    random_state=50,
                    shuffle=True,
                    cluster_std=0.25,
                    center_box=[-1,1],)
    print('原始比例：\t',Counter(y))

    # make dataset imblance
    index_0 = np.random.choice(np.where(y==0)[0],150)  
    index_1 = np.random.choice(np.where(y==1)[0],300)
    index_2 = np.random.choice(np.where(y==2)[0],500)
    index = np.hstack((index_0,index_1,index_2))
    X,y = X[index],y[index]
    print('不平衡比例：\t',Counter(y))

    # make dataset noise
    noise_index = random.sample(list(range(len(y))),int(len(y)*noise_rate))
    for i in noise_index:
        if y[i] == 0: y[i] = random.choice([1,2])
        elif y[i] == 1: y[i] = random.choice([0,2])
        elif y[i] == 2: y[i] = random.choice([0,1])
    print('加噪比例：\t',Counter(y))


    # settings of figure ***********************************************************
    plt.figure(figsize=(18,6),dpi=800)
    color = {0: 'darkcyan', 1: 'tan', 2: 'green',-1:'blue'}
    X_colors = [color[label] for label in y]
    font = {'family':'Times New Roman',
            'size':18,}
    X_max, y_max = max(X[:,0]),max(X[:,1])
    X_min, y_min = min(X[:,0]),min(X[:,1])
    Z_min, Z_max = min(X[:,2]),max(X[:,2])
    X_min, X_max = X_min-0.01, X_max+0.01
    y_min, y_max = y_min-0.01, y_max+0.01
    Z_min, Z_max = Z_min-0.01, Z_max+0.01


    # origin data ***********************************************************
    ax = plt.subplot(131,projection='3d')
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_xlim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_ylim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_zlim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_title('(a) Original dataset',font)
    ax.view_init(elev=10,azim=23)
    ax.scatter(X[:,0], X[:,1],X[:,2],c=X_colors,s=15,alpha=0.8)
    

    # Dataset and GBs ***********************************************************
    ax = plt.subplot(132,projection='3d')
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_xlim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_ylim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_zlim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_title('(b) Dataset and GBs',font)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=X_colors,s=15,alpha=0.8)
    ax.view_init(elev=10,azim=23)
    
    cmap = {0:'winter',1:'autumn',2:'summer'}
    nums, dims = X.shape
    data = np.hstack((X,y.reshape(nums,1)))
    balls = INGB.GBList(data)
    balls.init_granular_balls(purity=1,min_sample=dims+1)
    for ball in balls.granular_balls:
        label = ball.label
        # if label == 2:continue  # 跳过多数类
        center, radius = ball.center, ball.radius
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        a = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        b = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        c = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(a,b,c,rstride=1, cstride=1, cmap=cmap[label],alpha=0.4)
        

    # only GBs ***********************************************************
    ax = plt.subplot(133,projection='3d')
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_xlim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_ylim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_zlim(max(X_min,y_min,Z_min), max(X_max,y_max,Z_max))
    ax.set_title('(c) GBs',font)
    ax.view_init(elev=10,azim=23)
    
    cmap = {0:'winter',1:'autumn',2:'summer'}
    for ball in balls.granular_balls:
        label = ball.label
        # if label == 2:continue  # 跳过多数类
        center, radius = ball.center, ball.radius
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        a = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        b = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        c = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(a,b,c,rstride=1, cstride=1, cmap=cmap[label],alpha=0.4)

    # save and show ***********************************************************
    plt.tight_layout()
    plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        plt.savefig(fname=r"pdf/3d/"+now+'.pdf',format='pdf',bbox_inches='tight')
    if is_show:plt.show()


if __name__ == '__main__':
    noise_rate = 0.1
    motivation(is_save=1,is_show=0,noise_rate=noise_rate)

