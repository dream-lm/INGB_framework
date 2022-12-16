'''
Author: Zhou Hao
Date: 2021-02-22 14:05:42
LastEditors: Zhou Hao
LastEditTime: 2022-10-27 16:18:13
Description: 画图和测试的代码
            main_post : _ENN',
                        '_TomekLinks',
                        '_RSB',
                        '_IPF'
E-mail: 2294776770@qq.com
'''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import imbalanced_databases as imbd
import pandas as pd
from api import (binary_data, fourclass_data)
import  _smote_variants_original as sv_original   # 原始采样框架
import _smote_variants_INGB as sv_gb    # INGB_framework



"""draw subfig, called by main_pre/main_post"""
def draw(X,y,ax:plt.subplot,X_samp,title,num,main):

    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                c='tan', marker='o', s=5, )
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                c='darkcyan', marker='o', s=5, )
    X_new = pd.DataFrame(X_samp).iloc[len(X):, :]
    ax.scatter(X_new[0], X_new[1], c='red', s=5, marker='+')
    a = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    b = ['(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)']

    if main == 'pre':
        title = a[num%10-1]+' '+title
    elif main == 'post':
        title = b[num%10-1]+' '+title
    plt.title(title)
    plt.grid()      #网格线


"""smote_variants"""
def main_post(data=1,is_save=False,is_show=False):

    all = [
        'SMOTE_ENN',
        'SMOTE_TomekLinks',
        'SMOTE_RSB',
        'SMOTE_IPF',
    ]
    if data == 2:X, y = binary_data(data_name='make_moons')
    elif data == 1:X,y = fourclass_data()
    elif data ==3:X,y = binary_data(data_name='make_circles')

    oversamplers_1 = sv_original.get_all_oversamplers(all=all)     
    oversamplers_5 = sv_gb.get_all_oversamplers(all=all)     
    num=240 # index of subfig
    plt.figure(figsize=(20, 10))

    for o in zip(oversamplers_1,oversamplers_5):
        num +=1
        oversampler_1 = o[0]()      # 原始采样
        oversampler_5 = o[1]()      # 加权采样
        X_samp_original, y_samp_1= oversampler_1.sample(X, y,)
        X_samp_gbsmote, y_samp_5 = oversampler_5.sample(X, y)
    
        ax= plt.subplot(num)
        draw(main= 'post',num=num,X=X,y=y,X_samp=X_samp_original,ax=ax,title=oversampler_1.__class__.__name__)

        num+=1
        ax = plt.subplot(num)
        draw(main='post',num=num,X=X, y=y, X_samp=X_samp_gbsmote, ax=ax,title='GB-'+str(oversampler_5.__class__.__name__))

    if is_save:
        plt.savefig(fname='./pdf/'+'enn_tome_rsb_ipf_smote'+'.pdf',format='pdf',bbox_inches='tight')
    if is_show:
        plt.show()


if __name__ == "__main__":
    """
        data :  1:fourclass_data
                2:make_moons
                3:make_circles
    """
    main_post(data=2,is_save=False,is_show=True)