'''
Author: Zhou Hao
Date: 2021-10-23 21:40:01
LastEditors: Zhou Hao
LastEditTime: 2022-12-16 16:55:05
Description: 
        # source code of INGB_framework
E-mail: 2294776770@qq.com
'''

from sklearn.preprocessing import MinMaxScaler, scale
from scipy.special import gamma     # used to compute V_Ball
from math import log
import math
from sklearn.metrics import pairwise_distances
import random
from sklearn.cluster import k_means
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


class GBList:
    def __init__(self, data) -> None:
        self.data: np.ndarray = data
        self.granular_balls: GranularBall = [GranularBall(self.data)]

    # Set the purity threshold to 1.0
    def init_granular_balls(self, purity=1.0, min_sample=1, balls_list=None) -> None:
        """
            Function : calculate the particle partition under the current purity threshold
            Input: purity threshold, the minimum number of points in the process of pellet division
        """
        if balls_list == None:
            balls_list = self.granular_balls

        length = len(balls_list)
        i = 0
        while True:
            # 粒球划分条件
            if balls_list[i].purity < purity and balls_list[i].num > min_sample:
                # split the current granular_ball
                split_clusters = balls_list[i].split_clustering()
                if len(split_clusters) > 1:     # current ball can be splited
                    balls_list[i] = split_clusters[0]
                    balls_list.extend(split_clusters[1:])
                    length += len(split_clusters) - 1
                elif len(split_clusters) == 1:  # current ball can not be splited
                    i += 1  # split the next granular_ball
                else:
                    balls_list.pop(i)  # 异常
                    length -= 1
            else:
                i += 1  # split the next granular_ball
            if i >= length:
                break
        # print('划分结束后粒球个数:\t',length)

    def remove_overlap(self):
        '''去除重复的粒球'''
        print('进行粒球去重')
        length = len(self.granular_balls)
        pre, cur = 0, 1  # index of balls

        while True:
            pre_ball = self.granular_balls[pre]
            cur_ball = self.granular_balls[cur]
            # print("*****pre, cur*****:\t",pre,cur)

            # 如果两个球标签不同，且有重叠(圆心距离<r1+r2)。
            if (pre_ball.label != cur_ball.label) \
                    and np.sum((pre_ball.center - cur_ball.center) ** 2, axis=0) ** 0.5 \
                    < (pre_ball.radius + cur_ball.radius):
                print("发现重叠:\t", pre, cur)

                # 选择半径大的圆重新划分
                if pre_ball.radius >= cur_ball.radius:
                    split_clusters = pre_ball.split_clustering()
                    len_split = len(split_clusters)
                    if len_split == 1:  # 无法划分
                        pre, cur = pre + 1, cur + 1
                    elif len_split > 1:
                        self.granular_balls[pre] = split_clusters[0]
                        self.granular_balls.extend(split_clusters[1:])
                        length += len_split - 1
                else:
                    split_clusters = cur_ball.split_clustering()
                    len_split = len(split_clusters)
                    if len_split == 1:  # 无法划分
                        pre, cur = pre+1, cur+1
                    elif len_split > 1:
                        self.granular_balls[cur] = split_clusters[0]
                        self.granular_balls.extend(split_clusters[1:])
                        length += len_split - 1

            else:   # not need to split
                pre, cur = pre + 1, cur + 1
            if cur >= length:
                break
        print('去重完毕')


class GranularBall:

    def __init__(self, data):  # data为带标签数据，最后一列为标签.
        self.data: np.ndarray = data
        self.data_no_label: np.ndarray = data[:, :-1]
        self.num, self.dim = self.data_no_label.shape  # rows, columns
        self.center: np.ndarray = self.data_no_label.mean(0)  # 球心:每列属性的均值
        self.label_num: int = len(set(data[:, -1]))  # num of labels
        self.label, self.purity, self.radius = self.info_of_ball()

    def info_of_ball(self):
        count = Counter(self.data[:, -1])  # count the labels
        label = max(count, key=count.get)  # label of majornity
        purity = count[label] / self.num  # purity  = majornity / num
        radius = np.sum(np.sum((self.data_no_label - self.center)**2, axis=1)
                        ** 0.5, axis=0) / self.num   # r = avg(all samples to the center of the ball)
        return label, purity, radius

    def print_info(self) -> None:
        print('\n\n\t **************the infomation of the current ball**************')
        for k, v in self.__dict__.items():
            print(k, ':\t', v)
        print('\t **************the infomation of the current ball**************\n\n')

    def split_clustering(self):
        """
            Function : continue to divide the GranularBall into several new GranularBalls
            Output: new ball list
        """
        # each class has one random center, uesd to split clusters.
        center_array: np.ndarray = np.empty(
            shape=[0, len(self.data_no_label[0, :])])
        for i in set(self.data[:, -1]):     # in each label
            # All samples of the current label
            data_set = self.data_no_label[np.where(self.data[:, -1] == i)]
            # A random point in the dataset
            random_data = data_set[random.randrange(len(data_set)), :]
            center_array = np.append(center_array, [random_data], axis=0)

        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label,
                               init=center_array, n_clusters=self.label_num,)
        data_label = ClusterLists[1]  # Get a list of labels

        for i in set(data_label):
            Cluster_data = self.data[np.where(data_label == i)]
            if len(Cluster_data) > 1:
                Cluster = GranularBall(Cluster_data)
                Clusterings.append(Cluster)
        return Clusterings


class GB_OverSampling:
    """粒球过采样"""

    def __init__(self, purity=0.5) -> None:
        self.sampling_strategy_: dict = {}
        self.purity: float = purity

    def _check_sampling_strategy_(self, y) -> dict:
        '返回一个排序字典 key:value = 要采样的类:数量'
        count = Counter(y)
        most_label = max(count, key=count.get)  # label of the most class
        most_nums: int = count[most_label]  # nums of the most class
        ordered_dict: dict = {}
        for k, v in count.items():
            if k != most_label:
                ordered_dict[k] = most_nums - v
        return ordered_dict

    def _find_ball_sparsity(self, X: np.ndarray):
        """Compute the sparsity of balls."""
        euclidean_distances = pairwise_distances(X, metric="euclidean",)

        # negate diagonal elements
        for ind in range(X.shape[0]):
            euclidean_distances[ind, ind] = 0

        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = euclidean_distances.sum() / non_diag_elements
        exponent = (math.log(X.shape[0], 1.6) ** 1.8 * 0.16)
        return (mean_distance ** exponent) / X.shape[0]

    # 计算少数类球的熵
    def _find_ball_entropy(self, y: np.array):
        """Compute the entropy"""
        num = len(y)
        label_counts = Counter(y)
        shannon_ent = 0
        for k, v in label_counts.items():
            prob = float(v)/num
            shannon_ent -= prob * log(prob, 2)
        return shannon_ent


    # 只计算少数类的球稀疏程度
    def _find_ball_density(self, radius, dims, min_num):
        V = np.pi ** (dims/2.0) / gamma(dims/2.0 + 1) * radius**dims
        if min_num/V == np.inf:   # 当体积特别小的时，
            return 0
        return min_num / V


    # 计算少数类球样本Instance-Wise Statistic(相对密度)
    def _find_ball_RD(self, ball):
        RD = np.zeros(ball.data.shape[0])
        # print(type(RD), RD)
        # 分别获取少数类和多数类的下标
        min = np.where(ball.data[:, -1] == ball.label)[0]
        maj = np.where(ball.data[:, -1] != ball.label)[0]
        data_min = ball.data_no_label[min]
        data_maj = ball.data_no_label[maj]

        # 相对密度与绝对密度结合起来计算真的合适吗？全为同质点的少数类与有异质点的少数类能有区分吗？
        # 计算少数类的相对密度
        for i in min:
            if len(maj) == 0:   # 无异质点则计算绝对密度
                # radius = np.sum(np.sum((self.data_no_label - self.center)**2, axis=1)** 0.5, axis=0) / self.num
                RD[i] = np.sum(np.sum((data_min - ball.data_no_label[i])
                                      ** 2, axis=1) ** 0.5, axis=0) / (len(data_min)-1)
            elif len(min) == 1 and len(maj) != 0:   # 球中只有一个少数点，则可以直接过滤掉这个球
                continue
            else:               # 有异质点则计算绝对密度
                homo = np.sum(np.sum(
                    (data_min - ball.data_no_label[i])**2, axis=1) ** 0.5, axis=0) / (len(data_min)-1)
                hete = np.sum(np.sum(
                    (data_maj - ball.data_no_label[i])**2, axis=1) ** 0.5, axis=0) / (len(data_maj))
                RD[i] = homo / hete
        return RD


    # 计算少数类球的Ball-Wise Statistic(加权信息熵)
    def _find_ball_WIE(self, RD):
        if RD.sum() == 0:
            return 0
        RD_ratio = RD/RD.sum()
        WIE = 0
        for i in RD_ratio:
            if i != 0:
                WIE -= i * log(i, 2)
        return WIE/len(RD)


    def _fit_resample(self, X, y):
        X_resampled = X.copy()
        y_resampled = y.copy()
        self.sampling_strategy_ = self._check_sampling_strategy_(y_resampled)

        # 用初始数据获取粒球
        nums, dims = X.shape
        data = np.hstack((X, y.reshape(nums, 1)))     # 不会改变X，y原来的shape
        balls = GBList(data)        # 初始化粒球列表
        balls.init_granular_balls(
            purity=self.purity, min_sample=dims+1)  # 划分粒球
        # balls.remnove_overlap()      # 去重

        # 对每个少数类进行插值
        for class_sample, n_samples in self.sampling_strategy_.items():
            balls_w_e = []         # 记录加权熵的当前类插值球
            balls_density_we = []
            balls_density_p1 = []
            balls_RD = []
            balls_WIE = []
            balls_purity1 = []  # 记录纯度为1的当前类插值球
            for ball in balls.granular_balls:
                if ball.label == class_sample:
                    if ball.purity == 1 and ball.num > dims+1:
                        balls_purity1.append(ball)
                        balls_density_p1.append(self._find_ball_density(
                            ball.radius, dims, ball.num*ball.purity))
                    else:
                        # 计算少数类球样本Instance-Wise Statistic(相对密度)
                        RD = self._find_ball_RD(ball)
                        # 计算少数类球的Ball-Wise Statistic(加权信息熵)
                        balls_WIE.append(self._find_ball_WIE(RD))
                        balls_RD.append(RD)
                        balls_w_e.append(ball)

            # 选择标签相同的球,计算每个球的权重
            balls_WIE = [0 if math.isnan(x) else x for x in balls_WIE] # 将list中的nan特换为0
            Entropy_Threshold = np.mean(balls_WIE)
            for ind in range(len(balls_w_e)-1, -1, -1):
                # 小于阈值的粒球，过滤掉
                if balls_WIE[ind] < Entropy_Threshold:
                    del balls_w_e[ind]
                    del balls_RD[ind]
                else:
                    # 计算筛选出的种子粒球的密度 =》 分配每个球 合成样本数量
                    balls_density_we.append(self._find_ball_density(
                        balls_w_e[ind].radius, dims, balls_w_e[ind].num*balls_w_e[ind].purity))

            weights_densi_we = np.array(balls_density_we)
            weights_densi_p1 = np.array(balls_density_p1)

            if (len(balls_w_e)+len(balls_purity1)) == 0:
                raise RuntimeError("can not find ball to smote")

            X_new_res = np.zeros(dims)
            y_new_res = np.zeros(1)

            # 1：在选出来的加权熵球中插值
            for index, ball in enumerate(balls_w_e):
                # print('++++++++')
                # print("加权球", ind, ball.purity, ball.num)

                # (根据稀疏程度)计算每个球的插值数量
                ball_n_samples = int(
                    math.ceil(n_samples * weights_densi_we[index]/(weights_densi_we.sum()+weights_densi_p1.sum())))
                # print("before", balls_RD[index])

                # 根据相对密度
                if ball.purity != 1:
                    balls_RD[index] = np.delete(
                        balls_RD[index], np.where(balls_RD[index] == 0))  # 筛选多数类

                balls_RD[index] = balls_RD[index]/balls_RD[index].sum()
                index_array = balls_RD[index].argsort()    # 升序排序后的索引
                distance = np.sort(balls_RD[index])        # 升序排序，从近到远

                # 按相对密度选择少数类样本进行插值
                for i in range(ball_n_samples):

                    # 选择样本,计算权重
                    # 循环抽取
                    seed_sample_ind = index_array[i % len(index_array)]
                    seed_sample = ball.data_no_label[seed_sample_ind]

                    # 种子样本与其随机近邻进行插值，无权重参与 =>
                    seed_neigbor_ind = random.randint(0, len(index_array)-1)
                    # 防止随机到种子样本
                    while seed_neigbor_ind == index_array[i % len(index_array)]:
                        seed_neigbor_ind = random.randint(
                            0, len(index_array)-1)
                    seed_neigbor = ball.data_no_label[seed_neigbor_ind]


                    # 加权平均中心
                    new_center = (seed_sample * balls_RD[index][seed_sample_ind] + seed_neigbor *
                                  balls_RD[index][seed_neigbor_ind]) / (balls_RD[index][seed_sample_ind]+balls_RD[index][seed_neigbor_ind])

                    new_radius = np.sum(
                        (seed_sample - seed_neigbor)**2, axis=0) ** 0.5
                    # 从正态（高斯）分布中，以new_center为均值，new_radius/dims为半径，center.size为新样本的维度。
                    X_new = np.random.normal(
                        loc=new_center, scale=new_radius/dims, size=seed_sample.size)
                    X_new_res = np.vstack((X_new, X_new_res))

                y_new = np.full(ball_n_samples, class_sample)
                y_new_res = np.hstack((y_new, y_new_res))

            # 2：在选出来的纯度为1的球中插值
            for index, ball in enumerate(balls_purity1):

                # (根据稀疏程度)计算每个球的插值数量
                ball_n_samples = int(
                    math.ceil(n_samples * weights_densi_p1[index]/(weights_densi_we.sum()+weights_densi_p1.sum())))
                # print('********',ball.num,weights_densi[index],ball_n_samples)

                # 计算球中每个少数类样本到球心的距离; 由于纯度为1，则不用过滤多数类
                distance = np.sum(
                    (ball.data_no_label - ball.center)**2, axis=1) ** 0.5
                index_array = distance.argsort()    # 升序排序后的索引

                def ball_sampling(center, seed, weight=None):       # 采样
                    # 根据中心与种子样本，计算新中心与新半径
                    new_center = center + (seed - center)/2
                    new_radius = np.sum((center - seed)**2, axis=0) ** 0.5
                    # 从正态（高斯）分布中，以new_center为均值，new_radius/dims为半径，center.size为新样本的维度。
                    new_sample = np.random.normal(
                        loc=new_center, scale=new_radius/dims, size=center.size)
                    return new_sample

                # 按距离从近到远选择少数类样本进行插值
                for i in range(ball_n_samples):

                    seed_sample_ind = index_array[i % len(index_array)]
                    seed_sample = ball.data_no_label[seed_sample_ind]

                    # 球心与种子样本进行插值，无权重参与 => 目前好像有点集中

                    # 种子样本与其随机近邻进行插值，无权重参与 =>
                    seed_neigbor_ind = random.randint(0, len(index_array)-1)
                    # 防止随机到种子样本
                    while seed_neigbor_ind == index_array[i % len(index_array)]:
                        seed_neigbor_ind = random.randint(
                            0, len(index_array)-1)
                    seed_neigbor = ball.data_no_label[seed_neigbor_ind]

                    X_new = ball_sampling(seed_sample, seed_neigbor, 0)
                    X_new_res = np.vstack((X_new, X_new_res))

                y_new = np.full(ball_n_samples, class_sample)
                y_new_res = np.hstack((y_new, y_new_res))

            X_new_res = np.delete(X_new_res, -1, 0)
            y_new_res = np.delete(y_new_res, -1, 0)
            X_resampled = np.vstack((X_resampled, X_new_res))
            y_resampled = np.hstack((y_resampled, y_new_res))
        return X_resampled, y_resampled