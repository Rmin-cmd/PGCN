import os
import sys

from scipy.spatial import distance
import scipy.io as sio
import numpy as np
import scipy.sparse as sp


def get_ini_dis_m_FACED():
    """get initial distance matrix"""

    # m1 = np.load('../data/pos.npy')[:30] * 100
#     chan_names = ['Fp1', 'Fp2', 'Fz','F3','F4','F7',
# 'F8',
# 'FC1',
# 'FC2',
# 'FC5',
# 'FC6',
# 'Cz',
# 'C3',
# 'C4',
# 'T3',
# 'T4',
# 'CP1',
# 'CP2',
# 'CP5',
# 'CP6',
# 'Pz',
# 'P3',
# 'P4',
# 'T5',
# 'T6',
# 'PO3',
# 'PO4',
# 'Oz',
# 'O1',
# 'O2']
    m1 = sio.loadmat('..\data\pos.mat')['pos'] * 100
    dis_m1 = distance.cdist(m1, m1, 'euclidean')

    # 对元素进行检查，小于0时置0
    zero_matrix = np.zeros((30, 30))
    dis_m1 = np.where(dis_m1 > 0, dis_m1, zero_matrix)

    return dis_m1


def convert_dis_m_FACED(adj_matrix, delta=8):
    """
    将距离有关的adj矩阵换算成距离平方反比矩阵.
    :param adj_matrix:
    :param delta: 调和系数，控制距离的相关度
    :return: 返回处理好的adjacency matrix
    """
    eye_62 = np.eye(30, dtype=float)
    adj_matrix = eye_62 + adj_matrix  # 将对角线上的元素置1，防止除法报错
    adj_quadratic = np.power(adj_matrix, 2)#
    adj_matrix = delta / adj_quadratic
    adj_matrix = np.where(adj_matrix > 1, 1, adj_matrix)  # adj最大值不大于1
    adj_matrix = np.where(adj_matrix < 0.1, 0, adj_matrix)  # 稀疏化，将较小的值置0

    return adj_matrix


def return_coordinates():
    "return absolute coordinates"
    return sio.loadmat('..\data\pos.mat')['pos'] * 100


