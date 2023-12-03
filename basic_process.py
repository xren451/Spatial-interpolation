import tensorflow as tf
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import read_csv
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import math

def adj_dist(lat_arr,long_arr):#计算初始距离，用a表示
    u=[]
    for i in range(lat_arr.shape[0]):
        for j in range(long_arr.shape[0]):
            u.append(arc2dis(lat_arr.iloc[i],long_arr.iloc[i],lat_arr.iloc[j],long_arr.iloc[j]))
    u=np.array(u)
    u=exp(-u**2/(2*(np.std(u))**2))
    u=np.array(u)
    a=u.reshape(len(lat_arr),len(long_arr))
    adj=a
    return adj

def Txt2Tensor(pathroot):#Input:Path root Output:samp*var*station.Eg:data=Getdire('Data/COVID/raw')(503,3,7)
    import os
    import pandas as pd
    from sklearn import linear_model
    import random
    import numpy as np
    PATH_ROOT = os.getcwd()
    ROOT = os.path.join(PATH_ROOT, pathroot)
    filenames = os.listdir(ROOT)
    data = []
    for i in filenames:
        PATH_CSV = os.path.join(ROOT, i)
        print(np.array(PATH_CSV).shape)
        with open(PATH_CSV, 'r') as file:
            # 使用splitlines()方法将文件内容按行分割成列表
            content_list = file.read().splitlines()

        # 将列表转换为NumPy数组
        content_matrix = np.array([list(map(float, line.split())) for line in content_list])
        data.append(content_matrix)
    data = np.array(data).transpose(1, 2, 0)
    return data
def arc2dis(LatA,LonA,LatB,LonB):
    import numpy as np
    import pandas as pd
    import math
    LatA,LatB,LonA,LonB = map(math.radians, [float(LatA), float(LatB), float(LonA), float(LonB)])
    C = math.sin((LatA-LatB)/2)* math.sin((LatA-LatB)/2)+ math.cos(LatA)*math.cos(LatB)*math.sin((LonA-LonB)/2)*math.sin((LonA-LonB)/2)
    ra=6378137
    pi=3.1415926
    dist=2*math.asin(math.sqrt(C))*6371000
    dist=round(dist/1000,3)
    return dist

def adj_dist(lat_arr,long_arr):#计算初始距离，用a表示
    u=[]
    for i in range(lat_arr.shape[0]):
        for j in range(long_arr.shape[0]):
            u.append(arc2dis(lat_arr.iloc[i],long_arr.iloc[i],lat_arr.iloc[j],long_arr.iloc[j]))
    u=np.array(u)
    u=exp(-u**2/(2*(np.std(u))**2))
    u=np.array(u)
    a=u.reshape(len(lat_arr),len(long_arr))
    adj=a
    return adj
def exp(a):#该函数输入列表类型的数据后，返回列表数据的指数。
    exp_a=[]
    for i in range(a.shape[0]):
        exp_a.append(math.exp(a[i]))
    return exp_a


def normalize_3d_array(arr):
    # 获取数组形状
    shape = arr.shape

    # 遍历每个 z 方向的二维矩阵
    for z in range(shape[2]):
        # 获取当前 z 方向的二维矩阵
        matrix = arr[:, :, z]

        # 归一化
        normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        # 将归一化后的矩阵保存回原始数组
        arr[:, :, z] = normalized_matrix

    return arr

def calc_MI(X,Y,bins):
    import numpy as np
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    import numpy as np
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H