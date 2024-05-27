from __future__ import division
from torch import nn
import geopandas as gp
import numpy as np
import sklearn
import joblib
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mlt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import random
import copy
import scipy.sparse as sp
from utils import *
import warnings
warnings.filterwarnings('ignore')
from basic_structure import D_GCN, C_GCN, K_GCN,IGNNK
# %matplotlib inline
import seaborn as sns
import heapq
from joblib.parallel import Parallel,delayed

plt.rcParams['figure.figsize'] = (20, 10)

url_census='data/ushcn/tl_2017_us_state/tl_2017_us_state.shp'
A,X,_= load_udata()
X = X[:,:,:,0]
X = X.reshape(1218,120*12)
X = X/100
capacities = np.max(X,axis=1)
X=X.T
meta_locations = pd.read_csv('data/ushcn/latlon.csv',header=None, names=['latitude','longitude'])
meta_locations = meta_locations.astype('float32')
map_us=gp.read_file(url_census,encoding="utf-8")

w = np.max(meta_locations['longitude'])-np.min(meta_locations['longitude'])
h = np.max(meta_locations['latitude'])-np.min(meta_locations['latitude'])
lng_cond = (meta_locations['longitude']>(np.min(meta_locations['longitude']) + w/4))&(meta_locations['longitude']<(np.min(meta_locations['longitude']) + w/4*3  ))
lat_cond = (meta_locations['latitude']>(np.min(meta_locations['latitude']) + h/4))&(meta_locations['latitude']<(np.min(meta_locations['latitude'])+ h/4*3  ))
unknow_set_central = np.where(lng_cond&lat_cond)[0]
unknow_set_central = set(unknow_set_central)
full_set = set(range(0,X.shape[1]))
know_set = full_set - unknow_set_central

def kNN(A_new, test_set, full_set, unknow_set):
    know_set = full_set - unknow_set

    prediction = np.zeros(test_set.shape)
    prediction[:, list(know_set)] = test_set[:, list(know_set)]
    
    for index in list(unknow_set):
        Distance = []
        for index_k in list(know_set):
            Distance.append(A_new[index, index_k])
            min_num_index_list = map(Distance.index, heapq.nlargest(3, Distance))
            
        for choose in min_num_index_list:
            prediction[:, index] = prediction[:, index] + prediction[:, list(know_set)[choose]]/3
    output = prediction.copy()
    prediction[test_set == 0] = 0
    
    missing_index = np.ones(np.shape(test_set))
    missing_index[:, list(unknow_set)] = 0
    
    test_mask = 1 - missing_index
    test_mask[test_set == 0] = 0
    MAE = np.sum(np.abs(prediction - test_set))/np.sum(test_mask)
    
    RMSE = np.sqrt(np.sum((prediction - test_set)*(prediction - test_set))/np.sum(test_mask)) 
    MAPE = np.sum(np.abs(prediction - test_set)/(test_set + 1e-5))/np.sum(test_mask)
    return MAE, RMSE, MAPE , output

# Load IGNNC model for central missing
space_dim = 900 # randomly set 50 of them are missing, training with dynamic graph
time_dim = 6
hidden_dim_s = 100
hidden_dim_t = 15
rank_s = 20
rank_t = 4
K=1
E_maxvalue = 80
STmodel = IGNNK(time_dim,hidden_dim_s,K)

params_old = torch.load('model/GCNmodel_udata_carea.pth')
params_new = {'GNN1.Theta1':params_old['SC1.Theta1'],
              'GNN1.bias':params_old['SC1.bias'],
              'GNN2.Theta1':params_old['SC2.Theta1'],
              'GNN2.bias':params_old['SC2.bias'],
              'GNN3.Theta1':params_old['SC3.Theta1'],
              'GNN3.bias':params_old['SC3.bias']} # Keys redefined, does not influece the result
STmodel.load_state_dict(params_new)

udata_sim = joblib.load('model/udata_sim.joblib')
MAE_t, RMSE_t, MAPE_t, udata_ignnk= test_error_cap(STmodel, unknow_set_central, full_set, X, A,time_dim,capacities)
MAE, RMSE, MAPE, udata_knn = kNN(udata_sim, X,  full_set, unknow_set_central)
print(MAE_t, RMSE_t, MAPE_t)
print(MAE, RMSE, MAPE)