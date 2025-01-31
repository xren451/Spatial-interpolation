from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
#from sklearn.externals import joblib
import joblib
import scipy.io
import torch
from torch import nn
from basic_process import *
"""
Geographical information calculation
"""
def get_long_lat(sensor_index,loc = None):
    """
        Input the index out from 0-206 to access the longitude and latitude of the nodes
    """
    if loc is None:
        locations = pd.read_csv('data/metr/graph_sensor_locations.csv')
    else:
        locations = loc
    lng = locations['longitude'].loc[sensor_index]
    lat = locations['latitude'].loc[sensor_index]
    return lng.to_numpy(),lat.to_numpy()

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000


"""
Load datasets
"""

def load_metr_la_rdata():
    if (not os.path.isfile("data/metr/adj_mat.npy")
            or not os.path.isfile("data/metr/node_values.npy")):
        with zipfile.ZipFile("data/metr/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/metr/")

    A = np.load("data/metr/adj_mat.npy")
    X = np.load("data/metr/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    return A, X

def load_ndbc_data():
    X_raw = np.load('data/NDBC/all.npy')
    # Raw data and raw adjacency matrix without mapminmax
    X_raw = np.load('data/NDBC/all.npy')
    # X_raw=normalize_3d_array(X_raw)
    Station_info = pd.read_csv('data/NDBC/Station_info.csv')
    NDBC_lat = pd.DataFrame(Station_info.iloc[:, 1])
    NDBC_long = pd.DataFrame(Station_info.iloc[:, 3])
    NDBC_ID = pd.DataFrame(Station_info.iloc[:, 0])
    Adj_dist = adj_dist(NDBC_lat, NDBC_long)
    Station_info = pd.read_csv('data/NDBC/Station_info.csv')
    NDBC_lat = pd.DataFrame(Station_info.iloc[:, 1])
    NDBC_long = pd.DataFrame(Station_info.iloc[:, 3])
    NDBC_ID = pd.DataFrame(Station_info.iloc[:, 0])
    Adj_dist = adj_dist(NDBC_lat, NDBC_long)
    j = 5  # J-th colum, start from 5 as the first feature.
    X_raw_0 = X_raw[:, j, :]  # GET the first feature
    # print(X_raw_0.shape)
    # Count the num of missing values in each column
    missing_counts_per_column = np.sum(np.isnan(X_raw_0), axis=0)
    # print results
    # print("Incomplete data number in each column：", missing_counts_per_column)

    # Get the index if the value is not zero
    # Find the columns where missing values exist.
    columns_with_missing_data = np.any(np.isnan(X_raw_0), axis=0)

    # Get the column numbers when missing value exist.
    missing_columns = np.where(columns_with_missing_data)[0]

    # Print results
    # print("The column numbers with missing values are：", missing_columns)

    # Get new data after deletion.
    # Delete those columns(Stations) if there is not any features.
    result = np.delete(X_raw_0, missing_columns, axis=1)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    # Get new adjacency matrix after deletion.
    NDBC_long = NDBC_long.transpose()
    NDBC_long_aft = NDBC_long.drop(columns=missing_columns)
    # Reconstruct the index in the DataFrame and get the longitude after deletion.
    NDBC_long_aft = NDBC_long_aft.reset_index(drop=True)
    print(NDBC_long_aft.shape)

    NDBC_lat = NDBC_lat.transpose()
    NDBC_lat_aft = NDBC_lat.drop(columns=missing_columns)
    # Reconstruct the index in the DataFrame and get the latitude after deletion.
    NDBC_lat_aft = NDBC_lat_aft.reset_index(drop=True)
    # print(NDBC_lat_aft.shape)
    # GEt the new ADJ matrix.
    Adj_dist = adj_dist(NDBC_lat_aft.transpose(), NDBC_long_aft.transpose())
    X=result.transpose()
    A=Adj_dist
    X = X.astype(np.float32)
    return A, X


def generate_nerl_data():
    # %% Obtain all the file names
    filepath = 'data/nrel/al-pv-2006'
    files = os.listdir(filepath)

    # %% Begin parse the file names and store them in a pandas Dataframe
    tp = [] # Type
    lat = [] # Latitude
    lng =[] # Longitude
    yr = [] # Year
    pv_tp = [] # PV_type
    cap = [] # Capacity MW
    time_itv = [] # Time interval
    file_names =[]
    for _file in files:
        parse = _file.split('_')
        if parse[-2] == '5':
            tp.append(parse[0])
            lat.append(np.double(parse[1]))
            lng.append(np.double(parse[2]))
            yr.append(np.int(parse[3]))
            pv_tp.append(parse[4])
            cap.append(np.int(parse[5].split('MW')[0]))
            time_itv.append(parse[6])
            file_names.append(_file)
        else:
            pass

    files_info = pd.DataFrame(
        np.array([tp,lat,lng,yr,pv_tp,cap,time_itv,file_names]).T,
        columns=['type','latitude','longitude','year','pv_type','capacity','time_interval','file_name']
    )
    # %% Read the time series into a numpy 2-D array with 137x105120 size
    X = np.zeros((len(files_info),365*24*12))
    for i in range(files_info.shape[0]):
        f = filepath + '/' + files_info['file_name'].loc[i]
        d = pd.read_csv(f)
        assert d.shape[0] == 365*24*12, 'Data missing!'
        X[i,:] = d['Power(MW)']
        print(i/files_info.shape[0]*100,'%')

    np.save('data/nrel/nerl_X.npy',X)
    files_info.to_pickle('data/nrel/nerl_file_infos.pkl')
    # %% Get the adjacency matrix based on the inverse of distance between two nodes
    A = np.zeros((files_info.shape[0],files_info.shape[0]))

    for i in range(files_info.shape[0]):
        for j in range(i+1,files_info.shape[0]):
            lng1 = lng[i]
            lng2 = lng[j]
            lat1 = lat[i]
            lat2 = lat[j]
            d = haversine(lng1,lat1,lng2,lat2)
            A[i,j] = d

    A = A/7500 # distance / 7.5 km
    A += A.T + np.diag(A.diagonal())
    A = np.exp(-A)
    np.save('data/nrel/nerl_A.npy',A)

def load_nerl_data():
    if (not os.path.isfile("data/nrel/nerl_X.npy")
            or not os.path.isfile("data/nrel/nerl_A.npy")):
        with zipfile.ZipFile("data/nrel/al-pv-2006.zip", 'r') as zip_ref:
            zip_ref.extractall("data/nrel/al-pv-2006")
        generate_nerl_data()
    X = np.load('data/nrel/nerl_X.npy')
    A = np.load('data/nrel/nerl_A.npy')
    files_info = pd.read_pickle('data/nrel/nerl_file_infos.pkl')

    X = X.astype(np.float32)
    # X = (X - X.mean())/X.std()
    return A,X,files_info

def generate_ushcn_data():
    pos = []
    Utensor = np.zeros((1218, 120, 12, 2))
    Omissing = np.ones((1218, 120, 12, 2))
    with open("data/ushcn/Ulocation", "r") as f:
        loc = 0
        for line in f.readlines():
            poname = line[0:11]
            pos.append(line[13:30])
            with open("data/ushcn/ushcn.v2.5.5.20191231/"+ poname +".FLs.52j.prcp", "r") as fp:
                temp = 0
                for linep in fp.readlines():
                    if int(linep[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linep[17 + 9*i:22 + 9*i]
                            p_temp = int(str_temp)
                            if p_temp == -9999:
                                Omissing[loc, temp, i, 0] = 0
                            else:
                                Utensor[loc, temp, i, 0] = p_temp
                        temp = temp + 1   
            with open("data/ushcn/ushcn.v2.5.5.20191231/"+ poname +".FLs.52j.tavg", "r") as ft:
                temp = 0
                for linet in ft.readlines():
                    if int(linet[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linet[17 + 9*i:22 + 9*i]
                            t_temp = int(str_temp)
                            if t_temp == -9999:
                                Omissing[loc, temp, i, 1] = 0
                            else:
                                Utensor[loc, temp, i, 1] = t_temp
                        temp = temp + 1    
            loc = loc + 1
            
    latlon =np.loadtxt("data/ushcn/latlon.csv",delimiter=",")
    sim = np.zeros((1218,1218))

    for i in range(1218):
        for j in range(1218):
            sim[i,j] = haversine(latlon[i, 1], latlon[i, 0], latlon[j, 1], latlon[j, 0]) #RBF
    sim = np.exp(-sim/10000/10)

    joblib.dump(Utensor,'data/ushcn/Utensor.joblib')
    joblib.dump(Omissing,'data/ushcn/Omissing.joblib')
    joblib.dump(sim,'data/ushcn/sim.joblib')            

def load_udata():
    if (not os.path.isfile("data/ushcn/Utensor.joblib")
            or not os.path.isfile("data/ushcn/sim.joblib")):
        with zipfile.ZipFile("data/ushcn/ushcn.v2.5.5.20191231.zip", 'r') as zip_ref:
            zip_ref.extractall("data/ushcn/ushcn.v2.5.5.20191231/")
        generate_ushcn_data()
    X = joblib.load('data/ushcn/Utensor.joblib')
    A = joblib.load('data/ushcn/sim.joblib')
    Omissing = joblib.load('data/ushcn/Omissing.joblib')
    X = X.astype(np.float32)
    return A,X,Omissing

def load_sedata():
    assert os.path.isfile('data/sedata/A.mat')
    assert os.path.isfile('data/sedata/mat.csv')
    A_mat = scipy.io.loadmat('data/sedata/A.mat')
    A = A_mat['A']
    X = pd.read_csv('data/sedata/mat.csv',index_col=0)
    X = X.to_numpy()
    return A,X

def load_pems_data():
    assert os.path.isfile('data/pems/pems-bay.h5')
    assert os.path.isfile('data/pems/distances_bay_2017.csv')
    df = pd.read_hdf('data/pems/pems-bay.h5')
    transfer_set = df.as_matrix()
    distance_df = pd.read_csv('data/pems/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1

    dist_mx = np.zeros((325, 325), dtype=np.float32)

    dist_mx[:] = np.inf

    sensor_ids = df.columns.values.tolist()

    sensor_id_to_ind = {}

    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
        
    for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0

    A_new = adj_mx
    return transfer_set,A_new
"""
Dynamically construct the adjacent matrix
"""

def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def test_error_missing(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0,test_truth):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros([test_truth.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period

    for i in range(0, test_truth.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))

        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]

    o = o*E_maxvalue
    truth = test_truth[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]

    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o

def test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period

    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))

        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    o = o*E_maxvalue
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]
    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]
    
    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o

def test_error_new(STmodel, unknow_set, test_data, A_s, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period
    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        flags = parse_args(sys.argv[1:])
        E_maxvalue = flags.E_maxvalue
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    flags = parse_args(sys.argv[1:])
    dataset=flags.dataset
    if dataset == 'NREL':
        dataset=dataset
    else:
        o = o*E_maxvalue
    truth = test_inputs_s[0:test_inputs.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_inputs.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_inputs.shape[0]//time_dim*time_dim] == 1]
    test_mask =  1 - missing_index_s[0:test_inputs.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    return MAE, RMSE, MAPE

def rolling_test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue,Missing0):
    """
    :It only calculates the last time points' prediction error, and updates inputs each time point
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """  
    
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros([test_data.shape[0] - time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_data.shape[0] - time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs * missing_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i, :] = imputation[0, time_dim-1, :]
    
 
    truth = test_inputs_s[time_dim:test_data.shape[0]]
    o[missing_index_s[time_dim:test_data.shape[0]] == 1] = truth[missing_index_s[time_dim:test_data.shape[0]] == 1]
    
    o = o*E_maxvalue
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    test_mask =  1 - missing_index_s[time_dim:test_data.shape[0]]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
        
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)  #avoid x/0
        
    return MAE, RMSE, MAPE, o

def test_error_cap(STmodel, unknow_set, full_set, test_set, A,time_dim,capacities):
    unknow_set = set(unknow_set)
    
    test_omask = np.ones(test_set.shape)
    test_omask[test_set == 0] = 0
    test_inputs = (test_set * test_omask).astype('float32')
    test_inputs_s = test_inputs#[:, list(proc_set)]

    
    missing_index = np.ones(np.shape(test_inputs))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index#[:, list(proc_set)]
    
    A_s = A#[:, list(proc_set)][list(proc_set), :]
    o = np.zeros([test_set.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_set.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs*missing_inputs
        MF_inputs = MF_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
    o = o*capacities
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    truth = truth*capacities
    o[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1]
    o[truth == 0] = 0
    
    test_mask =  1 - missing_index_s[0:test_set.shape[0]//time_dim*time_dim]
    test_mask[truth == 0] = 0
    
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o
    
def load_data_MI(dataset):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacity = []
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
    elif dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
    elif dataset == 'ndbc':
        A, X = load_ndbc_data()
        print("A's shape is:",A.shape)# A's shape is: (93, 93)
        print("X's shape is:",X.shape)#X's shape is: (93, 8784)
    elif dataset == 'ushcn':
        A,X,Omissing = load_udata()
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
    elif dataset == 'pems':
        A,X = load_pems_data()
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')
    split_line1 = int(X.shape[1] * 0.7)
    training_set = X[:,:split_line1].transpose()
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output
    Station_info=pd.read_csv('data/NDBC/Station_info.csv')
    NDBC_lat=pd.DataFrame(Station_info.iloc[:,1])
    NDBC_long=pd.DataFrame(Station_info.iloc[:,3])
    NDBC_ID=pd.DataFrame(Station_info.iloc[:,0])
    Adj_dist=adj_dist(NDBC_lat,NDBC_long)
    # Read all data
    pathroot = 'data/NDBC/all_stations'
    PATH_ROOT = os.getcwd()
    ROOT = os.path.join(PATH_ROOT, pathroot)
    filenames = os.listdir(ROOT)
    # Sort all files
    filenames.sort()
    data = []
    for i in filenames:
        PATH_CSV = os.path.join(ROOT, i)
        with open(PATH_CSV, 'r') as file:
            # Use splitlines() to divide contents in the documents into lists.
            content_list = file.read().splitlines()
        # Transform lists into Numpy.
        content_matrix = np.array([list(map(float, line.split())) for line in content_list])
        data.append(content_matrix)
    data = np.array(data).transpose(1, 2, 0)
    X_raw = data
    j=5#J-th colum, start from 5 as the first feature.
    X_raw_0=X_raw[:,j,:]#GET the first feature
    print(X_raw_0.shape)
    # Count the num of missing values in each column
    missing_counts_per_column = np.sum(np.isnan(X_raw_0), axis=0)
    # print results
    print("Incomplete data number in each column：", missing_counts_per_column)

    #Get the index if the value is not zero
    # Find the columns where missing values exist.
    columns_with_missing_data = np.any(np.isnan(X_raw_0), axis=0)

    # Get the column numbers when missing value exist.
    missing_columns = np.where(columns_with_missing_data)[0]

    # Print results
    print("The column numbers with missing values are：", missing_columns)

    # Get new data after deletion.
    # Delete those columns (Stations) if there is not any features.
    result = np.delete(X_raw_0, missing_columns, axis=1)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    print(result.shape)#(8784, 93)
    flags = parse_args(sys.argv[1:])
    N_u = flags.n_u
    #Determine group num here
    group_num=result.shape[1]-N_u

    # Initialize MI_store
    MI_store = np.zeros((result.shape[1], result.shape[1]))

    # Calculate mutual information
    for i in range(result.shape[1]):
        for j in range(result.shape[1]):
            MI_store[i, j] = calc_MI(result[:, i], result[:, j], 10)

    # Return column numbers and corresponding mutual information
    column_numbers = []
    MI_values = []

    for i in range(MI_store.shape[0]):
        row = MI_store[i, :]
        row_index = np.arange(MI_store.shape[0])
        row_index = np.delete(row_index, i)  # Exclude the current row's index
        sorted_indices = np.argsort(row[row_index])
        max_indices = row_index[sorted_indices[-group_num:][::-1]]  # Get largest column_num
        column_numbers.append(max_indices)
        MI_values.append([row[j] for j in max_indices])

    # Output results
    for i, (columns, values) in enumerate(zip(column_numbers, MI_values)):
        print(f"Row {i} - Column Numbers: {columns}, MI Values: {values}")


    know_set=np.array(column_numbers)[random.randint(0, X.shape[0]-1),:]
    know_set=set(know_set)
    full_set = set(range(0,X.shape[0]))
    unknow_set = full_set - know_set
    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance,
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix
    return A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity

import torch
import numpy as np
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import *
import random
import pandas as pd
from basic_structure import IGNNK
from basic_process import *
import argparse
import sys
import os
import time
def parse_args(args):
    '''Parse training options user can specify in command line.
    Specify hyper parameters here
    Returns
    -------
    argparse.Namespace
        the output parser object
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when training IGNNK model.",
        epilog="python IGNNK_train.py DATASET, for example: python IGNNK_train.py 'metr' ")
    # Requird input parametrs
    parser.add_argument(
        'dataset',type=str,default='metr',
        help = 'Name of the datasets, select from metr,ndbc, nrel, ushcn, sedata or pems'
    )
    # optional input parameters
    parser.add_argument(
        '--n_o',type=int,default=20,
        help='sampled space dimension'
    )
    parser.add_argument(
        '--h',type=int,default=24,
        help='sampled time dimension'
    )
    parser.add_argument(
        '--z',type=int,default=100,
        help='hidden dimension for graph convolution'
    )
    parser.add_argument(
        '--K',type=int,default=1,
        help='If using diffusion convolution, the actual diffusion convolution step is K+1'
    )
    parser.add_argument(
        '--n_m',type=int,default=10,
        help='number of mask node during training'
    )
    parser.add_argument(
        '--n_u',type=int,default=50,
        help='target locations, n_u locations will be deleted from the training data'
    )
    parser.add_argument(
        '--max_iter',type=int,default=100,
        help='max training episode'
    )
    parser.add_argument(
        '--learning_rate',type=float,default=0.0001,
        help='the learning_rate for Adam optimizer'
    )
    parser.add_argument(
        '--E_maxvalue',type=int,default=80,
        help='the max value from experience'
    )
    parser.add_argument(
        '--batch_size',type=int,default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--to_plot',type=bool,default=True,
        help='Whether to plot the RMSE training result'
    )
    return parser.parse_known_args(args)[0]
def load_data(dataset):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacity = []
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
    elif dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
    elif dataset == 'ndbc':
        A, X = load_ndbc_data()
        print("A's shape is:",A.shape)# A's shape is: (93, 93)
        print("X's shape is:",X.shape)#X's shape is: (93, 8784)
    elif dataset == 'ushcn':
        A,X,Omissing = load_udata()
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
    elif dataset == 'pems':
        A,X = load_pems_data()
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')
    split_line1 = int(X.shape[1] * 0.7)
    training_set = X[:,:split_line1].transpose()
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output
    unknow_set = rand.choice(list(range(0,X.shape[0])),n_u,replace=False)
    unknow_set = set(unknow_set)
    full_set = set(range(0,X.shape[0]))
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance,
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix
    return A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity

def load_data(dataset): ###Raw load data document
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacity = []
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
    elif dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
    elif dataset == 'ndbc':
        A, X = load_ndbc_data()
        print("A's shape is:",A.shape)# A's shape is: (93, 93)
        print("X's shape is:",X.shape)#X's shape is: (93, 8784)
    elif dataset == 'ushcn':
        A,X,Omissing = load_udata()
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
    elif dataset == 'pems':
        A,X = load_pems_data()
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')
    split_line1 = int(X.shape[1] * 0.7)
    training_set = X[:,:split_line1].transpose()
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output
    unknow_set = rand.choice(list(range(0,X.shape[0])),n_u,replace=False)
    unknow_set = set(unknow_set)
    full_set = set(range(0,X.shape[0]))
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance,
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix
    return A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity
"""
Define the test error
"""
def test_error(STmodel, unknow_set, test_data, A_s, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    E_maxvalue = 80
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period
    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    if dataset == 'NREL':
        o = o*capacities[None,:]
    else:
        o = o*E_maxvalue
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1]
    test_mask =  1 - missing_index_s[0:test_set.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    return MAE, RMSE, MAPE
def rolling_test_error(STmodel, unknow_set, test_data, A_s, Missing0):
    """
    :It only calculates the last time points' prediction error, and updates inputs each time point
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    E_maxvalue = 80
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    o = np.zeros([test_set.shape[0] - time_dim, test_inputs_s.shape[1]])
    for i in range(0, test_set.shape[0] - time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs * missing_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i, :] = imputation[0, time_dim-1, :]
    truth = test_inputs_s[time_dim:test_set.shape[0]]
    o[missing_index_s[time_dim:test_set.shape[0]] == 1] = truth[missing_index_s[time_dim:test_set.shape[0]] == 1]
    if dataset == 'NREL':
        o = o*capacities[None,:]
    else:
        o = o*E_maxvalue
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    test_mask =  1 - missing_index_s[time_dim:test_set.shape[0]]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)  #avoid x/0
    return MAE, RMSE, MAPE
def plot_res(RMSE_list,dataset,time_batch):
    """
    Draw Learning curves on testing error
    """
    fig,ax = plt.subplots()
    ax.plot(RMSE_list,label='RMSE_on_test_set',linewidth=3.5)
    ax.set_xlabel('Training Batch (x{:})'.format(time_batch),fontsize=20)
    ax.set_ylabel('RMSE',fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig/learning_curve_{:}.pdf'.format(dataset))
def parse_args(args):
        '''Parse training options user can specify in command line.
        Specify hyper parameters here
        Returns
        -------
        argparse.Namespace
            the output parser object
        '''
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Parse argument used when training IGNNK model.",
            epilog="python IGNNK_train.py DATASET, for example: python IGNNK_train.py 'metr' ")
        # Requird input parametrs
        parser.add_argument(
            'dataset', type=str, default='metr',
            help='Name of the datasets, select from metr,ndbc, nrel, ushcn, sedata or pems'
        )
        # optional input parameters
        parser.add_argument(
            '--n_o', type=int, default=20,
            help='sampled space dimension'
        )
        parser.add_argument(
            '--h', type=int, default=24,
            help='sampled time dimension'
        )
        parser.add_argument(
            '--z', type=int, default=100,
            help='hidden dimension for graph convolution'
        )
        parser.add_argument(
            '--K', type=int, default=1,
            help='If using diffusion convolution, the actual diffusion convolution step is K+1'
        )
        parser.add_argument(
            '--n_m', type=int, default=10,
            help='number of mask node during training'
        )
        parser.add_argument(
            '--n_u', type=int, default=50,
            help='target locations, n_u locations will be deleted from the training data'
        )
        parser.add_argument(
            '--max_iter', type=int, default=100,
            help='max training episode'
        )
        parser.add_argument(
            '--learning_rate', type=float, default=0.0001,
            help='the learning_rate for Adam optimizer'
        )
        parser.add_argument(
            '--E_maxvalue', type=int, default=80,
            help='the max value from experience'
        )
        parser.add_argument(
            '--batch_size', type=int, default=4,
            help='Batch size'
        )
        parser.add_argument(
            '--to_plot', type=bool, default=True,
            help='Whether to plot the RMSE training result'
        )
        return parser.parse_known_args(args)[0]
def load_data_MI_3D(dataset):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacity = []
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
    elif dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
    elif dataset == 'ndbc':
        A, X = load_ndbc_data()
        print("A's shape is:",A.shape)# A's shape is: (93, 93)
        print("X's shape is:",X.shape)#X's shape is: (93, 8784)
    elif dataset == 'ushcn':
        A,X,Omissing = load_udata()
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
    elif dataset == 'pems':
        A,X = load_pems_data()
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')
    split_line1 = int(X.shape[1] * 0.7)
    training_set = X[:,:split_line1].transpose()
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output
    Station_info = pd.read_csv('/home/xren451/rxb/phd/Spatial_interpolation/XBSPA/ModIGNNK/NDBC/Station_info.csv')
    NDBC_lat=pd.DataFrame(Station_info.iloc[:,1])
    NDBC_long=pd.DataFrame(Station_info.iloc[:,3])
    NDBC_ID=pd.DataFrame(Station_info.iloc[:,0])
    Adj_dist=adj_dist(NDBC_lat,NDBC_long)

    split_line1 = int(X.shape[1] * 0.7)
    training_set = X[:,:split_line1].transpose()
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output

    j=5#J-th colum, start from 5 as the first feature.
    pathroot='/home/xren451/rxb/phd/Spatial_interpolation/XBSPA/ModIGNNK/NDBC/all_stations'
    PATH_ROOT = os.getcwd()
    ROOT = os.path.join(PATH_ROOT, pathroot)
    filenames = os.listdir(ROOT)
    # Sort all files
    filenames.sort()
    data = []
    for i in filenames:
        PATH_CSV = os.path.join(ROOT, i)
        with open(PATH_CSV, 'r') as file:
    # Use splitlines() to divide contents in the documents into lists.
            content_list = file.read().splitlines()
    # Transform lists into Numpy.
        content_matrix = np.array([list(map(float, line.split())) for line in content_list])
        data.append(content_matrix)
    data = np.array(data).transpose(1, 2, 0)
    X_raw=data
    X_raw_0=X_raw[:,j,:]#GET the first feature
    print(X_raw_0.shape)
    # Count the num of missing values in each column
    missing_counts_per_column = np.sum(np.isnan(X_raw_0), axis=0)
    # print results
    print("Incomplete data number in each column：", missing_counts_per_column)

    #Get the index if the value is not zero
    # Find the columns where missing values exist.
    columns_with_missing_data = np.any(np.isnan(X_raw_0), axis=0)

    # Get the column numbers when missing value exist.
    missing_columns = np.where(columns_with_missing_data)[0]

    # Print results
    print("The column numbers with missing values are：", missing_columns)

    # Get new data after deletion.
    # Delete those columns (Stations) if there is not any features.
    result = np.delete(X_raw_0, missing_columns, axis=1)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    print(result.shape)#(8784, 93)

    #Determine group num here
    flags = parse_args(sys.argv[1:])
    N_u=flags.n_u
    group_num=result.shape[1]-N_u

    # Initialize MI_store
    MI_store = np.zeros((result.shape[1], result.shape[1]))

    # Calculate mutual information
    for i in range(result.shape[1]):
        for j in range(result.shape[1]):
            MI_store[i, j] = calc_MI(result[:, i], result[:, j], 10)

    # Return column numbers and corresponding mutual information
    column_numbers = []
    MI_values = []

    for i in range(MI_store.shape[0]):
        row = MI_store[i, :]
        row_index = np.arange(MI_store.shape[0])
        row_index = np.delete(row_index, i)  # Exclude the current row's index
        sorted_indices = np.argsort(row[row_index])
        max_indices = row_index[sorted_indices[-group_num:][::-1]]  # Get largest column_num
        column_numbers.append(max_indices)
        MI_values.append([row[j] for j in max_indices])

    # Output results
    for i, (columns, values) in enumerate(zip(column_numbers, MI_values)):
        print(f"Row {i} - Column Numbers: {columns}, MI Values: {values}")

    #Step1: Use full_set
    full_set = set(range(0,X.shape[0]))
    #Step2: Use column_numbers to be the know_set,know_set is the list
    know_set_2D=column_numbers
    #Step3: Get the unknow_set.
    unknow_set_2D={}
    for i in range (len(know_set_2D)):
        unknow_set_2D[i]=full_set-set(know_set_2D[i])
    know_set=np.array(column_numbers)[random.randint(0, X.shape[0]-1),:]# For an arbitray row(station), return closest neighbors to be known set.
    know_set=set(know_set)
    unknow_set = full_set - know_set
    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance,
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix
    training_set_s_3D=[]
    for i in range(len(know_set_2D)):
        training_set_s_3D.append(training_set[:, list(know_set_2D[i])])
    training_set_s_3D=np.array(training_set_s_3D)
    return A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity,unknow_set_2D,know_set_2D,training_set_s_3D