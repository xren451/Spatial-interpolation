
#2D---可以运行

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from spektral.layers import GCNConv
# from sklearn.metrics import mean_squared_error
#
# # Generate synthetic data
# num_nodes = 5
# num_features = 3
# time_steps = 100
#
# np.random.seed(42)
#
# # Random node features (shape: num_nodes, num_features)
# node_features = np.random.rand(time_steps,num_nodes, num_features)
#
# # Random adjacency matrix (shape: num_nodes, num_nodes)
# adj_matrix = np.random.rand(num_nodes, num_nodes)
# adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make it symmetric
# np.fill_diagonal(adj_matrix, 1)  # Add self-loops
#
# # Random labels (shape: num_nodes, 1)
# labels = np.random.rand(time_steps,num_nodes, 1)
#
# # Initialize the Spektral GCN model
# def initialize_spektral_model(input_shape, adj_shape):
#     inputs_feat = Input(shape=(input_shape,))
#     inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)
#
#     gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
#     dense = Dense(8, activation='relu')(gcn_output)
#     dense_out = Dropout(0.2)(dense)
#     outputs = Dense(1)(dense_out)
#
#     model = Model(inputs=[inputs_feat, inputs_adj], outputs=outputs)
#     model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=["mae", "mape"])
#     model.summary()
#     return model
#
# # Convert adjacency matrix to sparse format
# adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)
#
# # Initialize and train the model
# model = initialize_spektral_model(num_features, adj_matrix.shape)
#
# # Train the model
# history = model.fit([node_features, adj_matrix_sparse], labels, epochs=100, verbose=1)
#
# # Predict and evaluate
# predictions = model.predict([node_features, adj_matrix_sparse])
# mse = mean_squared_error(labels, predictions)
# print(f'MSE: {mse}')





#
#
# # Set Hyperpaprameters
#
# import numpy as np
# import pandas as pd
# import datetime
# import matplotlib.pyplot as plt
# import networkx as nx
# from networkx.algorithms import bipartite
# import copy
# from sklearn.preprocessing import StandardScaler
# import scipy.special as special
# import scipy.sparse as sp
# import pickle
# import os
# from math import sqrt
# from numpy import concatenate
# from matplotlib import pyplot
# from pandas import read_csv
# from pandas import DataFrame
# from pandas import concat
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
#
# import tensorflow as tf
# #import tensorflow_addons as tfa
# import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Sequential
# #from tensorflow.keras.layers import Dense, GRU, Input
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Input, Dropout, Dense, LeakyReLU, GRU, Concatenate, Reshape, Softmax, Attention
# from tensorflow.keras import activations
# from tensorflow.keras.models import Model
# #from keras.layers import LeakyReLU
# from spektral.layers import  GCSConv, DiffusionConv, GATConv, ARMAConv, GCNConv
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Bidirectional
# import torch
#
# tf.keras.backend.clear_session()#clear all
# tf.autograph.set_verbosity(0)#reset to zero
# learning_rate = 0.0001
# batch_size =24
# epochs = 100
# seed = 42
# verbose = 1
#
# COVID_features_aftslid = np.random.rand(473, 3, 7)
# COVID_X1_all_aftslid = np.random.rand(473, 7, 30)
# COVID_Y1_all = np.random.rand(473, 7, 1)
# Adj_dist = np.random.rand(7, 7)
# Adj_MeanMI=np.random.rand(473,7,7)
#
# inputs_feat = Input(shape=(COVID_features_aftslid.shape[2], COVID_features_aftslid.shape[1], ))#7*6=Node*feature
# inputs_adj = Input(shape=(Adj_dist.shape[1],Adj_dist.shape[1], ))
#
# #GCNConv, ARMAConv,etc are free to replace.
# GCN_output_P = GCSConv(16, dropout_rate=0.3,
#                         activation='relu', use_bias=True)([inputs_feat, inputs_adj])
# dense_P = Dense(8, activation='relu')(GCN_output_P)
# dense_out = Dropout(0.2)(dense_P)
# outputs = Dense(1)(dense_out)
#
# model = Model(inputs=[inputs_feat, inputs_adj], outputs=outputs)
# model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=["mae", "mape"])
# model.summary()
#
# scaler = MinMaxScaler(feature_range=(0, 1))
#
#
# print(COVID_features_aftslid.shape)#473*3*7
# COVID_features_aftslid= np.transpose(COVID_features_aftslid,(0,2,1))#473*7*3
# print(COVID_X1_all_aftslid.shape)#473*7*30
# print(np.array(COVID_Y1_all).shape)#(473, 7, 1)
# print(np.array(Adj_MeanMI).shape)#473*7*7
#
# #Normalization
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaled1 = scaler.fit_transform(COVID_Y1_all[:,:,0])
# COVID_Y1_all[:,:,0]=scaled1
#
# tf.keras.backend.clear_session()#clear all
# tf.autograph.set_verbosity(0)#reset to zero
#
# ChangeAdj=Adj_MeanMI
# x_train=COVID_X1_all_aftslid[:int(len(COVID_X1_all_aftslid)*0.8),:,:]
# x_val=COVID_X1_all_aftslid[int(len(COVID_X1_all_aftslid)*0.8):int(len(COVID_X1_all_aftslid)*0.9),:,:]
# x_test=COVID_X1_all_aftslid[int(len(COVID_X1_all_aftslid)*0.9):,:,:]
# features_train=COVID_features_aftslid[:int(len(COVID_features_aftslid)*0.8),:,:]
# features_val=COVID_features_aftslid[int(len(COVID_features_aftslid)*0.8):int(len(COVID_features_aftslid)*0.9),:,:]
# features_test=COVID_features_aftslid[int(len(COVID_features_aftslid)*0.9):,:,:]
# adj_train=ChangeAdj[:int(len(ChangeAdj)*0.8),:,:]
# adj_val=ChangeAdj[int(len(ChangeAdj)*0.8):int(len(ChangeAdj)*0.9),:,:]
# adj_test=ChangeAdj[int(len(ChangeAdj)*0.9):,:,:]
# y_train=COVID_Y1_all[:int(len(COVID_Y1_all)*0.8),:,:]
# y_val=COVID_Y1_all[int(len(COVID_Y1_all)*0.8):int(len(COVID_Y1_all)*0.9),:,:]
# y_test=COVID_Y1_all[int(len(COVID_Y1_all)*0.9):,:,:]
#
# print(y_train.shape)
# #%%
# tf.config.experimental_run_functions_eagerly(True)
# # Set Hyperpaprameters
# tf.keras.backend.clear_session()#clear all
# tf.autograph.set_verbosity(0)#reset to zero
#
#
# # hists = model.fit(x= [x_train, features_train, adj_train],
# #                   y= y_train, verbose=verbose, epochs=500,
# #                   batch_size=batch_size, #validation_split=0.1,
# #                   validation_data=([x_val, features_val, adj_val], y_val),
# #                   #test_data=([x_test, features_test, adj_test], y_test),
# #   )
#
# ##If you want to train faster, try earlystopping.
#
# hists = model.fit(x= [features_train, adj_train],
#                   y= y_train, verbose=verbose, epochs=500,
#                   batch_size=batch_size, #validation_split=0.1,
#                   validation_data=([features_val, adj_val], y_val),
#                   #test_data=([x_test, features_test, adj_test], y_test),
#                     callbacks=[EarlyStopping(#monitor="val_mean_absolute_error",
#                       monitor="val_loss", patience=20, restore_best_weights=True)],
#   )
# def mae(y_true, y_pred):
#     return np.mean(np.abs(y_pred - y_true))
#
# def rmse(y_true, y_pred):
#     return np.sqrt(np.mean(np.square(y_pred - y_true)))
#
# def mape(y_true, y_pred, threshold=0.1):
#     v = np.clip(np.abs(y_true), threshold, None)
#     diff = np.abs((y_true - y_pred) / v)
#     return np.mean(diff, axis=-1).mean()
# ###Write the function of mre
# def mre(y_true, y_pred):
#     return np.sum(np.abs(y_pred - y_true))/np.sum(np.abs(y_true))
# def get_metrics(y, yp):
#     return {
#
#         "mae": np.round(mae(y, yp), 4),
#         "mape": np.round(mape(y, yp),4),
#         "mre": np.round(mre(y, yp),4),
#         "rmse": np.round(rmse(y, yp), 4),
#         #"MPE": np.round(MPE(y, yp), 4),
#         #"R2": np.round(R_squared(y, yp),4)
#     }
#
# # scaled1 = scaler.fit_transform(MeanMI)
# #Get the tensor of test set after masking it.
# test_predicted = model.predict([features_test, adj_test])
# get_metrics(y_test.flatten(), test_predicted.flatten())





# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from spektral.layers import GCNConv
# from sklearn.metrics import mean_squared_error
#
# # Generate synthetic data
# num_stations = 5
# num_features = 1
# time_steps = 100  # T
# d_model = 16  # d_model
#
# np.random.seed(42)
# station_coords = np.random.rand(num_stations, 2)
# temperature_data = np.random.rand(num_stations, time_steps, num_features) * 30  # 0-30度的温度数据
#
# # 待插值站点的索引
# target_index = 4
# target_coords = station_coords[target_index]
# true_values = temperature_data[target_index].flatten()
#
# # 距离反向加权插值
# def idw_interpolation(station_coords, temperatures, target_point, power=2):
#     distances = np.linalg.norm(station_coords - target_point, axis=1)
#     weights = 1 / (distances ** power)
#     weights /= weights.sum()
#     interpolated_value = np.dot(weights, temperatures)
#     return interpolated_value
#
# # 初始化插值结果
# idw_estimate = np.array([idw_interpolation(station_coords[:4], temperature_data[:4, t].flatten(), target_coords) for t in range(time_steps)])
#
# # 构建邻接矩阵
# def calculate_inverse_distance_adj(station_coords):
#     num_nodes = station_coords.shape[0]
#     adj = np.zeros((num_nodes, num_nodes))
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i != j:
#                 distance = np.linalg.norm(station_coords[i] - station_coords[j])
#                 adj[i, j] = np.exp(-distance)
#     np.fill_diagonal(adj, 1)  # Add self-loops
#     return adj
#
# adj_matrix = calculate_inverse_distance_adj(station_coords)
# adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)
#
# # 定义GCN模型
# def GCN_model(input_shape, adj_shape):
#     inputs_feat = Input(shape=(input_shape,))
#     inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)
#     gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
#     model = Model(inputs=[inputs_feat, inputs_adj], outputs=gcn_output)
#     return model
#
# # 定义MLP模型
# def MLP_model(input_dim, output_dim):
#     inputs = Input(shape=(input_dim,))
#     x = Dense(64, activation='relu')(inputs)
#     x = Dense(128, activation='relu')(x)
#     outputs = Dense(output_dim, activation='relu')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
#
# # 初始化模型
# gcn_model = GCN_model(num_features, adj_matrix.shape)
# mlp_all_model = MLP_model(2 * num_stations, num_stations * d_model)
# mlp_one_model = MLP_model(2, d_model)
#
# # 编码位置信息
# all_coords = station_coords.flatten().reshape(1, -1)
# H_pos_all = mlp_all_model.predict(all_coords).reshape((num_stations, d_model))
# H_pos_one = mlp_one_model.predict(target_coords.reshape(1, -1)).reshape((d_model,))
#
# # 训练GCN
# x = temperature_data.transpose((1, 0, 2))  # (time_steps, num_stations, num_features)
# adj = adj_matrix_sparse
#
# # Apply TimeDistributed wrapper for GCN
# gcn_outputs = []
# for t in range(time_steps):
#     gcn_output = gcn_model([x[t], adj])
#     gcn_outputs.append(gcn_output)
#
# H_gcn = tf.stack(gcn_outputs, axis=0)
#
# # 计算最终估计值
# H_gcn_reshaped = tf.reshape(H_gcn, (time_steps, num_stations, -1))  # (time_steps, num_stations, 16)
# H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
# H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)
#
# # Perform multiplication and sum across appropriate dimensions
# estimated_values = tf.einsum('tnd,nd->tn', H_gcn_reshaped, H_pos_all_reshaped)  # (time_steps, num_stations)
# estimated_values = tf.einsum('tn,d->t', estimated_values, H_pos_one)  # (time_steps,)
#
# # 计算损失
# mse = mean_squared_error(true_values, estimated_values)
# print(f'MSE: {mse}')


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from spektral.layers import GCNConv
# from sklearn.metrics import mean_squared_error
#
# # Generate synthetic data
# num_stations = 5
# num_features = 1
# time_steps = 100  # T
# d_model = 16  # d_model
#
# np.random.seed(42)
# station_coords = np.random.rand(num_stations, 2)
# temperature_data = np.random.rand(num_stations, time_steps, num_features) * 30  # 0-30度的温度数据
#
# # 待插值站点的索引
# target_index = 4
# target_coords = station_coords[target_index]
# true_values = temperature_data[target_index].flatten()
#
#
# # 距离反向加权插值
# def idw_interpolation(station_coords, temperatures, target_point, power=2):
#     distances = np.linalg.norm(station_coords - target_point, axis=1)
#     weights = 1 / (distances ** power)
#     weights /= weights.sum()
#     interpolated_value = np.dot(weights, temperatures)
#     return interpolated_value
#
#
# # 初始化插值结果
# idw_estimate = np.array(
#     [idw_interpolation(station_coords[:4], temperature_data[:4, t].flatten(), target_coords) for t in
#      range(time_steps)])
#
#
# # 构建邻接矩阵
# def calculate_inverse_distance_adj(station_coords):
#     num_nodes = station_coords.shape[0]
#     adj = np.zeros((num_nodes, num_nodes))
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i != j:
#                 distance = np.linalg.norm(station_coords[i] - station_coords[j])
#                 adj[i, j] = np.exp(-distance)
#     np.fill_diagonal(adj, 1)  # Add self-loops
#     return adj
#
#
# adj_matrix = calculate_inverse_distance_adj(station_coords)
# adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)
#
#
# # 定义GCN模型
# def GCN_model(input_shape, adj_shape):
#     inputs_feat = Input(shape=(input_shape,))
#     inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)
#     gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
#     model = Model(inputs=[inputs_feat, inputs_adj], outputs=gcn_output)
#     return model
#
#
# # 定义MLP模型
# def MLP_model(input_dim, output_dim):
#     inputs = Input(shape=(input_dim,))
#     x = Dense(64, activation='relu')(inputs)
#     x = Dense(128, activation='relu')(x)
#     outputs = Dense(output_dim, activation='relu')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
#
#
# # 初始化模型
# gcn_model = GCN_model(num_features, adj_matrix.shape)
# mlp_all_model = MLP_model(2 * num_stations, num_stations * d_model)
# mlp_one_model = MLP_model(2, d_model)
#
# # 编码位置信息
# all_coords = station_coords.flatten().reshape(1, -1)
# H_pos_all = mlp_all_model.predict(all_coords).reshape((num_stations, d_model))
# H_pos_one = mlp_one_model.predict(target_coords.reshape(1, -1)).reshape((d_model,))
#
# # 训练GCN
# x = temperature_data.transpose((1, 0, 2))  # (time_steps, num_stations, num_features)
# adj = adj_matrix_sparse
#
# # Define training process
# optimizer = Adam(learning_rate=0.1)
# loss_fn = tf.keras.losses.MeanSquaredError()
#
# # Training loop
# epochs = 1000
# batch_size = 10
#
# for epoch in range(epochs):
#     epoch_loss = 0
#     num_batches = time_steps // batch_size
#     for batch in range(num_batches):
#         with tf.GradientTape() as tape:
#             batch_indices = range(batch * batch_size, (batch + 1) * batch_size)
#             gcn_outputs = []
#             for t in batch_indices:
#                 gcn_output = gcn_model([x[t], adj])
#                 gcn_outputs.append(gcn_output)
#
#             H_gcn = tf.stack(gcn_outputs, axis=0)
#
#             # 计算最终估计值
#             H_gcn_reshaped = tf.reshape(H_gcn, (batch_size, num_stations, -1))  # (batch_size, num_stations, 16)
#             H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
#             H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)
#
#             # Perform multiplication and sum across appropriate dimensions
#             estimated_values = tf.einsum('bnd,nd->bn', H_gcn_reshaped, H_pos_all_reshaped)  # (batch_size, num_stations)
#             estimated_values = tf.einsum('bn,d->b', estimated_values, H_pos_one)  # (batch_size,)
#
#             batch_true_values = true_values[batch_indices]
#             loss = loss_fn(batch_true_values, estimated_values)
#
#         grads = tape.gradient(loss, gcn_model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, gcn_model.trainable_variables))
#         epoch_loss += loss.numpy()
#
#     epoch_loss /= num_batches
#     print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')
#
# # Predict final values
# gcn_outputs = []
# for t in range(time_steps):
#     gcn_output = gcn_model([x[t], adj])
#     gcn_outputs.append(gcn_output)
#
# H_gcn = tf.stack(gcn_outputs, axis=0)
#
# # 计算最终估计值
# H_gcn_reshaped = tf.reshape(H_gcn, (time_steps, num_stations, -1))  # (time_steps, num_stations, 16)
# H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
# H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)
#
# # Perform multiplication and sum across appropriate dimensions
# estimated_values = tf.einsum('tnd,nd->tn', H_gcn_reshaped, H_pos_all_reshaped)  # (time_steps, num_stations)
# estimated_values = tf.einsum('tn,d->t', estimated_values, H_pos_one)  # (time_steps,)
#
# # 计算损失
# mse = mean_squared_error(true_values, estimated_values)
# print(f'MSE: {mse}')





#Normalization on the last code.
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data
num_stations = 5
num_features = 1
time_steps = 100  # T
d_model = 16  # d_model

np.random.seed(42)
station_coords = np.random.rand(num_stations, 2)
temperature_data = np.random.rand(num_stations, time_steps, num_features) * 30  # 0-30度的温度数据

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_data = temperature_data.reshape(-1, 1)
temperature_data = scaler.fit_transform(temperature_data)
temperature_data = temperature_data.reshape(num_stations, time_steps, num_features)

# 待插值站点的索引
target_index = 4
target_coords = station_coords[target_index]
true_values = temperature_data[target_index].flatten()

# 距离反向加权插值
def idw_interpolation(station_coords, temperatures, target_point, power=2):
    distances = np.linalg.norm(station_coords - target_point, axis=1)
    weights = 1 / (distances ** power)
    weights /= weights.sum()
    interpolated_value = np.dot(weights, temperatures)
    return interpolated_value

# 初始化插值结果
idw_estimate = np.array(
    [idw_interpolation(station_coords[:4], temperature_data[:4, t].flatten(), target_coords) for t in
     range(time_steps)])

# 构建邻接矩阵
def calculate_inverse_distance_adj(station_coords):
    num_nodes = station_coords.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = np.linalg.norm(station_coords[i] - station_coords[j])
                adj[i, j] = np.exp(-distance)
    np.fill_diagonal(adj, 1)  # Add self-loops
    return adj

adj_matrix = calculate_inverse_distance_adj(station_coords)
adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)

# 定义GCN模型
def GCN_model(input_shape, adj_shape):
    inputs_feat = Input(shape=(input_shape,))
    inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)
    gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
    model = Model(inputs=[inputs_feat, inputs_adj], outputs=gcn_output)
    return model

# 定义MLP模型
def MLP_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(output_dim, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 初始化模型
gcn_model = GCN_model(num_features, adj_matrix.shape)
mlp_all_model = MLP_model(2 * num_stations, num_stations * d_model)
mlp_one_model = MLP_model(2, d_model)

# 编码位置信息
all_coords = station_coords.flatten().reshape(1, -1)
H_pos_all = mlp_all_model.predict(all_coords).reshape((num_stations, d_model))
H_pos_one = mlp_one_model.predict(target_coords.reshape(1, -1)).reshape((d_model,))

# 训练GCN
x = temperature_data.transpose((1, 0, 2))  # (time_steps, num_stations, num_features)
adj = adj_matrix_sparse

# Define training process
optimizer = Adam(learning_rate=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training loop
epochs = 100
batch_size = 10

for epoch in range(epochs):
    epoch_loss = 0
    num_batches = time_steps // batch_size
    for batch in range(num_batches):
        with tf.GradientTape() as tape:
            batch_indices = range(batch * batch_size, (batch + 1) * batch_size)
            gcn_outputs = []
            for t in batch_indices:
                gcn_output = gcn_model([x[t], adj])
                gcn_outputs.append(gcn_output)

            H_gcn = tf.stack(gcn_outputs, axis=0)

            # 计算最终估计值
            H_gcn_reshaped = tf.reshape(H_gcn, (batch_size, num_stations, -1))  # (batch_size, num_stations, 16)
            H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
            H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)

            # Perform multiplication and sum across appropriate dimensions
            estimated_values = tf.einsum('bnd,nd->bn', H_gcn_reshaped, H_pos_all_reshaped)  # (batch_size, num_stations)
            estimated_values = tf.einsum('bn,d->b', estimated_values, H_pos_one)  # (batch_size,)

            batch_true_values = true_values[batch_indices]
            loss = loss_fn(batch_true_values, estimated_values)

        grads = tape.gradient(loss, gcn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, gcn_model.trainable_variables))
        epoch_loss += loss.numpy()

    epoch_loss /= num_batches
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

# Predict final values
gcn_outputs = []
for t in range(time_steps):
    gcn_output = gcn_model([x[t], adj])
    gcn_outputs.append(gcn_output)

H_gcn = tf.stack(gcn_outputs, axis=0)

# 计算最终估计值
H_gcn_reshaped = tf.reshape(H_gcn, (time_steps, num_stations, -1))  # (time_steps, num_stations, 16)
H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)

# Perform multiplication and sum across appropriate dimensions
estimated_values = tf.einsum('tnd,nd->tn', H_gcn_reshaped, H_pos_all_reshaped)  # (time_steps, num_stations)
estimated_values = tf.einsum('tn,d->t', estimated_values, H_pos_one)  # (time_steps,)

# Convert Tensor to NumPy array before reshaping
estimated_values = estimated_values.numpy()

# Denormalize the predictions
# estimated_values = scaler.inverse_transform(estimated_values.reshape(-1, 1)).flatten()
# true_values = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()

# 计算损失
mse = mean_squared_error(true_values, estimated_values)
print(f'MSE: {mse}')
