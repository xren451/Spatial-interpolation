import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from spektral.layers import GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import os
import pandas as pd
import numpy as np
import torch



# Generate synthetic data
num_stations = 7
num_features = 1
time_steps = 100 # T
d_model = 16  # d_model

np.random.seed(42)
station_coords = np.random.rand(num_stations, 2)

temperature_data = np.random.rand(num_stations, time_steps, num_features)   # 温度数据

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_data = temperature_data.reshape(-1, 1)
temperature_data = scaler.fit_transform(temperature_data)
temperature_data = temperature_data.reshape(num_stations, time_steps, num_features)
print(type(temperature_data))
print("temperature_data.shape",temperature_data.shape)
print(type(station_coords))
print("station_coords.shape",station_coords.shape)
# Function to calculate the inverse distance adjacency matrix
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

# Define GCN model
def GCN_model(input_shape, adj_shape):
    inputs_feat = Input(shape=(input_shape,))
    inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)
    gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
    model = Model(inputs=[inputs_feat, inputs_adj], outputs=gcn_output)
    return model

# Define MLP model
def MLP_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(output_dim, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Training function
def train_and_predict(station_coords, temperature_data, epochs=100, batch_size=10, learning_rate=0.1, patience=10):
    results = []

    for target_index in range(num_stations):
        target_coords = station_coords[target_index]
        true_values = temperature_data[target_index].flatten()

        adj_matrix = calculate_inverse_distance_adj(station_coords)
        adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)

        gcn_model = GCN_model(num_features, adj_matrix.shape)
        mlp_all_model = MLP_model(2 * num_stations, num_stations * d_model)
        mlp_one_model = MLP_model(2, d_model)

        all_coords = station_coords.flatten().reshape(1, -1)
        H_pos_all = mlp_all_model.predict(all_coords).reshape((num_stations, d_model))
        H_pos_one = mlp_one_model.predict(target_coords.reshape(1, -1)).reshape((d_model,))

        x = temperature_data.transpose((1, 0, 2))  # (time_steps, num_stations, num_features)
        adj = adj_matrix_sparse

        optimizer = Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Initialize early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Training loop
        epoch_loss = []
        val_loss = []
        best_val_loss = np.inf
        patience_counter = 0

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

                    H_gcn_reshaped = tf.reshape(H_gcn, (batch_size, num_stations, -1))  # (batch_size, num_stations, 16)
                    H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
                    H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)

                    estimated_values = tf.einsum('bnd,nd->bn', H_gcn_reshaped, H_pos_all_reshaped)  # (batch_size, num_stations)
                    estimated_values = tf.einsum('bn,d->b', estimated_values, H_pos_one)  # (batch_size,)

                    batch_true_values = true_values[batch_indices]
                    loss = loss_fn(batch_true_values, estimated_values)

                grads = tape.gradient(loss, gcn_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, gcn_model.trainable_variables))
                epoch_loss += loss.numpy()

            epoch_loss /= num_batches
            val_loss.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

            # Early stopping check
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        # Predict final values
        gcn_outputs = []
        for t in range(time_steps):
            gcn_output = gcn_model([x[t], adj])
            gcn_outputs.append(gcn_output)

        H_gcn = tf.stack(gcn_outputs, axis=0)

        H_gcn_reshaped = tf.reshape(H_gcn, (time_steps, num_stations, -1))  # (time_steps, num_stations, 16)
        H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
        H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)

        estimated_values = tf.einsum('tnd,nd->tn', H_gcn_reshaped, H_pos_all_reshaped)  # (time_steps, num_stations)
        estimated_values = tf.einsum('tn,d->t', estimated_values, H_pos_one)  # (time_steps,)

        estimated_values = estimated_values.numpy()

        # Denormalize the predictions
        # predicted_values = scaler.inverse_transform(estimated_values.reshape(-1, 1)).flatten()
        # true_values = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
        predicted_values = estimated_values

        # Calculate metrics
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predicted_values)

        results.append((target_index, rmse, mse, mae))

    return results

# Run training and prediction
results = train_and_predict(station_coords, temperature_data)

# Display results
for target_index, rmse, mse, mae in results:
    print(f'Target Index: {target_index}, RMSE: {rmse}, MSE: {mse}, MAE: {mae}')
