import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from spektral.layers import GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import time

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Generate synthetic data
num_stations = 5
num_features =4
time_steps = 76400  # T
d_model = 16  # d_model
num_subgraphs = 3  # Number of subgraphs to generate

np.random.seed(42)
station_coords = np.random.rand(num_stations, 2)
temperature_data = np.random.rand(num_stations, time_steps, num_features) * 30  # 0-30 degree temperature data

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_data = temperature_data.reshape(-1, 1)
temperature_data = scaler.fit_transform(temperature_data)
temperature_data = temperature_data.reshape(num_stations, time_steps, num_features)

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

# Define MoE model
class MoE(tf.keras.layers.Layer):
    def __init__(self, num_experts, output_dim):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = [Dense(output_dim, activation='relu') for _ in range(num_experts)]
        self.gate = Dense(num_experts)

    def call(self, inputs):
        gate_outputs = self.gate(inputs)
        gate_weights = tf.nn.softmax(gate_outputs, axis=-1)
        # Apply each expert to the input
        expert_outputs = [self.experts[i](inputs) for i in range(self.num_experts)]
        # Combine the outputs of each expert using the gate weights
        outputs = tf.add_n([gate_weights[:, :, i:i+1] * expert_outputs[i] for i in range(self.num_experts)])
        return outputs

# Training function
def train_and_predict(station_coords, temperature_data, epochs=2, batch_size=10, learning_rate=0.1, patience=10):
    results = []
    training_times = []
    testing_times = []

    for target_index in range(num_stations):
        target_coords = station_coords[target_index]
        true_values = temperature_data[target_index].flatten()

        adj_matrix = calculate_inverse_distance_adj(station_coords)
        adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)

        gcn_models = [GCN_model(num_features, adj_matrix.shape) for _ in range(num_subgraphs)]
        mlp_all_model = MLP_model(2 * num_stations, num_stations * d_model)
        mlp_one_model = MLP_model(2, d_model)
        moe_model = MoE(num_experts=num_subgraphs, output_dim=d_model)

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
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()  # Start time for training
            epoch_loss = 0
            num_batches = time_steps // batch_size
            for batch in range(num_batches):
                with tf.GradientTape() as tape:
                    batch_indices = range(batch * batch_size, (batch + 1) * batch_size)
                    subgraph_features = []

                    for gcn_model in gcn_models:
                        gcn_outputs = []
                        for t in batch_indices:
                            gcn_output = gcn_model([x[t], adj])
                            gcn_outputs.append(gcn_output)

                        H_gcn = tf.stack(gcn_outputs, axis=0)
                        H_gcn_reshaped = tf.reshape(H_gcn, (batch_size, num_stations, -1))  # (batch_size, num_stations, 16)
                        subgraph_features.append(H_gcn_reshaped)

                    subgraph_features = tf.concat(subgraph_features, axis=-1)  # Concatenate features from subgraphs
                    aggregated_features = moe_model(subgraph_features)  # Aggregate using MoE
                    H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
                    H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)

                    estimated_values = tf.einsum('bnd,nd->bn', aggregated_features, H_pos_all_reshaped)  # (batch_size, num_stations)
                    estimated_values = tf.einsum('bn,d->b', estimated_values, H_pos_one)  # (batch_size,)

                    batch_true_values = true_values[batch_indices]
                    loss = loss_fn(batch_true_values, estimated_values)

                grads = tape.gradient(loss, gcn_model.trainable_variables + moe_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, gcn_model.trainable_variables + moe_model.trainable_variables))
                epoch_loss += loss.numpy()

            epoch_loss /= num_batches
            end_time = time.time()  # End time for training
            training_times.append(end_time - start_time)  # Record training time for the epoch
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Time: {end_time - start_time} seconds')

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
        start_time = time.time()  # Start time for testing
        subgraph_features = []

        for gcn_model in gcn_models:
            gcn_outputs = []
            for t in range(time_steps):
                gcn_output = gcn_model([x[t], adj])
                gcn_outputs.append(gcn_output)

            H_gcn = tf.stack(gcn_outputs, axis=0)
            H_gcn_reshaped = tf.reshape(H_gcn, (time_steps, num_stations, -1))  # (time_steps, num_stations, 16)
            subgraph_features.append(H_gcn_reshaped)

        subgraph_features = tf.concat(subgraph_features, axis=-1)  # Concatenate features from subgraphs
        aggregated_features = moe_model(subgraph_features)  # Aggregate using MoE
        H_pos_all_reshaped = tf.reshape(H_pos_all, (num_stations, d_model))  # (num_stations, d_model)
        H_pos_one_reshaped = tf.reshape(H_pos_one, (d_model, 1))  # (d_model, 1)

        estimated_values = tf.einsum('tnd,nd->tn', aggregated_features, H_pos_all_reshaped)  # (time_steps, num_stations)
        estimated_values = tf.einsum('tn,d->t', estimated_values, H_pos_one)  # (time_steps,)

        estimated_values = estimated_values.numpy()
        end_time = time.time()  # End time for testing
        testing_times.append(end_time - start_time)  # Record testing time
        print(f'Testing Time: {end_time - start_time} seconds')

        # Denormalize the predictions
        predicted_values = estimated_values

        # Calculate metrics
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predicted_values)

        results.append((target_index, rmse, mse, mae))

    return results, training_times, testing_times

# Run training and prediction
results, training_times, testing_times = train_and_predict(station_coords, temperature_data)

# Display results
for target_index, rmse, mse, mae in results:
    print(f'Target Index: {target_index}, RMSE: {rmse}, MSE: {mse}, MAE: {mae}')

# Display training and testing times
for epoch, train_time in enumerate(training_times):
    print(f'Epoch {epoch + 1} Training Time: {train_time} seconds')

for target_index, test_time in enumerate(testing_times):
    print(f'Target Index {target_index} Testing Time: {test_time} seconds')
