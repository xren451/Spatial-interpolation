import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv
from sklearn.metrics import mean_squared_error

# Generate station locations
np.random.seed(42)
num_stations = 10
station_coords = np.random.rand(num_stations, 2)

# Temperature data for one year (one data point per hour, 24 hours * 365 days)
time_steps = 24 * 365
temperature_data = np.random.rand(num_stations, time_steps) * 30  # Temperature range 0 to 30 degrees

# Generate interpolation point location and true temperature values
interpolation_point = np.random.rand(1, 2)
true_interpolation_temp = np.random.rand(time_steps) * 30  # Assume true temperature range is also 0 to 30 degrees

# Define two subgraphs
subgraph_1 = [0, 1, 2, 3, 4]
subgraph_2 = [5, 6, 7, 8, 9]


# Inverse Distance Weighting (IDW) interpolation
def idw_interpolation(station_coords, temperatures, target_point, power=2):
    distances = np.linalg.norm(station_coords - target_point, axis=1)
    weights = 1 / (distances ** power)
    weights /= weights.sum()
    interpolated_value = np.dot(weights, temperatures)
    return interpolated_value


# Apply IDW interpolation to the two subgraphs
idw_estimate_1 = np.array(
    [idw_interpolation(station_coords[subgraph_1], temperature_data[subgraph_1, t], interpolation_point) for t in
     range(time_steps)])
idw_estimate_2 = np.array(
    [idw_interpolation(station_coords[subgraph_2], temperature_data[subgraph_2, t], interpolation_point) for t in
     range(time_steps)])


# Calculate adjacency matrix based on inverse distance
def calculate_inverse_distance_adj(station_coords, decay):
    num_nodes = station_coords.shape[0]
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = np.linalg.norm(station_coords[i] - station_coords[j])
                adj[i, j] = np.exp(-decay * distance)
    return adj


# Prepare data for GCN
def prepare_data(subgraph, temperature_data, idw_estimate, decay):
    adj = calculate_inverse_distance_adj(station_coords[subgraph], decay)
    x = temperature_data[subgraph].T  # Shape (time_steps, num_stations_in_subgraph)
    y = idw_estimate.T  # Shape (time_steps,)
    return x, adj, y


# Initialize models using Spektral GCN
def initialize_spektral_model(input_shape, adj_shape):
    inputs_feat = Input(shape=(input_shape,))
    inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)

    gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
    dense = Dense(8, activation='relu')(gcn_output)
    dense_out = Dropout(0.2)(dense)
    outputs = Dense(1)(dense_out)

    model = Model(inputs=[inputs_feat, inputs_adj], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=["mae", "mape"])
    model.summary()
    return model


# Prepare data for subgraphs
x1, adj1, y1 = prepare_data(subgraph_1, temperature_data, idw_estimate_1, decay=0.1)
x2, adj2, y2 = prepare_data(subgraph_2, temperature_data, idw_estimate_2, decay=0.1)

# Convert adjacency matrices to sparse format
adj1 = tf.sparse.from_dense(adj1)
adj2 = tf.sparse.from_dense(adj2)

# Check shapes
print(f"x1 shape: {x1.shape}, adj1 shape: {adj1.shape}, y1 shape: {y1.shape}")
print(f"x2 shape: {x2.shape}, adj2 shape: {adj2.shape}, y2 shape: {y2.shape}")

# Initialize and train models
model_1 = initialize_spektral_model(x1.shape[1], adj1.shape)
model_2 = initialize_spektral_model(x2.shape[1], adj2.shape)


# Train the model
def train_model(model, x, adj, y, num_epochs=200):
    history = model.fit([x, adj], y, epochs=num_epochs, verbose=1)
    return history


train_model(model_1, x1, adj1, y1)
train_model(model_2, x2, adj2, y2)


# Predict temperature for the interpolation point
def predict(model, x, adj):
    predictions = model.predict([x, adj])
    return predictions


predicted_temp_1 = predict(model_1, x1, adj1)
predicted_temp_2 = predict(model_2, x2, adj2)


# Merge predictions from the two models
def merge_predictions(pred1, pred2):
    merged_pred = (pred1 + pred2) / 2
    return merged_pred


merged_predictions = merge_predictions(predicted_temp_1, predicted_temp_2)

# Ensure the merged predictions match the length of true interpolation temp
merged_predictions = merged_predictions.flatten()

# Compute the MSE for the merged predictions
mse = mean_squared_error(true_interpolation_temp, merged_predictions)
print(f'MSE: {mse}')
