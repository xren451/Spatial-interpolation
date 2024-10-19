import numpy as np
from MOESTKF_functions import Toy_generation, get_complete_stations, Feature_wise_Subgraph
import tensorflow as tf
from scipy.spatial import ConvexHull
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import time
import random
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import time
import random
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Load datasets
station_value_path = 'data/NDBC/all.npy'
station_info_path = 'data/NDBC/Station_info_edit.csv'

station_value = np.load(station_value_path)
station_value = station_value.transpose(2, 0, 1)[:, :, 5:13]  # Select relevant features
station_info = pd.read_csv(station_info_path, header=None).values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
station_value = station_value.reshape(-1, station_value.shape[-1])
station_value = scaler.fit_transform(station_value)
station_value = station_value.reshape(103, 8784, 8)

# Convert station_value to torch tensor
station_value = torch.tensor(station_value)


# Utility functions
def idw_interpolation(station_coords, temperatures, target_point, power=2):
    distances = np.linalg.norm(station_coords - target_point, axis=1)
    weights = 1 / (distances ** power)
    weights /= weights.sum()
    interpolated_value = np.dot(weights, temperatures)
    return interpolated_value


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


def GCN_model(input_shape, adj_shape):
    inputs_feat = Input(shape=(input_shape,))
    inputs_adj = Input(shape=(adj_shape[0], adj_shape[1]), sparse=True)
    gcn_output = GCNConv(16, activation='relu')([inputs_feat, inputs_adj])
    model = Model(inputs=[inputs_feat, inputs_adj], outputs=gcn_output)
    return model


def MLP_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(output_dim, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def convex_hull(points):
    if len(points) < 3:
        return points  # Convex hull is not defined for fewer than 3 points
    hull = ConvexHull(points)
    return [points[vertex] for vertex in hull.vertices]


def create_polygon_matrix(complete_sub_matrix, station_coords):
    complete_sub_polygon = np.empty((complete_sub_matrix.shape[0], complete_sub_matrix.shape[1]), dtype=object)
    for i in range(complete_sub_matrix.shape[0]):
        for j in range(complete_sub_matrix.shape[1]):
            stations = complete_sub_matrix[i, j]
            coords = [station_coords[station] for station in stations if station in station_coords]
            if len(coords) > 2:
                hull_coords = convex_hull(coords)
                complete_sub_polygon[i, j] = Polygon(hull_coords)
            else:
                complete_sub_polygon[i, j] = Polygon(coords)
    return complete_sub_polygon


def plot_polygon(polygon):
    fig, ax = plt.subplots()
    if polygon.is_valid:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'r-', lw=2)
        ax.fill(x, y, 'red', alpha=0.1)
        ax.scatter(x, y, color='red')
    ax.set_title('Polygon Visualization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Polygon', 'Vertices'])
    plt.show()


def subdivide_polygon(polygon, num_subdivisions):
    minx, miny, maxx, maxy = polygon.bounds
    width = (maxx - minx) / num_subdivisions
    height = (maxy - miny) / num_subdivisions
    sub_polygons = []
    for i in range(num_subdivisions):
        for j in range(num_subdivisions):
            sub_poly = Polygon([
                (minx + i * width, miny + j * height),
                (minx + (i + 1) * width, miny + j * height),
                (minx + (i + 1) * width, miny + (j + 1) * height),
                (minx + i * width, miny + (j + 1) * height)
            ])
            if sub_poly.intersects(polygon):
                sub_polygons.append(sub_poly.intersection(polygon))
    return sub_polygons


def create_fine_grained_polygons(Complete_Sub_polygon, num_subdivisions):
    fine_grained_polygons = np.empty_like(Complete_Sub_polygon, dtype=object)
    for i in range(Complete_Sub_polygon.shape[0]):
        for j in range(Complete_Sub_polygon.shape[1]):
            coarse_polygon = Complete_Sub_polygon[i, j]
            if coarse_polygon.is_valid:
                fine_polygons = subdivide_polygon(coarse_polygon, num_subdivisions)
                fine_grained_polygons[i, j] = fine_polygons
    return fine_grained_polygons


def plot_fine_polygons(fine_polygons):
    fig, ax = plt.subplots()
    for poly in fine_polygons:
        if poly.is_valid:
            x, y = poly.exterior.xy
            ax.plot(x, y, 'b-', lw=1)
            ax.fill(x, y, 'blue', alpha=0.1)
            ax.scatter(x, y, color='blue')
    ax.set_title('Fine-Grained Polygons Visualization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Polygon', 'Vertices'])
    plt.show()


def generate_adj_matrix_for_polygons(fine_grained_polygons, K):  # K represents the number of neighbours
    polygon_matrices = np.empty_like(fine_grained_polygons, dtype=object)
    for i in range(fine_grained_polygons.shape[0]):
        for j in range(fine_grained_polygons.shape[1]):
            fine_polygons = fine_grained_polygons[i, j]
            if fine_polygons:
                matrices = []
                for _ in fine_polygons:
                    matrix = np.random.rand(K + 1 + 1, K + 1 + 1)
                    np.fill_diagonal(matrix, 1)
                    matrix = (matrix + matrix.T) / 2  # Make the matrix symmetric
                    matrices.append(matrix)
                polygon_matrices[i, j] = matrices
    return polygon_matrices


def compute_weighted_correlation(station_value, coords, target_coords, den_fac, feature_index):
    correlations = []
    for coord in coords:
        distance = euclidean(coord, target_coords)
        weight = np.exp(-distance) ** den_fac
        correlation, _ = pearsonr(station_value, station_value)  # Using station_value correctly
        weighted_corr = correlation * weight
        correlations.append(weighted_corr)
    return correlations

#
# Convert station_info to a dictionary for easy lookup
station_coords_dict = {row[0]: (float(row[1]), float(row[2])) for row in station_info}
station_coords = np.array(list(station_coords_dict.values()))  # Convert to numpy array for vectorized operations

# Generate complete stations and subgraphs
complete_stations, complete_indices = get_complete_stations(station_info, station_value)
K = 5
subgraph_matrix = Feature_wise_Subgraph(station_info, station_value, complete_stations, complete_indices, K)

# Initialize the new matrix with the desired shape
complete_sub_matrix = np.empty((subgraph_matrix.shape[0], subgraph_matrix.shape[1], 6), dtype=object)

# Fill the new matrix
for i, complete_station in enumerate(complete_stations):
    for j in range(subgraph_matrix.shape[1]):
        complete_sub_matrix[i, j] = np.insert(subgraph_matrix[i, j], 0, complete_station)

# Generate the Complete_Sub_polygon matrix
Complete_Sub_polygon = create_polygon_matrix(complete_sub_matrix, station_coords_dict)

# Generate the fine-grained polygons
num_subdivisions = 5  # Adjust this as needed for finer or coarser subdivisions
fine_grained_polygons = create_fine_grained_polygons(Complete_Sub_polygon, num_subdivisions)

# Generate the adjacency matrices for the fine-grained polygons
polygon_matrices = generate_adj_matrix_for_polygons(fine_grained_polygons, K)

# Select a random target node and feature for interpolation
target_index = random.choice(range(station_info.shape[0]))
target_coords = station_info[target_index, 1:3].astype(float)  # Ensure target_coords is of float type
target_feature = random.choice(range(station_value.shape[2]))
print(f'Selected target node: {target_index}, target feature: {target_feature}')

# IDW interpolation for the target node and feature
idw_estimate = np.array(
    [idw_interpolation(station_coords, station_value[:, t, target_feature].numpy().flatten(), target_coords) for t in
     range(8784)]
)

# Prepare data for GCN
adj_matrix = calculate_inverse_distance_adj(station_info[:, 1:3].astype(float))
adj_matrix_sparse = tf.sparse.from_dense(adj_matrix)
x = station_value[:, :, target_feature].numpy().transpose(1, 0)  # (time_steps, num_stations, 1)

# Ensure the input to the GCN is 2D
x = x.reshape(x.shape[0], x.shape[1], 1)

# Define and train the GCN model
gcn_model = GCN_model(x.shape[1], adj_matrix.shape)
optimizer = Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training loop
epochs = 100
batch_size = 10
start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0
    num_batches = x.shape[0] // batch_size
    for batch in range(num_batches):
        with tf.GradientTape() as tape:
            batch_indices = range(batch * batch_size, (batch + 1) * batch_size)
            gcn_outputs = []
            for t in batch_indices:
                gcn_output = gcn_model([x[t], adj_matrix_sparse])
                gcn_outputs.append(gcn_output)

            H_gcn = tf.stack(gcn_outputs, axis=0)
            H_gcn = tf.reshape(H_gcn, (batch_size, station_info.shape[0], -1))  # (batch_size, num_stations, 16)

            # MoE aggregation
            H_pos_all = np.random.rand(station_info.shape[0], 16)  # Dummy positional encoding for stations
            H_pos_one = np.random.rand(16)  # Dummy positional encoding for target station
            estimated_values = tf.einsum('bnd,nd->bn', H_gcn, H_pos_all)  # (batch_size, num_stations)
            estimated_values = tf.einsum('bn,d->b', estimated_values, H_pos_one)  # (batch_size,)

            batch_true_values = idw_estimate[batch_indices]
            loss = loss_fn(batch_true_values, estimated_values)

        grads = tape.gradient(loss, gcn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, gcn_model.trainable_variables))
        epoch_loss += loss.numpy()

    epoch_loss /= num_batches
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

training_time = time.time() - start_time

# Predict final values
gcn_outputs = []
for t in range(x.shape[0]):
    gcn_output = gcn_model([x[t], adj_matrix_sparse])
    gcn_outputs.append(gcn_output)

H_gcn = tf.stack(gcn_outputs, axis=0)
H_gcn = tf.reshape(H_gcn, (x.shape[0], station_info.shape[0], -1))  # (time_steps, num_stations, 16)

# MoE aggregation for predictions
H_pos_all = np.random.rand(station_info.shape[0], 16)  # Dummy positional encoding for stations
H_pos_one = np.random.rand(16)  # Dummy positional encoding for target station
estimated_values = tf.einsum('tnd,nd->tn', H_gcn, H_pos_all)  # (time_steps, num_stations)
estimated_values = tf.einsum('tn,d->t', estimated_values, H_pos_one)  # (time_steps,)

# Convert Tensor to NumPy array before reshaping
estimated_values = estimated_values.numpy()

# Denormalize the predictions
estimated_values = scaler.inverse_transform(estimated_values.reshape(-1, 1)).flatten()
idw_estimate = scaler.inverse_transform(idw_estimate.reshape(-1, 1)).flatten()

# Compute MAE and RMSE
mae = mean_absolute_error(idw_estimate, estimated_values)
rmse = mean_squared_error(idw_estimate, estimated_values, squared=False)
print(f'MAE: {mae}, RMSE: {rmse}, Training Time: {training_time} seconds')

