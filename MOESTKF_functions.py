import numpy as np
import torch
from sklearn.metrics import mutual_info_score


def Toy_generation(N, T, F, Prop):
    """
    Generate a random dataset for a number of stations over time with specified features.

    Parameters:
    N (int): Number of stations.
    T (int): Number of timesteps.
    F (int): Number of features.
    Prop (str): Propagation function, can be 'Gaussian', 'Random', or 'Decay'.

    Returns:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    station_value (torch.Tensor): Tensor containing the features values for each station over time.
                                  If a station does not have a feature, the value is NaN.
    """

    # Center point for the stations
    center_lat = np.random.uniform(-90, 90)
    center_lon = np.random.uniform(-180, 180)

    # Function to generate random latitudes and longitudes within 100 km
    def generate_coordinates(center_lat, center_lon, max_distance_km=100):
        R = 6371.0  # Radius of the Earth in kilometers

        # Convert max distance to radians
        max_distance_rad = max_distance_km / R

        # Generate random angle and distance within the circle
        angle = np.random.uniform(0, 2 * np.pi, N)
        distance = np.random.uniform(0, max_distance_rad, N)

        # Calculate new latitudes and longitudes
        delta_lat = distance * np.cos(angle)
        delta_lon = distance * np.sin(angle) / np.cos(np.radians(center_lat))

        new_lats = center_lat + np.degrees(delta_lat)
        new_lons = center_lon + np.degrees(delta_lon)

        return new_lats, new_lons

    latitudes, longitudes = generate_coordinates(center_lat, center_lon)
    station_names = [f'Station_{i}' for i in range(N)]

    # Combine station names, latitudes, and longitudes into a single structured array
    dtype = [('name', 'U10'), ('latitude', 'f4'), ('longitude', 'f4')]
    station_info_np = np.array(list(zip(station_names, latitudes, longitudes)), dtype=dtype)

    # Convert structured array to a regular numpy array for PyTorch compatibility
    station_info_matrix = np.zeros((N, 3), dtype=object)
    station_info_matrix[:, 0] = station_info_np['name']
    station_info_matrix[:, 1] = station_info_np['latitude']
    station_info_matrix[:, 2] = station_info_np['longitude']

    # Initialize a tensor with NaNs
    station_value = torch.full((N, T, F), float('nan'))

    for i in range(N):
        if Prop == 'Gaussian':
            values = np.random.normal(size=(T, F))
        elif Prop == 'Random':
            values = np.random.rand(T, F)
        elif Prop == 'Decay':
            values = np.exp(-np.linspace(0, 10, T).reshape(-1, 1)) * np.random.rand(T, F)
        else:
            raise ValueError("Prop should be 'Gaussian', 'Random', or 'Decay'")

        # Each station can have a different number of features
        num_features = np.random.randint(1, F + 1)
        station_value[i, :, :num_features] = torch.tensor(values[:, :num_features], dtype=torch.float32)

    return station_info_matrix, station_value


def get_complete_stations(station_info, station_value):
    """
    Get a list of stations where all features are present at every timestep.

    Parameters:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    station_value (torch.Tensor): Tensor containing the features values for each station over time.

    Returns:
    complete_stations (list): List of station names where all features are present.
    """
    complete_stations = []

    for i in range(station_value.shape[0]):
        if not torch.isnan(station_value[i]).any():
            complete_stations.append(station_info[i, 0])

    return complete_stations


def mutual_information(x, y):
    """Calculate mutual information between two vectors."""
    return mutual_info_score(x, y)


def get_complete_stations(station_info, station_value):
    """
    Get a list of stations where all features are present at every timestep.

    Parameters:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    station_value (torch.Tensor): Tensor containing the features values for each station over time.

    Returns:
    complete_stations (list): List of station names where all features are present.
    """
    complete_stations = []
    complete_indices = []

    for i in range(station_value.shape[0]):
        if not torch.isnan(station_value[i]).any():
            complete_stations.append(station_info[i, 0])
            complete_indices.append(i)

    return complete_stations, complete_indices


def Feature_wise_Subgraph(station_info, station_value, complete_stations, complete_indices, K):
    """
    Generate a feature-wise subgraph matrix.

    Parameters:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    station_value (torch.Tensor): Tensor containing the features values for each station over time.
    complete_stations (list): List of station names where all features are present.
    complete_indices (list): List of indices corresponding to complete stations.
    K (int): The number of top stations to select based on mutual information.

    Returns:
    subgraph_matrix (np.ndarray): A matrix where each cell contains a list of top K stations based on mutual information.
    """
    N_complete = len(complete_stations)
    F = station_value.shape[2]
    subgraph_matrix = np.empty((N_complete, F), dtype=object)

    for i, station_index in enumerate(complete_indices):
        for feature in range(F):
            mi_scores = []
            for j in range(station_value.shape[0]):
                if j != station_index and not torch.isnan(station_value[j, :, feature]).any():
                    mi = mutual_information(station_value[station_index, :, feature].numpy(),
                                            station_value[j, :, feature].numpy())
                    mi_scores.append((mi, j))

            mi_scores.sort(reverse=True, key=lambda x: x[0])
            top_k_stations = [station_info[j, 0] for _, j in mi_scores[:K]]
            subgraph_matrix[i, feature] = top_k_stations

    return subgraph_matrix
from scipy.stats import pearsonr

def pearson_correlation(x, y):
    """Calculate Pearson correlation between two vectors."""
    return pearsonr(x, y)[0]


def get_complete_stations(station_info, station_value):
    """
    Get a list of stations where all features are present at every timestep.

    Parameters:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    station_value (torch.Tensor): Tensor containing the features values for each station over time.

    Returns:
    complete_stations (list): List of station names where all features are present.
    """
    complete_stations = []
    complete_indices = []

    for i in range(station_value.shape[0]):
        if not torch.isnan(station_value[i]).any():
            complete_stations.append(station_info[i, 0])
            complete_indices.append(i)

    return complete_stations, complete_indices


def Feature_wise_Subgraph(station_info, station_value, complete_stations, complete_indices, K):
    """
    Generate a feature-wise subgraph matrix.

    Parameters:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    station_value (torch.Tensor): Tensor containing the features values for each station over time.
    complete_stations (list): List of station names where all features are present.
    complete_indices (list): List of indices corresponding to complete stations.
    K (int): The number of top stations to select based on Pearson correlation.

    Returns:
    subgraph_matrix (np.ndarray): A matrix where each cell contains a list of top K stations based on Pearson correlation.
    """
    N_complete = len(complete_stations)
    F = station_value.shape[2]
    subgraph_matrix = np.empty((N_complete, F), dtype=object)

    for i, station_index in enumerate(complete_indices):
        for feature in range(F):
            correlation_scores = []
            for j in range(station_value.shape[0]):
                if j != station_index and not torch.isnan(station_value[j, :, feature]).any():
                    correlation = pearson_correlation(station_value[station_index, :, feature].numpy(),
                                                      station_value[j, :, feature].numpy())
                    correlation_scores.append((correlation, j))

            correlation_scores.sort(reverse=True, key=lambda x: x[0])
            top_k_stations = [station_info[j, 0] for _, j in correlation_scores[:K]]
            subgraph_matrix[i, feature] = top_k_stations

    return subgraph_matrix

import matplotlib.pyplot as plt

def visualize_complete_stations_and_subgraph(station_info, subgraph_matrix, complete_stations, complete_indices):
    """
    Visualize the complete stations, their corresponding subgraph, and all other stations.

    Parameters:
    station_info (torch.Tensor): Tensor containing station names, latitudes, and longitudes.
    subgraph_matrix (np.ndarray): A matrix where each cell contains a list of top K stations based on Pearson correlation.
    complete_stations (list): List of station names where all features are present.
    complete_indices (list): List of indices corresponding to complete stations.
    """
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
    all_latitudes = station_info[:, 1].astype(float)
    all_longitudes = station_info[:, 2].astype(float)

    for i, complete_station_index in enumerate(complete_indices):
        plt.figure(figsize=(12, 8))

        # Plot all stations in gray
        plt.scatter(all_longitudes, all_latitudes, c='gray', label='Other Stations', alpha=0.5)

        # Plot the current complete station in red
        plt.scatter(all_longitudes[complete_station_index], all_latitudes[complete_station_index],
                    c='red', label=f'{complete_stations[i]} (Complete)', edgecolors='black', s=100)

        # Plot subgraph stations in different colors for each feature
        for feature in range(subgraph_matrix.shape[1]):
            subgraph_station_names = subgraph_matrix[i, feature]
            subgraph_indices = [j for j, name in enumerate(station_info[:, 0]) if name in subgraph_station_names]
            color = colors[feature % len(colors)]
            plt.scatter(all_longitudes[subgraph_indices], all_latitudes[subgraph_indices],
                        c=color, label=f'Feature {feature + 1}', alpha=0.7)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Visualization of Complete Station: {complete_stations[i]} and its Subgraph')
        plt.legend()
        plt.show()
# Geometric Construction and Correlation Visualization

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def midpoint(p1, p2):
    return (p1 + p2) / 2

def subdivide_triangle(A, B, C, min_length):
    triangles = []
    stack = [(A, B, C)]
    while stack:
        A, B, C = stack.pop()
        AB, BC, CA = distance(A, B), distance(B, C), distance(C, A)
        if AB < min_length and BC < min_length and CA < min_length:
            triangles.append((A, B, C))
            continue
        mid_AB, mid_BC, mid_CA = midpoint(A, B), midpoint(B, C), midpoint(C, A)
        stack.extend([(A, mid_AB, mid_CA), (mid_AB, B, mid_BC), (mid_CA, mid_BC, C), (mid_AB, mid_BC, mid_CA)])
    return triangles
def sample_points_in_triangle(A, B, C, N):
    points = []
    for _ in range(N):
        r1, r2 = np.random.rand(), np.random.rand()
        sqrt_r1 = np.sqrt(r1)
        point = (1 - sqrt_r1) * A + (sqrt_r1 * (1 - r2)) * B + (sqrt_r1 * r2) * C
        points.append(point)
    return np.array(points)

def correlation(distance, start_value, end_value, max_distance, lambda_param=1.0):
    normalized_distance = (distance / max_distance) ** lambda_param
    return start_value - (start_value - end_value) * normalized_distance


def calculate_correlations(points, A, B, C):
    lambda_param = 1.0
    correlations = []
    for point in points:
        dist_to_A, dist_to_B, dist_to_C = np.linalg.norm(point - A), np.linalg.norm(point - B), np.linalg.norm(point - C)
        max_dist = max(dist_to_A, dist_to_B, dist_to_C)
        corr_A = correlation(dist_to_A, 1, 0.8 if dist_to_B < dist_to_C else 0.6, max_dist, lambda_param)
        corr_B = correlation(dist_to_B, 1, 0.8 if dist_to_A < dist_to_C else 0.7, max_dist, lambda_param)
        corr_C = correlation(dist_to_C, 1, 0.7 if dist_to_B < dist_to_A else 0.6, max_dist, lambda_param)
        correlations.append([corr_A, corr_B, corr_C])
    return np.array(correlations)

def is_point_inside_triangle(point, triangle):
    A, B, C = triangle
    sign_1 = np.cross(B - A, point - A) < 0.0
    sign_2 = np.cross(C - B, point - B) < 0.0
    sign_3 = np.cross(A - C, point - C) < 0.0
    return (sign_1 == sign_2) and (sign_2 == sign_3)

def determine_points_in_triangles(triangles, sampled_points):
    points_in_triangles = []
    for triangle in triangles:
        points_in_triangles.append([point for point in sampled_points if is_point_inside_triangle(point, triangle)])
    return points_in_triangles

def calculate_average_properties_in_triangles(points_in_triangles, correlations):
    avg_properties = []
    for points in points_in_triangles:
        if len(points) == 0:
            avg_properties.append([0, 0, 0])
        else:
            avg_properties.append(np.mean([correlations[np.where((sampled_points == point).all(axis=1))[0][0]] for point in points], axis=0))
    return np.array(avg_properties)
import numpy as np
import torch
from shapely.geometry import Polygon
def plot_triangle_centroids(triangles, avg_properties):
    centroid_sums = np.sum(avg_properties, axis=1)
    for i, triangle in enumerate(triangles):
        polygon = Polygon(triangle, closed=True, facecolor=plt.cm.copper(centroid_sums[i] / np.max(centroid_sums)))
        plt.gca().add_patch(polygon)
        centroid = np.mean(triangle, axis=0)
        plt.scatter(centroid[0], centroid[1], c='blue', marker='o', s=50)
def distribution_gene(complete_stations, subgraph_matrix, station_info, station_value):
    # Convert station_info to a dictionary for easy access
    station_coords = {row[0]: (row[1], row[2]) for row in station_info}

    subgraph_polygon = np.empty_like(subgraph_matrix, dtype=object)
    polygon_attri = np.empty_like(subgraph_matrix, dtype=object)

    for i, station in enumerate(complete_stations):
        for j in range(subgraph_matrix.shape[1]):
            # Get the list of related stations from subgraph_matrix
            related_stations = subgraph_matrix[i, j]

            # Include the current station
            polygon_stations = [station] + list(related_stations)

            # Get the coordinates of the stations
            coords = [station_coords[stat] for stat in polygon_stations]

            # Create a polygon
            polygon = Polygon(coords)
            subgraph_polygon[i, j] = polygon

            # Calculate polygon attributes (e.g., area, perimeter)
            attri = {
                'area': polygon.area,
                'perimeter': polygon.length
            }
            polygon_attri[i, j] = attri

    return subgraph_polygon, polygon_attri


import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import matplotlib.colors as mcolors


# Convex hull function
def convex_hull(points):
    if len(points) < 3:
        return points  # Convex hull is not defined for fewer than 3 points
    hull = ConvexHull(points)
    return points[hull.vertices]

# Create polygon matrix
def create_polygon_matrix(complete_sub_matrix, station_coords):
    complete_sub_polygon = np.empty((complete_sub_matrix.shape[0], complete_sub_matrix.shape[1]), dtype=object)
    for i in range(complete_sub_matrix.shape[0]):
        for j in range(complete_sub_matrix.shape[1]):
            stations = complete_sub_matrix[i, j]
            coords = np.array([station_coords[station] for station in stations if station in station_coords])
            if len(coords) > 2:
                hull_coords = convex_hull(coords)
                complete_sub_polygon[i, j] = Polygon(hull_coords)
            else:
                complete_sub_polygon[i, j] = Polygon(coords)
    return complete_sub_polygon

# Compute adjacency matrices with distance decay
def compute_adjacency_matrix_with_decay(complete_sub_matrix, station_value, station_info, alpha=0.01):
    num_king_stations = complete_sub_matrix.shape[0]
    num_features = complete_sub_matrix.shape[1]
    num_sub_stations = complete_sub_matrix.shape[2]

    adjacency_matrices = [[None for _ in range(num_features)] for _ in range(num_king_stations)]

    # Create a dictionary to map station names to their indices in station_value
    station_indices = {station_info[i][0]: i for i in range(len(station_info))}

    def decay_function(distance, alpha=0.1):
        return np.exp(-alpha * distance**2)

    for i in range(num_king_stations):
        for j in range(num_features):
            sub_station_names = complete_sub_matrix[i, j]
            sub_station_indices = [station_indices[station] for station in sub_station_names]

            # Extract the corresponding features from station_value for the current feature j
            extracted_features = station_value[sub_station_indices, :, j].numpy().T

            # Compute the Pearson correlation matrix
            df = pd.DataFrame(extracted_features)
            corr_matrix = df.corr().values
            station_coords = {row[0]: (float(row[1]), float(row[2])) for row in station_info}
            # Adjust the correlation matrix using the decay function based on distances
            coords = np.array([station_coords[station] for station in sub_station_names])
            distances = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
            decay_values = decay_function(distances, alpha)

            corr_matrix *= decay_values

            adjacency_matrices[i][j] = corr_matrix

    return adjacency_matrices

# Generate random 6x6 matrices for each fine-grained polygon
def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(matrix)
    normalized = (matrix - min_val) / (max_val - min_val)
    return normalized

def generate_adj_matrix_for_polygons(fine_grained_polygons):
    polygon_matrices = np.empty_like(fine_grained_polygons, dtype=object)
    for i in range(fine_grained_polygons.shape[0]):
        for j in range(fine_grained_polygons.shape[1]):
            fine_polygons = fine_grained_polygons[i, j]
            if fine_polygons:
                matrices = []
                for _ in fine_polygons:
                    matrix = np.random.rand(6, 6)
                    np.fill_diagonal(matrix, 1)
                    matrices.append(matrix)
                polygon_matrices[i, j] = matrices
    return polygon_matrices

def idw_interpolation(target_coord, known_coords, known_values, power=2):
    distances = distance_matrix([target_coord], known_coords)[0]
    # Handle case where distance is zero (i.e., target_coord is exactly at a known_coord)
    if np.any(distances == 0):
        return known_values[np.argmin(distances)]
    weights = 1 / (distances ** power)
    weights /= weights.sum()  # Normalize weights
    interpolated_value = np.dot(weights, known_values)
    return interpolated_value

def perform_idw_for_feature(station_info, station_values, feature_index, target_coords, exclude_index):
    feature_values = station_values[:, :, feature_index]  # Extract specific feature values
    interpolated_values = []

    for t in range(feature_values.shape[1]):  # Iterate over timesteps
        known_indices = ~torch.isnan(feature_values[:, t])  # Identify stations with known values
        known_indices[exclude_index] = False  # Exclude the target station

        if known_indices.sum() == 0:
            print(f"No known values for timestep {t}")
            interpolated_values.append(np.nan)  # If no known values, return NaN
            continue

        known_coords = station_info[known_indices.numpy(), 1:3]  # Coordinates of known stations (2D)
        known_values = feature_values[known_indices, t].numpy()  # Known feature values

        # Ensure known_coords is a NumPy array of floats
        known_coords = known_coords.astype(float)
        target_coords = target_coords.astype(float)

        # Normalise known_coords and target_coords
        mean_coords = known_coords.mean(axis=0)
        std_coords = known_coords.std(axis=0)
        norm_known_coords = (known_coords - mean_coords) / std_coords
        norm_target_coords = (target_coords - mean_coords) / std_coords

        print(f"Known coordinates for timestep {t}: {known_coords}")
        print(f"Known values for timestep {t}: {known_values}")

        interpolated_value = idw_interpolation(norm_target_coords, norm_known_coords, known_values)
        interpolated_values.append(interpolated_value)

    return np.array(interpolated_values)

def compute_rmse(true_values, predicted_values):
    return np.sqrt(np.nanmean((true_values - predicted_values) ** 2))

def compute_mae(true_values, predicted_values):
    return np.nanmean(np.abs(true_values - predicted_values))