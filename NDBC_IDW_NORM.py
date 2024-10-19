import numpy as np
import torch
import pandas as pd
from scipy.spatial import distance_matrix
import random

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

# Example usage
file_path = 'data/NDBC/all.npy'
station_value = np.load(file_path)
station_value = station_value.transpose(2, 0, 1)
station_value = station_value[:, :, 5:13]
station_value = torch.tensor(station_value)

lat_file_path = 'data/NDBC/Station_info_edit.csv'
station_info = pd.read_csv(lat_file_path, header=None).values  # Convert to NumPy array directly

station_values_tensor = torch.tensor(station_value)  # Assuming station_values is your tensor
feature_index = 3  # Specify the feature index you are interested in

# Randomly select a target station
target_index = random.choice(range(station_info.shape[0]))
target_coords = station_info[target_index, 1:3]  # Assuming the coordinates are in columns 1 and 2

# Get the true values for the selected station
true_values = station_values_tensor[target_index, :, feature_index].numpy()

# Perform IDW interpolation
interpolated_results = perform_idw_for_feature(station_info, station_values_tensor, feature_index, target_coords, target_index)

# Compute the RMSE
rmse = compute_rmse(true_values, interpolated_results)
# Compute the MAE
mae = compute_mae(true_values, interpolated_results)

print(f"Target station index: {target_index}")
print(f"True values: {true_values}")
print(f"Interpolated values: {interpolated_results}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
