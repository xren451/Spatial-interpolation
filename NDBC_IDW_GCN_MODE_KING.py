from MOESTKF_functions import Toy_generation, get_complete_stations, Feature_wise_Subgraph
import os
import numpy as np
import six.moves.cPickle as pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import pandas as pd
import torch
import random
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
import random
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np
from shapely.ops import unary_union
from scipy.stats import pearsonr

##############################
#1. Input: Toy example generation #Can be replaced by other datasets here.
##############################
# Example usage

file_path = 'data/NDBC/all.npy'
station_value = np.load(file_path)
station_value = station_value.transpose(2, 0, 1)
station_value = station_value[:, :, 5:13]
station_value = torch.tensor(station_value)

lat_file_path = 'data/NDBC/Station_info_edit.csv'
station_info = pd.read_csv(lat_file_path, header=None).values  # Convert to NumPy array directly

station_value= torch.tensor(station_value)  # Assuming station_values is your tensor
print("station_value.shape:",station_value.shape)
complete_stations, complete_indices = get_complete_stations(station_info, station_value)
K = 5
subgraph_matrix = Feature_wise_Subgraph(station_info, station_value, complete_stations, complete_indices, K)#Subgraph Matrix: (16, 8)

# print("Feature-wise Subgraph Matrix:")
# print(subgraph_matrix.shape)#Feature-wise Subgraph Matrix: (16, 8)

# Initialize the new matrix with the desired shape
complete_sub_matrix = np.empty((subgraph_matrix.shape[0], subgraph_matrix.shape[1], 6), dtype=object)

# Fill the new matrix
for i, complete_station in enumerate(complete_stations):
    for j in range(subgraph_matrix.shape[1]):
        complete_sub_matrix[i, j] = np.insert(subgraph_matrix[i, j], 0, complete_station)

# Example output for verification
print("Complete_Sub_matrix.shape",complete_sub_matrix.shape)


# Convert station_info to a dictionary for easy lookup
station_coords = {row[0]: (float(row[1]), float(row[2])) for row in station_info}


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


# Generate the Complete_Sub_polygon matrix
Complete_Sub_polygon = create_polygon_matrix(complete_sub_matrix, station_coords)

# Example usage: plot the first polygon in the matrix
polygon = Complete_Sub_polygon[0, 0]


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


# Plot the first polygon
plot_polygon(polygon)


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


# Generate the fine-grained polygons
num_subdivisions = 5  # Adjust this as needed for finer or coarser subdivisions
fine_grained_polygons = create_fine_grained_polygons(Complete_Sub_polygon, num_subdivisions)

# Example usage: plot the fine-grained polygons and display adjacency matrix for the first coarse polygon
fine_polygons = fine_grained_polygons[0, 0]


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


# Plot the fine-grained polygons
plot_fine_polygons(fine_polygons)


def generate_adj_matrix_for_polygons(fine_grained_polygons,K):#K represents the number of neighbors
    polygon_matrices = np.empty_like(fine_grained_polygons, dtype=object)
    for i in range(fine_grained_polygons.shape[0]):
        for j in range(fine_grained_polygons.shape[1]):
            fine_polygons = fine_grained_polygons[i, j]
            if fine_polygons:
                matrices = []
                for _ in fine_polygons:
                    matrix = np.random.rand(K+1+1, K+1+1)
                    np.fill_diagonal(matrix, 1)
                    matrix = (matrix + matrix.T) / 2  # Make the matrix symmetric
                    matrices.append(matrix)
                polygon_matrices[i, j] = matrices
    return polygon_matrices

def generate_rand_adj_matrix_for_polygons_(fine_grained_polygons,K):#K represents the number of neighbors
    polygon_matrices = np.empty_like(fine_grained_polygons, dtype=object)
    for i in range(fine_grained_polygons.shape[0]):
        for j in range(fine_grained_polygons.shape[1]):
            fine_polygons = fine_grained_polygons[i, j]
            if fine_polygons:
                matrices = []
                for _ in fine_polygons:
                    matrix = np.random.rand(K+1+1, K+1+1)
                    np.fill_diagonal(matrix, 1)
                    matrix = (matrix + matrix.T) / 2  # Make the matrix symmetric
                    matrices.append(matrix)
                polygon_matrices[i, j] = matrices
    return polygon_matrices



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

def get_centroid(polygon):
    centroid = polygon.centroid
    return (centroid.x, centroid.y)


def compute_weighted_correlation(station_value, coords, target_coords, den_fac, feature_index):
    correlations = []
    for coord in coords:
        distance = euclidean(coord, target_coords)
        weight = np.exp(-distance) ** den_fac
        correlation, _ = pearsonr(station_value, station_value)  # Using station_value correctly
        weighted_corr = correlation * weight
        correlations.append(weighted_corr)
    return correlations

station_indices_lookup = {str(station[0]).replace('Station_', ''): idx for idx, station in enumerate(station_info)}

def find_station_index_np(station_info, station_id):
    return np.where(station_info[:, 0] == station_id)[0][0]  # Using [0][0] to extract the first match's index

def generate_try_adj_matrix_for_polygons(fine_grained_polygons,K,station_info,station_value):#K represents the number of neighbors
#def generate_try_adj_matrix_for_polygons(fine_grained_polygons,K,station_info,station_value, lambda):#K represents the number of neighbors
    polygon_matrices = np.empty_like(fine_grained_polygons, dtype=object)
    for i in range(fine_grained_polygons.shape[0]):
        for j in range(fine_grained_polygons.shape[1]):
            fine_polygons = fine_grained_polygons[i, j]
            if fine_polygons:
                matrices = []
                for _ in fine_polygons:
                    matrix = np.random.rand(K+1+1, K+1+1)
                    np.fill_diagonal(matrix, 1)
                    matrix = (matrix + matrix.T) / 2  # Make the matrix symmetric
                    matrices.append(matrix)
                polygon_matrices[i, j] = matrices
    #Loop for all elements except for the unknown potential stations.
    for i in range(polygon_matrices.shape[0]-1):
        for j in range(polygon_matrices.shape[1]-1):
            for k in range (len(polygon_matrices[i, j])):
                for l in range (K+1):
                    for m in range(K+1):
                        polygon_matrices[i, j][k][l,m],_=pearsonr(station_value[np.where(station_info[:, 0] == complete_sub_matrix[i,j,l])[0][0],:,j],station_value[np.where(station_info[:, 0] == complete_sub_matrix[i,j,m])[0][0],:,j])
    return polygon_matrices


polygon_matrices = generate_try_adj_matrix_for_polygons(fine_grained_polygons,K,station_info,station_value)
print("polygon_matrices[0,0][0]",polygon_matrices[0,0][0])


