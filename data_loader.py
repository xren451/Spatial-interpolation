import argparse
from math import radians
import numpy as np
import pandas as pd
import os
import six.moves.cPickle as pickle
from sklearn.metrics.pairwise import haversine_distances
import torch
import torch.nn as nn
import torch.utils.data as data



import torch

def parse_args(args):
    parser = argparse.ArgumentParser(description="Example usage")
    parser.add_argument("--dataset", type=str, default="Shenzhen", help="Dataset name")#Can also be Tehiku,NETHERLAND
    parser.add_argument("--mask_prob", type=float, default=0.3, help="Probability of mask")  # Can also be Tehiku,NETHERLAND
    parser.add_argument("--epoch", type=int, default=100, help="Num fo epoch")  # Can also be Tehiku,NETHERLAND

    # Add more arguments if needed
    return parser.parse_args(args)

def mask_features_3d_tensor(tensor,mask_prob):
    "This function is to mask featues randomly"
    "Input: tensor,(Timesteps,stations,features) "
    "mask_prob: float"
    "Returns: Masked tensor"
    masked_tensor=np.copy(tensor)
    timesteps,stations,features=tensor.shape
    for feature in range(features):
        for station in range(stations):
            mask=np.random.rand()
            if mask<mask_prob:
                masked_tensor[:,station,feature]=0
    return masked_tensor
class ShenzhenDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def read_csv_files(self):
        # List all CSV files in the folder
        csv_files = [file for file in os.listdir(self.folder_path) if file.endswith('.csv')]

        # Read CSV files and store them in a list
        data = []
        for csv_file in csv_files:
            file_path = os.path.join(self.folder_path, csv_file)
            df = pd.read_csv(file_path)
            data.append(df.values)  # Append data as numpy array
        # Convert list of numpy arrays to tensor
        tensor_data = torch.tensor(data)
        return tensor_data

    def inverse_distance_weighting(lat, lon, power=2):
        num_points = len(lat)
        adjacency_matrix = np.zeros((num_points, num_points))

        # Convert latitude and longitude to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        coords_rad = np.column_stack((lat_rad, lon_rad))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                # Compute Haversine distance between points i and j
                distance = haversine_distances(coords_rad[[i]], coords_rad[[j]])[0, 0]

                # Apply inverse distance weighting formula
                if distance != 0:
                    weight = 1 / distance ** power
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight  # Adjacency matrix is symmetric

        return adjacency_matrix

class TehikuDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def read_excel_files(self):
        # List all Excel files in the folder
        excel_files = [file for file in os.listdir(self.folder_path) if file.endswith('.xlsx')]

        # Read Excel files and store them in a list
        data = []
        for excel_file in excel_files:
            file_path = os.path.join(self.folder_path, excel_file)
            df = pd.read_excel(file_path)  # Use read_excel instead of read_csv
            data.append(df.values)  # Append data as numpy array
        # Convert list of numpy arrays to tensor
        tensor_data = torch.tensor(data)
        return tensor_data

    def inverse_distance_weighting(lat, lon, power=2):
        num_points = len(lat)
        adjacency_matrix = np.zeros((num_points, num_points))

        # Convert latitude and longitude to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        coords_rad = np.column_stack((lat_rad, lon_rad))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                # Compute Haversine distance between points i and j
                distance = haversine_distances(coords_rad[[i]], coords_rad[[j]])[0, 0]

                # Apply inverse distance weighting formula
                if distance != 0:
                    weight = 1 / distance ** power
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight  # Adjacency matrix is symmetric

        return adjacency_matrix

class NETHERLANDDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def read_csv_files(self):
        # List all CSV files in the folder
        csv_files = [file for file in os.listdir(self.folder_path) if file.endswith('.csv')]

        # Read CSV files and store them in a list
        data = []
        for csv_file in csv_files:
            file_path = os.path.join(self.folder_path, csv_file)
            df = pd.read_csv(file_path)
            data.append(df.values)  # Append data as numpy array
        # Convert list of numpy arrays to tensor
        tensor_data = torch.tensor(data)
        return tensor_data

    def inverse_distance_weighting(lat, lon, power=2):
        num_points = len(lat)
        adjacency_matrix = np.zeros((num_points, num_points))

        # Convert latitude and longitude to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        coords_rad = np.column_stack((lat_rad, lon_rad))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                # Compute Haversine distance between points i and j
                distance = haversine_distances(coords_rad[[i]], coords_rad[[j]])[0, 0]

                # Apply inverse distance weighting formula
                if distance != 0:
                    weight = 1 / distance ** power
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight  # Adjacency matrix is symmetric

        return adjacency_matrix
# Example usage
# folder_path = 'data/Shenzhen'
# shenzhen_dataset = ShenzhenDataset(folder_path+"/Value")
# dataset_tensor = shenzhen_dataset.read_csv_files()
# dataset_tensor = np.transpose(dataset_tensor, (1, 0, 2))
# print("dataset_tensor.shape",dataset_tensor.shape)#Three dimensions: Timesteps*stations*features
# Latitude=pd.read_csv(folder_path+"/Station info.csv")['Lat']
# print("Latitude",np.array(Latitude))
# num_points = len(Latitude)
# print("num_points",num_points)
# # a=[2,3,2,3]
# # b=[4,3,2,1]
# # try_tensor=ShenzhenDataset.inverse_distance_weighting(a,b)
# # print("try_tensor:",try_tensor)
# adj_tensor=ShenzhenDataset.inverse_distance_weighting(np.array(pd.read_csv(folder_path+"/Station info.csv")['Lat']),np.array(pd.read_csv(folder_path+"/Station info.csv")['Long']))
# print("Shape of the dataset tensor:", dataset_tensor.shape)
# print("Shape of the adj_tensor:", adj_tensor.shape)