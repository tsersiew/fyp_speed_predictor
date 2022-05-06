import sys 
import pandas as pd
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM,Dropout,Dense 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

print('runnning python on matlab')

# %%
# folder names
dc_folders = ['Europe', 'Japan', 'USA']

# Implement the dataset class
class DrivingCyclesDataset(Dataset):
    def __init__(self,
                 path_to_dc,
                 idxs_train,
                 idxs_test,
                 train=True):
        # path_to_dc: where you put the driving cycles dataset
        # idxs_train: training set indexes
        # idxs_test: test set indexes
        # train: return training set or test set
        
        # Load all the driving cycles
        alldata = []
        dcnames = []
        mat = loadmat('./DrivingCycles/WLTPextended.mat')
        df = pd.DataFrame(mat['V_z'], columns = ['V_z']) # velocity 
        df2 = pd.DataFrame(mat['T_z'], columns = ['T_z']) # time
        df3 = pd.DataFrame(mat['D_z'], columns = ['D_z']) # acceleration
        df = pd.concat([df, df2, df3], axis=1)
        alldata.append(df)
        dcnames.append('WLTPextended.mat')
        for folder in dc_folders:
            image_path = os.path.join(path_to_dc, folder)
            files = glob.glob(image_path + '/*.mat')
            for f in files:
                mat = loadmat(f)
                df = pd.DataFrame(mat['V_z'], columns = ['V_z'])
                df2 = pd.DataFrame(mat['T_z'], columns = ['T_z'])
                df3 = pd.DataFrame(mat['D_z'], columns = ['D_z'])
                df = pd.concat([df, df2, df3], axis=1)
                dcnames.append(os.path.basename(f))
                # each dataframe is a driving cycle 
                alldata.append(df)

        # Extract the driving cycles with the specified file indexes     
        alldata_np = (np.array(alldata, dtype=object))[p] #numpy array of dataframes 
        dcnames_np = (np.array(dcnames, dtype=object))[p]
        if train==True:
            self.data = np.concatenate([alldata_np[:idxs_test], alldata_np[idxs_test+1:]])
            self.names = np.concatenate([dcnames_np[:idxs_test], dcnames_np[idxs_test+1:]])
        else:
            self.data = alldata_np[idxs_test:idxs_test+1]
            self.names = dcnames_np[idxs_test:idxs_test+1]


    def __len__(self, idx):
        # Return the number of samples in a driving cycle 
        return (self.data[idx]).size
        
    def __getitem__(self, idx):
        # Get an item using its index
        # Return the driving cycle and its name 
        return self.data[idx]

# %%
def split_train_test(lendata, percentage=0.8):
    idxs_train = int(percentage * lendata)
    idxs_test = idxs_train + 1 
    return idxs_train, idxs_test

LENDATA = 36 + 6 + 4 + 1 # number of driving cycles = 47
np.random.seed(42)
idxs_train, idxs_test = split_train_test(LENDATA,0.8)
idxs_test = 22 # only 1 test driving cycle for easier visualisation
p = np.random.permutation(int(LENDATA))

# %%
def create_dataset(dataset, h, f, step, test):
    x = [] #append the last 50 values
    y = [] #append the future value 
    for df in dataset:
        features_considered = ['V_z', 'D_z']
        features = df[features_considered]
        for i in range(h, df.shape[0]-f):
            # for each driving cycle dataframe, have sets of 51 values 
            # h values are past values, f values are future value 
            features['v_ave'] = df['V_z'][i-h:i].mean()
            features['v_max'] = df['V_z'][i-h:i].max()
            features['v_min'] = df['V_z'][i-h:i].min()
            features['a_ave'] = df['D_z'][i-h:i].mean()
            x.append(features[i-h:i])
            if (test == False):
                y.append(df['V_z'][i:i+f])
            else:
                y.append(df['V_z'][i])
            # y.append(df.loc[i:i+f, "V_z"])
    x = np.array(x) 
    y = np.array(y)  
    # x = np.asarray(x).astype(np.float32)
    return x,y

# %% [markdown]
# ## Step 2: Explore the data. (15 Points)
# 
# ### Step 2.1: Data visualisation. (5 points)
# 
# - Plot data distribution, i.e. the number of samples per class.
# - Plot 1 sample from each of the five classes in the training set.

# %%
# loading datasets
dc_path = './DrivingCycles/DrivingCycles'
dataset_train  = DrivingCyclesDataset(dc_path, idxs_train, idxs_test, train=True)
dataset_test = DrivingCyclesDataset(dc_path, idxs_train, idxs_test, train=False)

# # Plot samples from the test set
# # Europe NEDC 
# sample_data = dataset_train.data[26]
# sample_name = dataset_train.names[26]
# print(sample_data.describe())
# plt.title(sample_name + " Driving Cycle")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.plot(sample_data['T_z'], sample_data['V_z'])
# plt.show()

# # Europe WLTP 
# sample_data = dataset_train.data[37]
# sample_name = dataset_train.names[37]
# print(sample_data.describe())
# plt.title(sample_name + " Driving Cycle")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.plot(sample_data['T_z'], sample_data['V_z'])
# plt.show()

# # Japan 10 Mode 
# sample_data = dataset_train.data[30]
# sample_name = dataset_train.names[30]
# print(sample_data.describe())
# plt.title(sample_name + " Driving Cycle")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.plot(sample_data['T_z'], sample_data['V_z'])
# plt.show()

# # US FTP_75 
# sample_data = dataset_train.data[17]
# sample_name = dataset_train.names[17]
# print(sample_data.describe())
# plt.title(sample_name + " Driving Cycle")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.plot(sample_data['T_z'], sample_data['V_z'])
# plt.show()

print('can run plotting on matlab')
