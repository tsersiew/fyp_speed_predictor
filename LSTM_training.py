# %%
# Dependencies
import sys 
import pandas as pd
from scipy.io import loadmat
import glob
from keras.models import Sequential, load_model
from keras.layers import LSTM,Dropout,Dense 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


# %% [markdown]
# ## Step 1: Load the data.
# Class has 3 elements
# - An ```__init__``` function that sets up your class and all the necessary parameters.
# - An ```__len__``` function that returns the size of your dataset.
# - An ```__getitem__``` function that given an index within the limits of the size of the dataset returns the associated image and label in tensor form.

# folder names
dc_folders = ['Europe', 'Japan', 'USA']

# Implement the dataset class
class DrivingCyclesDataset(Dataset):
    def __init__(self,
                 path_to_dc,
                 train=True):
        # path_to_dc: where you put the driving cycles dataset
        # idxs_train: training set indexes
        # idxs_test: test set indexes
        # train: return training set or test set
        
        # Load all the driving cycles
        alldata = []
        dcnames = []
        if (train == True):
            mat = loadmat('./DrivingCycles/DrivingCycles/WLTPextended.mat')
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
            self.data = (np.array(alldata, dtype=object))[p] #numpy array of dataframes 
            self.names = (np.array(dcnames, dtype=object))[p]
        
        else:
            image_path = os.path.join(path_to_dc, 'test')
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

            self.data = alldata
            self.names = dcnames


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
p = np.random.permutation(int(LENDATA-1))

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
dc_path = './DrivingCycles/DrivingCycles/'
dataset_train  = DrivingCyclesDataset(dc_path, train=True)
dataset_test = DrivingCyclesDataset(dc_path, train=False)

# parameters h and f
h = 20 # length of historical sequence
f = 1 # length of forecast sequence 
step = 1

# create training set and test set 
x_train, y_train = create_dataset(dataset_train, h, f, step, False)
x_test, y_test = create_dataset(dataset_test, h, f, step, True)

# check 
print(x_train.shape)
print(x_test.shape)

# reshaping input to LSTM model 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# %% [markdown]
# ## Step 3: LSTM
# In this section we will try to make a LSTM predictor to predict the future velocity. 
# 
# ### Step 3.1: Define the model. (15 points)
# 
# Design a neural network which consists of a number of convolutional layers and a few fully connected ones at the end.
# 
# The exact architecture is up to you but you do NOT need to create something complicated. For example, you could design a LeNet insprired network.

# %%
# Network 
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) # units depends on create_dataset function
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=f))

# %% [markdown]
# ### Step 3.2: Define the training parameters. (10 points)
# 
# - Loss function
# - Optimizer
# - Learning Rate
# - Number of iterations
# - Batch Size
# - Other relevant hyperparameters

# %%
# compile the model 
model.compile(loss='mse', optimizer='adam')

# %% [markdown]
# ### Step 3.3: Train the model.

# %%
model.fit(x_train, y_train, epochs=10, batch_size=50)
model.save('speed_prediction.h5')

# load the model 
model = load_model('speed_prediction.h5') 

print('done running lstm training')
