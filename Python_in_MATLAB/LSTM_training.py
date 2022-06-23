# %% [markdown]
# ## Step 0: Importing dependencies

# %%
# Dependencies
import pandas as pd
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM,Dropout,Dense 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf


# %% [markdown]
# ## Step 1: Set up dataset functions
# Class has 3 elements
# - An ```__init__``` function that sets up your class and all the necessary parameters.
# - An ```__len__``` function that returns the size of your dataset.
# - An ```__getitem__``` function that given an index within the limits of the size of the dataset returns the associated image and label in tensor form.

# %%
# folder names
dc_folders = ['Europe', 'Japan', 'USA', 'Upsample']

# shuffling data 
LENDATA = 36 + 6 + 4 + 1 # number of driving cycles = 47
np.random.seed(42)
p = np.random.permutation(int(LENDATA-1))

# Implement the dataset class
class DrivingCyclesDataset(Dataset):
    def __init__(self,
                 path_to_dc,
                 train=True):
        # path_to_dc: where you put the driving cycles dataset
        # train: return training set or test set
        
        # Load all the driving cycles
        alldata = []
        dcnames = []
        if (train == True):
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
def create_dataset(dataset, h, f, test):
    x = [] #append the last h values
    y = [] #append the future f value 
    for df in dataset:
        features_considered = ['V_z', 'D_z']
        # features_considered = ['V_z']
        features = df[features_considered]
        for i in range(h, df.shape[0]-f):
            # for each driving cycle dataframe, have sets of (h+f) values 
            # h values are past values, f values are future value 
            features['v_ave'] = df['V_z'][i-h:i].mean()
            features['v_max'] = df['V_z'][i-h:i].max()
            features['v_min'] = df['V_z'][i-h:i].min()
            features['a_ave'] = df['D_z'][i-h:i].mean()
            features['a_max'] = df['D_z'][i-h:i].max()
            features['a_min'] = df['D_z'][i-h:i].min()
            x.append(features[i-h:i])
            if (test == False):
                y.append(df['V_z'][i:i+f])
            else:
                y.append(df['V_z'][i])
    x = np.array(x) 
    y = np.array(y)  
    return x,y

# %% [markdown]
# ## Step 2: Exploring the dataset
# 
# ### Step 2.1: Data visualisation.

# %%
# loading datasets
dc_path = './DrivingCycles/'
dataset_train  = DrivingCyclesDataset(dc_path, train=True)
dataset_test = DrivingCyclesDataset(dc_path, train=False)

# Plot 1 sample from the test set
sample_data = dataset_test.data[0]
sample_name = dataset_test.names[0]
plt.title(sample_name + " Driving Cycle")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.plot(sample_data['T_z'], sample_data['V_z'])
plt.show()

# %% [markdown]
# ### Step 2.2 Load training and test datasets
# 
# 

# parameters h and f
h = 15 # length of historical sequence
f = 5 # length of forecast sequence 

# create training set and test set 
pd.options.mode.chained_assignment = None
x_train, y_train = create_dataset(dataset_train, h, f, test=False)
x_test, y_test = create_dataset(dataset_test, h, f, test=True)

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

# %%
model = Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) # units depends on create_dataset function
model.add(Dropout(0.2))
model.add(LSTM(units=256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dropout(0.2))
model.add(Dense(units=f))

# compile the model 
model.compile(loss='mse', optimizer='adam')
# early stopping 
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10, batch_size=50, validation_split=0.2, callbacks=[early_stopping])
model.save('speed_prediction.h5')

print('done building training dataset and training the LSTM model')