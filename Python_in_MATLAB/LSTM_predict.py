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

def predict(x_test):
    model = load_model('speed_prediction.h5') 
    print('loaded')
    predictions = model.predict(x_test)
    return predictions

predictions = predict(x_test)

# check
print(predictions.shape)