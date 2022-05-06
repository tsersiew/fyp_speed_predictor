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
    scaler = MinMaxScaler(feature_range=(0,1))
    model = load_model('speed_prediction.h5') 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

predictions = predict(x_test)

# check
print(predictions.shape)