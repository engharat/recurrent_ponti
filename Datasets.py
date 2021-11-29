
import os.path
import os
from os import path
from csv import writer
from csv import reader
import pathlib
from torch.serialization import save
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
from csv import reader
import random
import torchvision.transforms.functional as TF
import glob
import kazane
from sklearn.preprocessing import MinMaxScaler,StandardScaler

class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

class Standard3D(StandardScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x,copy=None), newshape=X.shape)

STD=torch.FloatTensor([0.0020, 0.0010, 0.0012, 0.0011, 0.0009, 0.0022])
MEAN=torch.FloatTensor([ 6.1557e-06, -6.0278e-07,  7.3163e-06,  1.0190e-06, -3.7250e-07,-1.1097e-08])

################################ Dataset ########################
select_list = ['aBD11Az','aBD17Ay','aBD17Az','aBD17Cz','aBD23Ay','aBD23Az']
#select_list = ['aBD11Az']
class KW51(Dataset):

    def __init__(self, base_folder="~/Downloads/traindata_csv/Train_folder_traindata/",substract=False,max_seq_len=31744,decimate_factor=100,scaler=None):
        base_folder = os.path.expanduser(base_folder)
        self.substract = substract
        self.data_paths = glob.glob(base_folder + "/**/*.csv", recursive = True)
        self.datas=[]
        self.max_seq_len = max_seq_len
        self.decimate_factor = decimate_factor
        self.decimater = kazane.Decimate(self.decimate_factor)
        self.mean = MEAN[0:len(select_list)]
        self.std = STD[0:len(select_list)]
        max_length = 0
        self.scaler = StandardScaler() if scaler is None else scaler

        if 'train' in base_folder:
            saved_file = 'train.pt'
        if 'normal' in base_folder:
            saved_file = 'normal.pt'
        if 'anomaly' in base_folder:
            saved_file = 'anomaly.pt'
        if not os.path.exists(saved_file):
            print("LOADING AND PROCESSING DATA...")
            for i in tqdm(range(len(self.data_paths))):
                df = pd.read_csv(self.data_paths[i])
                df = df[select_list]
                data =  torch.from_numpy(np.nan_to_num(df.astype(np.float32).to_numpy()))
                data = data[0:self.max_seq_len, :]
                data = self.decimater(data.T).T
                #data = (data - MEAN) / STD
                if self.substract:
                    initial_values = data[0, :].clone()
                    data -= torch.roll(data, 1, 0)
                    data[0, :] = initial_values

                if data.shape[0]>max_length : max_length = data.shape[0]
                self.datas.append(data)
            indexes = [i for i,elem in enumerate(self.datas) if elem.shape[0] < max_length] #ho ottenuto così alcuni elementi che erano più corti e danno problemi al batch
            for index in sorted(indexes, reverse=True): #li rimuovo
                del self.datas[index]
            self.datas = torch.stack([elem for elem in self.datas])
            ##scaling
            #self.datas = (self.datas - self.mean) / self.std
            torch.save(self.datas,saved_file)
            print("SAVING FILE: "+saved_file)
        else:
            print("LOADING SAVED FILE: "+saved_file)
            self.datas = torch.load(saved_file)
        self.n_samples,self.seq_len, self.n_features = self.datas.shape
        self.datas = self.datas.view(-1,self.n_features)
        self.datas = torch.from_numpy(self.scaler.fit_transform(self.datas.numpy())) if scaler is None else torch.from_numpy(self.scaler.transform(self.datas.numpy()))
        self.datas = self.datas.view(self.n_samples,self.seq_len,self.n_features)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.datas[i,:,:]
