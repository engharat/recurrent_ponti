import os.path
import os
from os import path
from csv import writer
from csv import reader
import pathlib
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
MEAN =1.2279458e-08
STD = 0.02488539
################################ Dataset ########################
select_list = ['aBD11Az','aBD17Ay','aBD17Az','aBD17Cz','aBD23Ay','aBD23Az']
#select_list = ['aBD11Az']
class KW51(Dataset):

    def __init__(self, base_folder="~/Downloads/traindata_csv/Train_folder_traindata/",substract=False):
        base_folder = os.path.expanduser(base_folder)
        self.substract = substract
        #import pdb; pdb.set_trace()
        self.data_paths = glob.glob(base_folder + "/**/*.csv", recursive = True)
        self.datas=[]
        self.max_seq_len = 32000
        self.decimate_factor = 100
        self.decimater = kazane.Decimate(self.decimate_factor)

        for i in range(len(self.data_paths)):
            df = pd.read_csv(self.data_paths[i])
            df = df[select_list]
            data =  torch.from_numpy(np.nan_to_num(df.astype(np.float32).to_numpy()))
            data = data[0:self.max_seq_len, :]
            data = self.decimater(data.T).T
            data = (data - MEAN) / STD
            if self.substract:
                initial_values = data[0, :].clone()
                data -= torch.roll(data, 1, 0)
                data[0, :] = initial_values
            data = data * 10
            self.datas.append(data)

        self.seq_len, self.n_features = self.datas[0].shape

        #self.seq_len = 100
        #self.n_features = 1

        # tmp = []
        # for path in self.data_paths:
        #     df = pd.read_csv(path)
        #     df = df[select_list]
        #     data =  torch.from_numpy(np.nan_to_num(df.astype(np.float32).to_numpy()))
        #     data = data[0:self.max_seq_len, :]
        #     data = self.decimater(data.T).T
        #     data = data.numpy()
        #     tmp.append(data)
        # import pdb; pdb.set_trace()
        # tmp = np.asarray(tmp)
        # mean = tmp.mean()
        # std = tmp.std()

    def __len__(self):
        return len(self.data_paths)

    def __getitem2__(self, i):
        signal=[np.sin(2*np.pi*i/10) for i in np.arange(0,100,1)]
        aa = np.asarray(signal)
        bb = torch.from_numpy(aa).view(-1,1).float()
        return bb

    def __getitem__(self, i):
        return self.datas[i]
