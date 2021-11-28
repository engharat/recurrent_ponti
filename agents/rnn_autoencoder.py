"""
The RNN autoencoder agent class
"""

import torch
from torch import nn

import numpy as np
from tqdm import tqdm
import shutil
import os

from agents.base import BaseAgent
from utils.metrics import AverageMeter
from utils.checkpoints import checkpoints_folder
from utils.config import save_config
from graphs.models.recurrent_autoencoder import RecurrentAE
from graphs.losses.MAEAUCLoss import MAEAUCLoss
from graphs.losses.MSEAUCLoss import MSEAUCLoss
from graphs.losses.MAELoss import MAELoss
from graphs.losses.MSELoss import MSELoss
from Datasets import KW51
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import randint
import pathlib
import sys
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

class RecurrentAEAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

         # Create an instance from the Model
        self.seed = randint(0,1000)
        self.exper_path = './runs/'+str(self.seed)
        pathlib.Path(self.exper_path).mkdir(parents=True,exist_ok=True)
        self.writer = SummaryWriter(self.exper_path)
        print("Experiment n: "+str(self.seed))
        self.model = RecurrentAE(self.config,self.device)
        self.train_dataset = KW51('~/Downloads/ambient_csv/train',substract=False,decimate_factor=config.decimate_factor,scaler=None)
        self.val_dataset = KW51('~/Downloads/ambient_csv/test/normal',substract=False,decimate_factor=config.decimate_factor,scaler=self.train_dataset.scaler)
        self.val_anomaly_dataset = KW51('~/Downloads/ambient_csv/test/anomaly',substract=False,decimate_factor=config.decimate_factor,scaler=self.train_dataset.scaler)
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=0,drop_last=False)
        self.val_dataloader = DataLoader(self.val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=0,drop_last=False)
        self.val_anomaly_dataloader = DataLoader(self.val_anomaly_dataset,batch_size=config.batch_size,shuffle=False,num_workers=0,drop_last=False)
        self.split_size = config.split_size
         # Create instance from the loss
        self.loss = {'MSE': MSELoss(),
                     'MAE': MAELoss(),
                     'MSEAUC': MSEAUCLoss(),
                     'MAEAUC': MAEAUCLoss()}[self.config.loss]

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate)

        # Training info
        self.current_epoch = 0

        # Creating folder where to save checkpoints
        self.checkpoints_path = checkpoints_folder(self.config)

        # Initialize my counters
        self.current_epoch = 0
        self.best_valid = 10e+16 # Setting a very large values
        self.train_loss = np.array([], dtype = np.float64)
        self.train_loss_parz = np.array([], dtype=np.float64)
        self.valid_loss = np.array([], dtype = np.float64)
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Loading chekpoint
        self.load_checkpoint(self.config.checkpoint_file)

    def predict(self,dataset):
        gts, predictions, losses = [], [], []
        criterion = nn.L1Loss(reduction='mean').to(self.device)
        with torch.no_grad():
            self.model.eval()
            for seq_true in tqdm(dataset):
                if len(seq_true.shape)==2:
                    seq_true = seq_true[None,...]
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
                gts.append(seq_true.cpu().numpy().flatten())
                losses.append(loss.item())
        return gts,predictions, losses

    def thres(self,losses,losses_anomaly):
        while(True):
            threshold = float(input("Scrivere la threshold: "))
            correct = sum(l <= threshold for l in losses)
            print("prediction loss is: "+ str( np.asarray(losses).mean()))
            print(f'Correct normal predictions: {correct}/{len(self.val_dataset)}')

            correct = sum(l > threshold for l in losses_anomaly)
            print("prediction loss is: "+ str( np.asarray(losses_anomaly).mean()))
            print(f'Correct anomaly predictions: {correct}/{len(self.val_anomaly_dataset)}')

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            #validating before training
            perf_valid = self.validate_one_epoch() if epoch == 0 else None

            # Training epoch
            perf_train = self.train_one_epoch()
            self.train_loss = np.append(self.train_loss, perf_train[0].avg)
            if self.current_epoch % 10 ==0:
                print('Training loss at epoch ' + str(self.current_epoch) + ' is ' + str(perf_train[0].avg))

            if self.current_epoch % 10 == 0:
                # Validation
                perf_valid = self.validate_one_epoch()
                self.valid_loss = np.append(self.valid_loss, perf_valid.avg)
                if self.current_epoch % 10 ==0:
                    print('Validation loss at epoch ' + str(self.current_epoch) + ' is ' + str(perf_valid.avg))
                sys.stdout.write("\x1b[1A\x1b[2K") # move up cursor and delete whole line
                sys.stdout.write("\x1b[1A\x1b[2K") # move up cursor and delete whole line
                # Saving
                is_best = perf_valid.sum < self.best_valid
                if is_best:
                    self.best_valid = perf_valid.sum
                self.save_checkpoint(is_best=is_best)

        correct_normal, correct_anomaly = 0, 0
        #playing the SVM game
        print("SVM training...")        
        gts_normal, preds_normal,losses_normal = self.predict(self.val_dataloader.dataset)
        gts_anomaly, preds_anomaly,losses_anomaly = self.predict(self.val_anomaly_dataloader.dataset)

        diff_normal = [elem1-elem2 for elem1,elem2 in zip(gts_normal, preds_normal)]
        label_normal = [0 for elem in diff_normal]
        diff_anomaly = [elem1-elem2 for elem1,elem2 in zip(gts_anomaly, preds_anomaly)]
        label_anomaly = [1 for elem in diff_anomaly]
        X = np.asarray(diff_normal + diff_anomaly)
        y = np.asarray(label_normal + label_anomaly)
        clf = svm.SVC()
        clf.fit(X, y)
        for elem,gt in zip(diff_normal,label_normal):
            svm_pred = clf.predict(elem[None,...])
            correct_normal += (gt == svm_pred).sum()
        print(f'SVM Correct normal : {correct_normal}/{len(diff_normal)}')

        for elem,gt in zip(diff_anomaly,label_anomaly):
            svm_pred = clf.predict(elem[None,...])
            correct_anomaly += (gt == svm_pred).sum()
        print(f'SVM Correct anomaly : {correct_anomaly}/{len(diff_anomaly)}')

        import umap
        fig = plt.figure(figsize=(8,8))
        colors = ['red','blue']
        umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap_2d.fit_transform(X)
        plt.scatter(proj_2d[:,0], proj_2d[:,1], c=y, cmap=matplotlib.colors.ListedColormap(colors))
        cb = plt.colorbar()
        loc = np.arange(0,max(y),max(y)/float(len(colors)))
        cb.set_ticks(loc)
        cb.set_ticklabels(colors)
        self.writer.add_figure('figure',fig)
        self.writer.flush()
        #except:
        #    pass

        #playing the threshold game
        self.thres(losses_normal,losses_anomaly)

    def train_one_epoch(self):
        """ One epoch training step """

        # Initialize tqdm
        tqdm_batch = tqdm(self.train_dataloader, total = len(self.train_dataloader),
                         desc ="Epoch-{}-".format(self.current_epoch),position=0,leave=True)

        # Set the model to be in training mode
        self.model.train()

        # Initialize your average meters
        epoch_loss = AverageMeter()
        epoch_loss_parz = AverageMeter()

        # One epoch of training
        for x in tqdm_batch: 
            #for x in torch.split(x_seq,self.split_size,1):

            x = x.to(self.device)
                # Model
            x_hat = self.model(x)
            cur_tr_loss = self.loss(x, x_hat)
           
                # Optimizer
            self.optimizer.zero_grad()
            cur_tr_loss.backward()
            self.optimizer.step()

            # Updating loss
            epoch_loss.update(cur_tr_loss.item())

        tqdm_batch.close()
      
        return epoch_loss, epoch_loss_parz

    def validate_one_epoch(self):
        """ One epoch validation step """
        # Initialize tqdm
        tqdm_batch = tqdm(self.val_dataloader, total = len(self.val_dataloader),
                         desc = "Validation at epoch -{}-".format(self.current_epoch),position=0,leave=True)

        # Set the model to be in evaluation mode
        self.model.eval()

        # Initialize your average meters
        epoch_loss = AverageMeter()
        with torch.no_grad():
            plot_feat = 0

            seq_true = self.train_dataloader.dataset[0]
            seq_true = seq_true[None,...].to(self.device)
            seq_pred = self.model(seq_true)
            seq_diff = torch.abs(seq_pred - seq_true)
            for i in range(seq_true.shape[1]):
                self.writer.add_scalars('train', {'seq_true':seq_true[0,i,plot_feat],'seq_pred':seq_pred[0,i,plot_feat]}, i)
            self.writer.add_scalars('mean_train', {'seq_diff_mean':seq_diff[0,:,:].mean()}, self.current_epoch)

            seq_true = self.val_dataloader.dataset[0]
            seq_true = seq_true[None,...].to(self.device)
            seq_pred = self.model(seq_true)
            seq_diff = torch.abs(seq_pred - seq_true)
            for i in range(seq_true.shape[1]):
                self.writer.add_scalars('val_norm', {'seq_true':seq_true[0,i,plot_feat],'seq_pred':seq_pred[0,i,plot_feat]}, i)
            self.writer.add_scalars('mean_val_norm', {'seq_diff_mean':seq_diff[0,:,:].mean()}, self.current_epoch)



            seq_true = self.val_anomaly_dataloader.dataset[0]
            seq_true = seq_true[None,...].to(self.device)
            seq_pred = self.model(seq_true)
            seq_diff = torch.abs(seq_pred - seq_true)
            for i in range(seq_true.shape[1]):
                self.writer.add_scalars('val_anom', {'anom_true':seq_true[0,i,plot_feat],'anom_pred':seq_pred[0,i,plot_feat]}, i)
            self.writer.add_scalars('mean_val_anom', {'anom_diff_mean':seq_diff[0,:,:].mean()}, self.current_epoch)

            self.writer.flush()

            for x in tqdm_batch:
                x = x.to(self.device)
                # Model
                x_hat = self.model(x)
                # Current training loss
                cur_val_loss = self.loss(x, x_hat)
                # Updating loss
                epoch_loss.update(cur_val_loss.item())

            tqdm_batch.close()
        return epoch_loss
    
    def save_checkpoint(self, filename ='checkpoint.pth.tar', is_best = 0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'valid_loss': self.valid_loss,
            'train_loss': self.train_loss,
            'train_loss_parz': self.train_loss_parz
        }

        # Save the state
        torch.save(state, self.checkpoints_path + filename)

        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.checkpoints_path + filename,
                            self.checkpoints_path + 'model_best.pth.tar')
            print('Saving a best model')
            sys.stdout.write("\x1b[1A\x1b[2K") # move up cursor and delete whole line

    def load_checkpoint(self, filename):

        if self.config.load_checkpoint:
            filename = self.checkpoints_path + filename
            try:
                checkpoint = torch.load(filename)
                self.current_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.valid_loss = checkpoint['valid_loss']
                self.train_loss = checkpoint['train_loss']
                self.train_loss_parz = checkpoint['train_loss_parz']

                print("Checkpoint loaded successfully from '{}' at (epoch {}) \n"
                                .format(self.checkpoints_path , checkpoint['epoch']))
            except OSError as e:
                print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
        else:
            print('Training a new model from scratch')
            
    def run(self):
        """
        The main operator
        :return:
        """
        # Saving config
        save_config(self.config, self.checkpoints_path)

        # Model training
        self.train()
 
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.save_checkpoint()
        self.data_loader.finalize()

        self.writer.close()


