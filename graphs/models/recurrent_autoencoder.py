"""
Recurrent Autoencoder PyTorch implementation
"""

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from functools import partial


class RecurrentEncoder(nn.Module):
    """Recurrent encoder"""

    def __init__(self, n_features, latent_dim, rnn):
        super().__init__()

        self.rec_enc1 = rnn(n_features, latent_dim, batch_first=True)

    def forward(self, x):
        seq_len = x.shape[1]
        _, h_n = self.rec_enc1(x)

        return h_n,seq_len

class RecurrentDecoder(nn.Module):
    """Recurrent decoder for RNN and GRU"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)
        # Squeezing
        h_i = h_0.squeeze(0)

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)

        return x.view(-1, seq_len, self.n_features)

class RecurrentEncoderConvLSTM(nn.Module):
    """Recurrent encoder"""

    def __init__(self, n_features, latent_dim, rnn):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features,n_features,kernel_size=3,stride=2, padding=1,bias=True)#,dilation=3
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_features,n_features,kernel_size=3,stride=2, padding=1,bias=True)
        self.bn1 = nn.BatchNorm1d(n_features)
        self.bn2 = nn.BatchNorm1d(n_features)
        self.rec_enc1 = rnn(n_features, latent_dim, batch_first=True)

    def forward(self, x):
        
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.conv1(x)
        x = self.bn1(self.relu(x))
        x = self.conv2(x)
        x = self.bn2(self.relu(x))
        x = x.swapaxes(1,2) #now we have bck on axis 0 the batch, on axis 1 the sequence and on axis 2 the features         
        seq_len = x.shape[1]
        _, h_n = self.rec_enc1(x)

        return h_n, seq_len

class RecurrentDecoderConvLSTM(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose1d(n_features,n_features,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv2 = nn.ConvTranspose1d(n_features,n_features,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.bn1 = nn.BatchNorm1d(n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = [h.squeeze(0) for h in h_0]

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            x = torch.cat([x, x_i], axis = 1)
        x = x.view(-1, seq_len, self.n_features)
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.deconv1(x)
        x = self.bn1(self.relu(x))
        x = self.deconv2(x)
        x = x.swapaxes(1,2) #now we have bck on axis 0 the batch, on axis 1 the  and on axis 2 the sequencefeatures 
        return x
class RecurrentDecoderLSTM(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = [h.squeeze(0) for h in h_0]

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            x = torch.cat([x, x_i], axis = 1)

        return x.view(-1, seq_len, self.n_features)


class RecurrentAE(nn.Module):
    """Recurrent autoencoder"""

    def __init__(self, config,device):
        super().__init__()

        # Encoder and decoder configuration
        self.config = config
        self.rnn, self.rnn_cell = self.get_rnn_type(self.config.rnn_type, self.config.rnn_act)
        self.decoder = self.get_decoder(self.config.rnn_type)
        self.encoder = self.get_encoder(self.config.rnn_type)
        
        self.latent_dim = self.config.latent_dim
        self.n_features = self.config.n_features
        self.device = device

        # Encoder and decoder
        self.encoder = self.encoder(self.n_features, self.latent_dim, self.rnn)
        self.decoder = self.decoder(self.latent_dim, self.n_features, self.rnn_cell, self.device)

    def forward(self, x):
        h_n,seq_len = self.encoder(x)
        out = self.decoder(h_n, seq_len)

        return torch.flip(out, [1])

    @staticmethod
    def get_rnn_type(rnn_type, rnn_act=None):
        """Get recurrent layer and cell type"""
        if rnn_type == 'RNN':
            rnn = partial(nn.RNN, nonlinearity=rnn_act)
            rnn_cell = partial(nn.RNNCell, nonlinearity=rnn_act)
        elif rnn_type == 'ConvLSTM':
            rnn = getattr(nn, 'LSTM')
            rnn_cell = getattr(nn, 'LSTM' + 'Cell')
        else:
            rnn = getattr(nn, rnn_type)
            rnn_cell = getattr(nn, rnn_type + 'Cell')

        return rnn, rnn_cell

    @staticmethod
    def get_decoder(rnn_type):
        """Get recurrent decoder type"""
        if rnn_type == 'LSTM':
            decoder = RecurrentDecoderLSTM
        elif rnn_type == 'ConvLSTM':
            decoder = RecurrentDecoderConvLSTM

        else:
            decoder = RecurrentDecoder

        return decoder


    @staticmethod
    def get_encoder(rnn_type):
        """Get recurrent decoder type"""
        if rnn_type == 'ConvLSTM':
            decoder = RecurrentEncoderConvLSTM
        else:
            decoder = RecurrentEncoder

        return decoder


if __name__ == '__main__':

    # Configuration
    config = {}
    config['n_features'] = 1
    config['latent_dim'] = 4
    config['rnn_type'] = 'GRU'
    config['rnn_act'] = 'relu'
    config['device'] = 'cpu'
    config = edict(config)

    # Adding random data
    X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32).unsqueeze(2)

    # Model
    model = RecurrentAE(config)

    # Encoder
    h = model.encoder(X)
    out =  model.decoder(h, seq_len = 10)
    out = torch.flip(out, [1])

    # Loss
    loss = nn.L1Loss(reduction = 'mean')
    l = loss(X, out)


