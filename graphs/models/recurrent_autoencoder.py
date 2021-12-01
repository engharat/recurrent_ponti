"""
Recurrent Autoencoder PyTorch implementation
"""

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from functools import partial
import math

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
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv1d(n_features*4,n_features*8,kernel_size=3,stride=2, padding=1,bias=True)
        self.conv4 = nn.Conv1d(n_features*8,n_features*16,kernel_size=3,stride=2, padding=1,bias=True)
        self.bn3 = nn.BatchNorm1d(n_features*8)
        self.bn4 = nn.BatchNorm1d(n_features*16)
        self.rec_enc1 = rnn(n_features*16, latent_dim, batch_first=True)

    def forward(self, x):
        
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.conv1(x)
        x = self.bn1(self.relu(x))
        x = self.conv2(x)
        x = self.bn2(self.relu(x))

        x = self.conv3(x)
        x = self.bn3(self.relu(x))
        x = self.conv4(x)
        x = self.bn4(self.relu(x))
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
        self.rec_dec1 = rnn_cell(n_features*16, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features*16)
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose1d(n_features*16,n_features*8,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv2 = nn.ConvTranspose1d(n_features*8,n_features*4,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv3 = nn.ConvTranspose1d(n_features*4,n_features*2,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv4 = nn.ConvTranspose1d(n_features*2,n_features,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.bn1 = nn.BatchNorm1d(n_features*8)
        self.bn2 = nn.BatchNorm1d(n_features*4)
        self.bn3 = nn.BatchNorm1d(n_features*2)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = [h.squeeze(0) for h in h_0]

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])
        bs = x_i.shape[0]
        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            x = torch.cat([x, x_i], axis = 1)
        x = x.view(bs, seq_len, self.dense_dec1.out_features)
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.deconv1(x)
        x = self.bn1(self.relu(x))

        x = self.deconv2(x)
        x = self.bn2(self.relu(x))

        x = self.deconv3(x)
        x = self.bn3(self.relu(x))

        x = self.deconv4(x)
        x = x.swapaxes(1,2) #now we have bck on axis 0 the batch, on axis 1 the  and on axis 2 the sequencefeatures 
        return x

class RecurrentDecoderConvLSTM_BACKUP(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features*16, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features*16)
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose1d(n_features*16,n_features*8,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv2 = nn.ConvTranspose1d(n_features*8,n_features*4,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv3 = nn.ConvTranspose1d(n_features*4,n_features*2,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv4 = nn.ConvTranspose1d(n_features*2,n_features,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.bn1 = nn.BatchNorm1d(n_features*8)
        self.bn2 = nn.BatchNorm1d(n_features*4)
        self.bn3 = nn.BatchNorm1d(n_features*2)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = [h.squeeze(0) for h in h_0]

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])
        bs = x_i.shape[0]
        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            x = torch.cat([x, x_i], axis = 1)
        x = x.view(bs, seq_len, self.dense_dec1.out_features)
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.deconv1(x)
        x = self.bn1(self.relu(x))

        x = self.deconv2(x)
        x = self.bn2(self.relu(x))

        x = self.deconv3(x)
        x = self.bn3(self.relu(x))

        x = self.deconv4(x)
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

        return out #torch.flip(out, [1])

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


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, 0.0) #instead of 0.0 it was dropout
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout,batch_first=True)
        decoder_layers = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder1 = nn.Linear(ninp, ninp)
        self.decoder2 = nn.Linear(ninp, ninp)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.decoder1.weight,-initrange,initrange)
        nn.init.uniform_(self.decoder2.weight,-initrange,initrange)

    def forward(self, src, has_mask=None):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
       # src = self.encoder(src) * math.sqrt(self.ninp)
       # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
#        output = self.decoder1(output)
#        output = self.transformer_decoder(src,output)
#        output = self.decoder2(output)
        return output

class TransformerAE(nn.Module):
    """Recurrent autoencoder"""

    def __init__(self, config,device):
        super().__init__()

        # Encoder and decoder configuration
        self.config = config
        seq_len = 512
        self.transformer = TransformerModel(ntoken=1,ninp=self.config.n_features*4,nhead=2*4,nhid=seq_len//4,nlayers=2,dropout=0.5)

        self.latent_dim = self.config.latent_dim
        self.n_features = self.config.n_features
        n_features = self.n_features
        self.device = device

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(n_features,n_features*2,kernel_size=3,stride=2, padding=1,bias=True)
        self.conv2 = nn.Conv1d(n_features*2,n_features*4,kernel_size=3,stride=2, padding=1,bias=True)
        self.bn1 = nn.BatchNorm1d(n_features*2)
        self.bn2 = nn.BatchNorm1d(n_features*4)
        self.bn3 = nn.BatchNorm1d(n_features*2)

        self.deconv1 = nn.ConvTranspose1d(n_features*4,n_features*2,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)
        self.deconv2 = nn.ConvTranspose1d(n_features*2,n_features,kernel_size=3,stride=2, padding=1,output_padding=1,bias=True)

    def forward(self, x):
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .

        x = self.transformer(x)

        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
        x = self.deconv1(x)
        x = self.bn3(self.relu(x))

        x = self.deconv2(x)
        x = x.swapaxes(1,2) #now we have on axis 0 the batch, on axis 1 the features and on axis 2 the sequence .
 
        return x #torch.flip(out, [1])


