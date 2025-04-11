import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding

from .ours_memory_module import MemoryModule

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)   
    

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()

        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        out = self.out_linear(x)
        return out      # N x L x c_out

class FrequencyFeatureExtractorV2(nn.Module):
    def __init__(self, feature_size, dropout_rate=0.1):
        super(FrequencyFeatureExtractorV2, self).__init__()
        self.linear = nn.Linear(2 * feature_size, feature_size)  
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        B, L, N = X.shape
        
        X_fft = torch.fft.fft(X, dim=1, norm='forward')
        
        X_fft_real = X_fft.real
        X_fft_imag = X_fft.imag
        X_fft_combined = torch.cat([X_fft_real, X_fft_imag], dim=-1)
        
        X_filtered = self.linear(X_fft_combined)
        
        X_out = self.dropout(X_filtered)
        
        return X_out

class TCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super(TCNEncoder, self).__init__()
        self.tcn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))  
        x = self.relu(x)
        x = self.dropout(x)
        return x

class FrequencyVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(FrequencyVar, self).__init__()

        self.memory_initial = memory_initial

        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)   # N x L x C(=d_model)
        
        self.freq_extractor = FrequencyFeatureExtractorV2(feature_size=enc_in, dropout_rate=dropout)
        self.tencoder = TCNEncoder(in_channels=enc_in, out_channels=2*d_model, kernel_size=3, dropout=dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=2 * d_model, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name, type="Frequency")
        
        self.linear_projection = nn.Linear(2 * d_model, 4 * d_model)
        self.weak_decoder = Decoder(4 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def forward(self, x):
        
        x_freq = self.freq_extractor(x) 
        x_encoded = self.tencoder(x_freq)  
        x_encoded = x_encoded.permute(0, 2, 1)
        queries = out = x_encoded
        
        outputs = self.mem_module(out)
    
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']

        mem = self.mem_module.mem
        
        if self.memory_initial:
            return {"out":out, "memory_item_embedding":None, "queries":queries, "mem":mem, "attn":attn}
        else:
            
            out = self.weak_decoder(out)
            return {"out":out, "memory_item_embedding":memory_item_embedding, "queries":queries, "mem":mem, "attn":attn}
