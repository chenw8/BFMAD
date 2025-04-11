import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer, AttentionLayer_v, FullAttention
from .embedding import TokenEmbedding, InputEmbedding, DataEmbedding_inverted

from .ours_memory_module import MemoryModule

class EncoderLayer_v(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_v, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
    

class Encoder_v(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder_v, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x

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
        return out    


class TimeVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TimeVar, self).__init__()

        self.memory_initial = memory_initial

        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)  
        self.enc_embedding = DataEmbedding_inverted(c_in=win_size, d_model=d_model)
       
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

        self.encoder_v = Encoder_v(
            [
                EncoderLayer_v(
                    AttentionLayer_v(
                        FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, win_size, bias=True)
        self.linear_projection = nn.Linear(enc_in, d_model)
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=2 * d_model, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name, type="Time")
        
        self.weak_decoder = Decoder(4 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)


    def forward(self, x):
        
        _, L, N = x.shape
        
        x_v = x
        x_v = self.enc_embedding(x_v, None)  
        x = self.embedding(x)  

        queries_v = out_v = self.encoder_v(x_v, attn_mask=None)   
        queries_v = out_v = self.projection(out_v).permute(0, 2, 1)[:, :, :N]        
        queries_v = out_v = self.linear_projection(queries_v)
                                
        queries = out = self.encoder(x)  

        Q = torch.cat((queries, queries_v), dim=-1)

        outputs = self.mem_module(Q) 
        
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']

        mem = self.mem_module.mem
        
        if self.memory_initial:
            return {"out":out, "memory_item_embedding":None, "queries":Q, "mem":mem, "attn":attn}
        else:
            
            out = self.weak_decoder(out)

            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out":out, "memory_item_embedding":memory_item_embedding, "queries":Q, "mem":mem, "attn":attn}
