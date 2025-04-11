import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.TimeVar import TimeVar
from model.FrequencyVar import FrequencyVar

class DynamicFusion(nn.Module):
    def __init__(self, d_model):
        super(DynamicFusion, self).__init__()
        self.d_model = d_model
        
        self.time_query_proj = nn.Linear(d_model, d_model)
        self.time_key_proj = nn.Linear(d_model, d_model)
        self.time_value_proj = nn.Linear(d_model, d_model)
        
        self.freq_query_proj = nn.Linear(d_model, d_model)
        self.freq_key_proj = nn.Linear(d_model, d_model)
        self.freq_value_proj = nn.Linear(d_model, d_model)
        
        self.temperature = nn.Parameter(torch.sqrt(torch.FloatTensor([d_model])))
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.GELU()
        )
        
    def compute_time_attention_weights(self, query):
        q = self.time_query_proj(query)  # [batch_size, seq_len, fusion_dim]
        k = self.time_key_proj(query)    # [batch_size, seq_len, fusion_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        weights = F.softmax(scores, dim=-1)
        
        return weights
    
    def compute_freq_attention_weights(self, query):
        q = self.freq_query_proj(query)  # [batch_size, seq_len, fusion_dim]
        k = self.freq_key_proj(query)    # [batch_size, seq_len, fusion_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        weights = F.softmax(scores, dim=-1)
        
        return weights
    
    def forward(self, time_feat, freq_feat, time_recon_error, freq_recon_error):

        time_attention = self.compute_time_attention_weights(time_feat)
        freq_attention = self.compute_freq_attention_weights(freq_feat)
        
        time_context = torch.matmul(time_attention, self.time_value_proj(time_feat))
        freq_context = torch.matmul(freq_attention, self.freq_value_proj(freq_feat))
        
        recon_weights = torch.stack([
            1 / (time_recon_error + 1e-6),
            1 / (freq_recon_error + 1e-6)
        ], dim=-1)
        recon_weights = F.softmax(recon_weights, dim=-1)
        
        time_weight = recon_weights[..., 0].unsqueeze(-1) * time_context
        freq_weight = recon_weights[..., 1].unsqueeze(-1) * freq_context
        
        fused_features = self.fusion_layer(torch.cat([time_weight, freq_weight], dim=-1))
        
        return fused_features, recon_weights

class TimeFrequencyVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, e_layers, d_model, n_memory, device, 
                 memory_initial=True, memory_init_embedding=None, phase_type=None, dataset_name=None):
        super(TimeFrequencyVar, self).__init__()
        
        self.time_model = TimeVar(win_size=win_size, enc_in=enc_in, c_out=c_out, e_layers=e_layers, d_model=d_model, n_memory=n_memory, 
                                device=device, memory_initial=memory_initial, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)
        self.freq_model = FrequencyVar(win_size=win_size, enc_in=enc_in, c_out=c_out, e_layers=e_layers, d_model=d_model, n_memory=n_memory, 
                                device=device, memory_initial=memory_initial, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)
        
        self.memory_initial = memory_initial        
        self.linear_projection = nn.Linear(c_out, 4 * d_model)
        self.dynamic_fusion = DynamicFusion(2 * d_model)
        
        self.criterion = nn.MSELoss(reduction='none')
        
        self.phase_weight = nn.Parameter(torch.tensor(0.5))  
        
    def compute_freq_error(self, pred_spectrum, true_spectrum):
        amplitude_error = torch.abs(torch.abs(pred_spectrum) - torch.abs(true_spectrum))
        
        phase_error = torch.abs(torch.angle(pred_spectrum) - torch.angle(true_spectrum))
        
        combined_error = amplitude_error + self.phase_weight * phase_error
        return torch.mean(combined_error, dim=-1)  
    
    
    def fft_transform(self, x):
        x_complex = torch.fft.fft(x, dim=1, norm='forward')
        return x_complex            
        
    def forward(self, x):
        time_out = self.time_model(x)
        time_output = time_out['out']
        time_queries = time_out['queries']
                
        freq_out = self.freq_model(x)
        freq_output = freq_out['out']
        freq_queries = freq_out['queries']

        if self.memory_initial:
            x = self.linear_projection(x)
            
        time_recon_error = torch.mean(self.criterion(time_output, x), dim=-1)
        
        x_freq = self.fft_transform(x)
        freq_output_freq = self.fft_transform(freq_output)
        freq_recon_error = self.compute_freq_error(freq_output_freq, x_freq)
        
        fused_features, dynamic_weights = self.dynamic_fusion(
            time_queries, freq_queries,
            time_recon_error, freq_recon_error
        )
        
        final_output = (dynamic_weights[..., 0].unsqueeze(-1) * time_output + 
                       dynamic_weights[..., 1].unsqueeze(-1) * freq_output)
        
        return {
            'out': final_output,
            'time_out': time_output,
            'freq_out': freq_output,
            'fused_features': fused_features,
            'dynamic_weights': dynamic_weights,
            'reconstruction_errors': {
                'time': time_recon_error,
                'freq': freq_recon_error
            },
            'queries': fused_features, 
            'mem': time_out['mem'], 
            'attn': time_out['attn'],  
            'memory_item_embedding': time_out['memory_item_embedding'] 
        }