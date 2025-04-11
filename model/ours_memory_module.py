from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans


class MemoryModule(nn.Module):
    def __init__(self, n_memory, fea_dim, shrink_thres=0.0025, device=None, memory_init_embedding=None, phase_type=None, dataset_name=None, type="Time"):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        self.memory_init_embedding = memory_init_embedding  
        
        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)
        
        if self.memory_init_embedding == None:
            if self.phase_type =='test':                

                load_path = f'./memory_item/{dataset_name}_memory_item.pth'
                self.mem = torch.load(load_path) 

                print(load_path)
                print('loading memory item vectors trained from kmeans (for test phase)')

            else:
                print('loading memory item with random initilzation (for first train phase)')

                self.mem = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1)

        else:
            if self.phase_type == 'second_train':
                print('second training (for second train phase)')

                self.mem = memory_init_embedding
            

    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        
        return output
    
    def get_attn_score(self, query, key):
        
        attn = torch.matmul(query, torch.t(key.cuda()))   
        attn = F.softmax(attn, dim=-1)

        if (self.shrink_thres > 0):
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            # re-normalize
            attn = F.normalize(attn, p=1, dim=1)    
                
        return attn
    
    def read(self, query): 

        self.mem = self.mem.cuda()
        attn = self.get_attn_score(query, self.mem.detach()) 
        add_memory = torch.matmul(attn, self.mem.detach())   

        read_query = torch.cat((query, add_memory), dim=1) 

        return {'output': read_query, 'attn': attn}

    def update(self, query):   

        self.mem = self.mem.cuda()
        

        attn = self.get_attn_score(self.mem, query.detach())  
        add_mem = torch.matmul(attn, query.detach())  

        update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem))
        self.mem = (1 - update_gate)*self.mem + update_gate*add_mem

    def forward(self, query):

        
        s = query.data.shape
        l = len(s) 

        query = query.contiguous()
        query = query.view(-1, s[-1])  

        if self.phase_type != 'test':
            self.update(query)
        
        outs = self.read(query)
        
        read_query, attn = outs['output'], outs['attn']
        
        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2*s[2])
            
            attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError('Wrong input dimension')
        '''
        output : N x L x 2C or N x 2C
        attn : N x L x M or N x M
        '''
        return {'output': read_query, 'attn': attn, 'memory_init_embedding':self.mem}
