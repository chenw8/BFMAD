# Some code based on https://github.com/thuml/Anomaly-Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.TimeVar import TimeVar
from model.FrequencyVar import FrequencyVar
from model.fusion import DynamicFusion, TimeFrequencyVar
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)

        self.test_loader, _ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = self.vali_loader
        
        if self.memory_initial == "False":
            self.memory_initial = False
        else:
            self.memory_initial = True

        self.memory_init_embedding = None
        self.build_model(memory_init_embedding=self.memory_init_embedding)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def build_model(self, memory_init_embedding):
        if self.mode == "train" or self.mode == "memory_initial":
            self.model = TimeFrequencyVar(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, 
                                    e_layers=3, d_model=self.d_model, n_memory=self.n_memory, device=self.device, 
                                    memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding, 
                                    phase_type=self.phase_type, dataset_name=self.dataset)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                self.model = torch.nn.DataParallel(self.model, device_ids=[0], output_device=0).to(self.device)
        elif self.mode == "test":
            self.model = TimeFrequencyVar(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, 
                                    e_layers=3, d_model=self.d_model, n_memory=self.n_memory, device=self.device, 
                                    memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding, 
                                    phase_type=self.phase_type, dataset_name=self.dataset)
            
            if torch.cuda.is_available():
                self.model = torch.nn.DataParallel(self.model, device_ids=[0], output_device=0).to(self.device)
            
    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = [] 
        valid_re_loss_list = [] 
        valid_entropy_loss_list = []
        valid_time_re_loss_list = []
        valid_freq_re_loss_list = []

        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output_dict = self.model(input)
                
                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']
                attn = output_dict['attn']
                time_output = output_dict['time_out']
                freq_output = output_dict['freq_out']
                
                rec_loss = self.criterion(output, input)
                
                time_rec_loss = self.criterion(time_output, input)
                freq_rec_loss = self.criterion(freq_output, input)
                
                entropy_loss = self.entropy_loss(attn)
                
                loss = rec_loss + self.lambd * entropy_loss

                valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
                valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                valid_loss_list.append(loss.detach().cpu().numpy())
                valid_time_re_loss_list.append(time_rec_loss.detach().cpu().numpy())
                valid_freq_re_loss_list.append(freq_rec_loss.detach().cpu().numpy())
                
                gc.collect()
                torch.cuda.empty_cache()

        return (np.average(valid_loss_list), np.average(valid_re_loss_list), 
                np.average(valid_entropy_loss_list), np.average(valid_time_re_loss_list),
                np.average(valid_freq_re_loss_list))

    def train(self, training_type):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset, type=training_type)
        train_steps = len(self.train_loader)

        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = []
            entropy_loss_list = []
            time_rec_loss_list = []
            freq_rec_loss_list = []
            dynamic_weights_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)    

                output_dict = self.model(input)
                
                output = output_dict['out']
                memory_item_embedding = output_dict['memory_item_embedding']
                queries = output_dict['queries']
                mem_items = output_dict['mem']
                attn = output_dict['attn']
                time_output = output_dict['time_out']
                freq_output = output_dict['freq_out']
                dynamic_weights = output_dict['dynamic_weights']
                
                rec_loss = self.criterion(output, input)
                
                time_rec_loss = self.criterion(time_output, input)
                freq_rec_loss = self.criterion(freq_output, input)
                
                entropy_loss = self.entropy_loss(attn)
                
                loss = rec_loss + self.lambd * entropy_loss

                loss_list.append(loss.detach().cpu().numpy())
                entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())
                time_rec_loss_list.append(time_rec_loss.detach().cpu().numpy())
                freq_rec_loss_list.append(freq_rec_loss.detach().cpu().numpy())
                
                dynamic_weights_avg = torch.mean(dynamic_weights, dim=[0, 1]).detach().cpu().numpy()
                dynamic_weights_list.append(dynamic_weights_avg)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                try:
                    loss.mean().backward()
                except:
                    import pdb; pdb.set_trace()
                    
                self.optimizer.step()
                
                gc.collect()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_entropy_loss = np.average(entropy_loss_list)
            train_rec_loss = np.average(rec_loss_list)
            train_time_rec_loss = np.average(time_rec_loss_list)
            train_freq_rec_loss = np.average(freq_rec_loss_list)
            train_dynamic_weights = np.mean(dynamic_weights_list, axis=0)
            
            valid_loss, valid_rec_loss, valid_entropy_loss, valid_time_rec_loss, valid_freq_rec_loss = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | VALID Rec Loss: {2:.7f} Entropy Loss: {3:.7f} Time Rec Loss: {4:.7f} Freq Rec Loss: {5:.7f}".format(
                    epoch + 1, train_steps, valid_rec_loss, valid_entropy_loss, valid_time_rec_loss, valid_freq_rec_loss))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN Rec Loss: {2:.7f} Entropy Loss: {3:.7f} Time Rec Loss: {4:.7f} Freq Rec Loss: {5:.7f}".format(
                    epoch + 1, train_steps, train_rec_loss, train_entropy_loss, train_time_rec_loss, train_freq_rec_loss))
            print(
                "Epoch: {0}, Steps: {1} | Dynamic Weights: Time: {2:.4f} Freq: {3:.4f}".format(
                    epoch + 1, train_steps, train_dynamic_weights[0], train_dynamic_weights[1]))

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            gc.collect()
            torch.cuda.empty_cache()
            
        return memory_item_embedding
    
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')))
        self.model.eval()
        
        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)   
        gathering_loss = GatheringLoss(reduce=False)  
        temperature = self.temperature

        train_attens_energy = [] 
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            
            output_dict = self.model(input)
            output = output_dict['out']
            queries = output_dict['queries']
            mem_items = output_dict['mem']
            
            rec_loss = torch.mean(criterion(input, output), dim=-1)
            
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            
            loss = latent_score * rec_loss
            
            cri = loss.detach().cpu().numpy()
            train_attens_energy.append(cri)

        train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)
        train_energy = np.array(train_attens_energy)

        valid_attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            
            output_dict = self.model(input)
            output = output_dict['out']
            queries = output_dict['queries']
            mem_items = output_dict['mem']
            
            rec_loss = torch.mean(criterion(input, output), dim=-1)
            
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            
            loss = latent_score * rec_loss
            
            cri = loss.detach().cpu().numpy()
            valid_attens_energy.append(cri)

        valid_attens_energy = np.concatenate(valid_attens_energy, axis=0).reshape(-1)
        valid_energy = np.array(valid_attens_energy)

        combined_energy = np.concatenate([train_energy, valid_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        test_labels = []
        test_attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            
            output_dict = self.model(input)
            output = output_dict['out']
            queries = output_dict['queries']
            mem_items = output_dict['mem']
            
            rec_loss = torch.mean(criterion(input, output), dim=-1)
            
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            
            loss = latent_score * rec_loss
            
            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_labels.append(labels)

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else: 
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)   
        gt = np.array(gt)  
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                            average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        print('='*50)

        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"number of items: {self.n_memory}")
        self.logger.info(f"Precision: {round(precision,4)}")
        self.logger.info(f"Recall: {round(recall,4)}")
        self.logger.info(f"f1_score: {round(f_score,4)} \n")
        return accuracy, precision, recall, f_score

    def get_memory_initial_embedding(self, training_type='second_train'):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_first_train.pth')))
        
        self.model.eval()
        
        for i, (input_data, labels) in enumerate(self.k_loader):
            input = input_data.float().to(self.device)
                        
            if i==0:
                output = self.model(input)['queries']
            else:
                output = torch.cat([output, self.model(input)['queries']], dim=0)

        self.memory_init_embedding = gmm_clustering(x=output, n_components=self.n_memory, d_model=2*self.d_model) 

        self.memory_initial = False
        self.build_model(memory_init_embedding=self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)

        memory_item_embedding = memory_item_embedding[:int(self.n_memory),:]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(item_folder_path, str(self.dataset) + '_memory_item.pth')
        torch.save(memory_item_embedding, item_path)