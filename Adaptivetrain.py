#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import random
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, Subset
import itertools
import pickle
from sklearn.preprocessing import MinMaxScaler
from skmultilearn.dataset import load_dataset
from skmultilearn.model_selection import IterativeStratification
from sklearn.neighbors import NearestNeighbors
import pdb
import scipy
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from enum import Enum
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from util import *
from layers import  *
from model import *
from function import *


# In[2]:


#####seed####
def seed_all(seed): 
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed=42
seed_all(seed)
device = torch.device("cuda:0" ) if torch.cuda.is_available() else torch.device("cpu")


# In[3]:


class CFG:
    def __init__(self,name,X,y):
        self.name=name      
        self.X_train=X
        self.y_train=y
        self.configs={} 
    def getconfig(self):
        self.configs['label_matrix']=np.array(self.y_train)
        self.configs['num_classes']=self.y_train.shape[1] 
        self.configs['num_ins']=self.X_train.shape[0] 
        self.configs['seed']=42 
        self.configs['batch_size']=128 
        self.configs['epoch']=100 
        self.configs['lr']=1e-4
        self.configs['device']=torch.device("cuda:0" ) if torch.cuda.is_available() else torch.device("cpu")
        self.configs['weight']=limb(np.array(self.X_train),np.array(self.y_train))
        self.configs['extra_sample']=int(self.X_train.shape[0]*0.1)
        self.configs['min_ins_idx']=minority_instance(np.array(self.y_train))
        self.configs['minority_label_indices'],_=Labeltype(np.array(self.y_train))
        self.configs['weight_list']=calweight(np.array(self.X_train),np.array(self.y_train))
        #DELA
        if self.name=='DELA':
            self.configs['in_features']=self.X_train.shape[1] 
            self.configs['latent_dim']=math.ceil(self.X_train.shape[1]/2)    
            self.configs['lr_ratio']=0.8
            self.configs['drop_ratio']=0.2
            self.configs['tau']=2/3
            self.configs['beta']=1e-4
            self.configs['out_index']=-1
        if self.name=='CLIF':
            self.configs['class_emb_size']=self.y_train.shape[1]  
            self.configs['input_x_size']=self.X_train.shape[1] 
            self.configs['num_layers']=2 
            self.configs['in_layers']=3 
            self.configs['hidden_list']=[math.ceil(self.y_train.shape[1]/2)]  
            self.configs['out_index']=0        
        if self.name=='PACA':
            self.configs['drop_ratio']=0.1
            self.configs['latent_dim']=math.ceil(self.X_train.shape[1]/2)
            self.configs['in_features']=self.X_train.shape[1] #输入x的维度
            self.configs['rand_seed']=self.configs['seed']
            self.configs['eps']=1e-8
            self.configs['out_index']=-2
            self.configs['lr_scheduler']='fix'
            self.configs['binary_data']=False
            self.configs['weight_decay']=1e-5
            self.configs['alpha']=2
            self.configs['gamma']=10
            self.configs['scheduler_warmup_epoch']=5
            self.configs['scheduler_decay_epoch']=10
            self.configs['scheduler_decay_rate']=1e-5            
        return self.configs


# In[4]:


seed_all(seed)
device = torch.device('cuda')
def FeatureSelect(X,p):
    if p==1:
        return X.toarray(),feature_names
    else:
        featurecount=int(X.shape[1]*p)
        Selectfeatureindex=[x[0] for x in (sorted(enumerate(X.sum(axis=0).tolist()[0]),key=lambda x: x[1],reverse=True))][:featurecount]
        Allfeatureindex=[i for i in range(X.shape[1])]
        featureindex=[i for i in Allfeatureindex if i not in Selectfeatureindex]
        new_x=np.delete(X.toarray(),featureindex,axis=1)
        new_featurename=[feature_names[i] for i in Selectfeatureindex] 
        return new_x,new_featurename
def LabelSelect(y):
    b=[]
    new_labelname=[i for i in label_names]
    for i in range(y.shape[1]):
        if y[:,i].sum()<=20:
            b.append(i)
            new_labelname.remove(label_names[i])
    new_y=np.delete(y.toarray(),b,axis=1)
    return new_y,new_labelname
def macro_averaging_auc(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(l)
    q = np.sum(Y, 0)

    zero_column_count = np.sum(q == 0)
#     print(f"all zero for label: {zero_column_count}")
    r, c = np.nonzero(Y)
    for i, j in zip(r, c):
        p[j] += np.sum((Y[ : , j] < 0.5) * (O[ : , j] <= O[i, j]))

    i = (q > 0) * (q < n)

    return np.sum(p[i] / (q[i] * (n - q[i]))) / l
def hamming_loss(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2
    l = (Y.shape[1] + P.shape[1]) // 2

    s1 = np.sum(Y, 1)
    s2 = np.sum(P, 1)
    ss = np.sum(Y * P, 1)

    return np.sum(s1 + s2 - 2 * ss) / (n * l)
def one_error(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2

    i = np.argmax(O, 1)

    return np.sum(1 - Y[range(n), i]) / n
def ranking_loss(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(n)
    q = np.sum(Y, 1)

    r, c = np.nonzero(Y)
    for i, j in zip(r, c): 
        p[i] += np.sum((Y[i, : ] < 0.5) * (O[i, : ] >= O[i, j]))

    i = (q > 0) * (q < l)

    return np.sum(p[i] / (q[i] * (l - q[i]))) / n
def micro_f1(Y, P, O):
    return f1_score(Y, P, average='micro')
def macro_f1(Y, P, O):
    return f1_score(Y, P, average='macro')
def eval_metrics(mod, metrics, datasets, idx,batch_size,device):
    res_dict = {}
    mod.eval()
    y_true_list = []
    y_scores_list = []
    test_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    for x, y in test_dataloader:
        _, y_pred=mod.predict(x)
#         pdb.set_trace()
        y_true_list.append(y.cpu().numpy())
        y_scores_list.append(y_pred.cpu().numpy())
    y_true = np.vstack(y_true_list)
    y_prob = np.vstack(y_scores_list)
    y_pred = np.round(y_prob).astype(int)
    res_dict1 = {metric.__name__: metric(y_true, y_pred,y_prob) for metric in metrics}
#         # Calculate metric.
#         res_dict1 = {metric.__name__: metric(y_true, y_pred) for metric in metrics[:2]}
#         res_dict2 = {metric.__name__: metric(y_true, y_prob) for metric in metrics[2:5]}
#         res_dict1.update(res_dict2)
#     res_dict[f'dataset_{ix}']=res_dict1
#         res_dict[f'dataset_{ix}'] = {metric.__name__: metric(y_true, y_pred) for metric in metrics[:8]}
#         res_dict[f'dataset_{ix}'] = {metric.__name__: metric(y_true, y_prob) for metric in metrics[8:9]}
    return res_dict1


# In[6]:


def calculate_probabilities(losses_dict,se):
    losses_np = np.array(list(losses_dict.values()))
    Delta=max(losses_np)/len(losses_np)
    se = se 
    N = len(losses_np)
    quantized_indices = np.ceil(losses_np / Delta).astype(int)
    probabilities = (np.exp(np.log(se) / N)) ** quantized_indices
    probabilities_dict = {key: prob for key, prob in zip(losses_dict.keys(), probabilities)}
    
    return probabilities_dict
def get_top_keys_by_value(data_dict, top_n):
    sorted_keys = sorted(data_dict, key=data_dict.get, reverse=True)[:top_n]
    return sorted_keys
class CustomDataLoader:
    def __init__(self, dataset, net, configs,sample_probabilities,warm_epoch):
        self.shuffle = False
        self.dataset = dataset
        self.net = net
        self.batch_size = configs['batch_size']
        self.sample_probabilities =sample_probabilities
        self.dataset_indices=list(range(len(self.dataset)))
        self.warm_epoch=warm_epoch       
    def __iter__(self):
        if self.current_epoch < self.warm_epoch:
            batch_counter = 0
            for start_idx in range(0, len(self.dataset_indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.dataset_indices))
                batch_indices = self.dataset_indices[start_idx:end_idx]
                batch_counter += 1
                yield self.get_data_from_indices(batch_indices)
        else:
            all_indices = list(self.sample_probabilities.keys())        
            all_probabilities = np.array([self.sample_probabilities[idx] for idx in all_indices])
            non_zero_mask = all_probabilities != 0
            non_zero_sum = all_probabilities[non_zero_mask].sum()
            all_probabilities[non_zero_mask] /= non_zero_sum
            total_samples = len(self.dataset)
            num_batches = np.ceil(total_samples / self.batch_size).astype(int)

            for _ in range(num_batches):
                num_required = self.batch_size 
                if num_required > 0:
                    chosen_indices = np.random.choice(all_indices, size=num_required, replace=False, p=all_probabilities)
                else:
                    chosen_indices = []
                batch_indices = list(chosen_indices)
                yield self.get_data_from_indices(batch_indices)
    def get_data_from_indices(self, indices):
        x, y = zip(*[self.dataset[i] for i in indices])
        return indices, torch.stack(x), torch.stack(y)
    def set_epoch(self, epoch):
        self.current_epoch = epoch


# In[13]:


def training(configs,warm_epoch,se):
    net=DELA(configs).to(device)
    num_epochs =configs['epoch']
    batch_size = configs['batch_size']
    lr = 1e-4
    label_dim=configs['num_classes']
    weight_decay = 1e-4  
    betas = (0.9, 0.999) 

    optimizer = torch.optim.Adam(net.parameters(), lr=lr,  
                           betas=betas, weight_decay=weight_decay)
    writer = SummaryWriter(comment=f'{dataname}')

    sample_probabilities = {}
    para_loss={key: 0 for key in range(configs['num_ins'])}
    custom_dataloader = CustomDataLoader(train_dataset, net=net, configs=configs,sample_probabilities=sample_probabilities,warm_epoch=warm_epoch)
    validation_dataset, test_dataset_new = train_test_split(test_dataset, test_size=0.5, random_state=42)   
    test_dataloader = DataLoader(test_dataset_new, batch_size=batch_size, shuffle=False, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    best_auc = 0
    best_model_state = None
    epoch_losses_train = []

    warmup_epochs =warm_epoch
    total_steps = num_epochs * int(configs['num_ins']/batch_size)
    warmup_steps = warmup_epochs * int(configs['num_ins']/batch_size)
    global_step = 0
    def update_learning_rate(optimizer, global_step, warmup_steps=warmup_steps, base_lr=lr):
        if global_step < warmup_steps:
            lr = base_lr * (global_step / warmup_steps)
        else:
            lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr         
    for epoch in range(num_epochs+1): 
        net.train()
        batch_counter = 0   
        loss_tracker = 0.0
        loss_tracker2=0.0
        all_individual_losses = {}
        custom_dataloader.set_epoch(epoch) 
        for idx, x, y in custom_dataloader:
            optimizer.zero_grad()
            outputs = net(x)
            loss_dict = net.loss_function_train(outputs, y)
            loss = loss_dict['Loss']
            individual_losses = net._loss_per_label(outputs[configs['out_index']], y)
#             individual_losses =net.custom_multilabel_soft_margin_loss(outputs[configs['out_index']], y,  [configs['weight_list'][i] for i in idx])
            loss.backward()
            optimizer.step()
            global_step += 1
            update_learning_rate(optimizer, global_step)

            for i, prob in zip(idx, individual_losses):
                para_loss[i]=prob.item()
            current_probabilities = calculate_probabilities(para_loss,se)
            for i in idx:
                if i in current_probabilities:
                    sample_probabilities[i] = current_probabilities[i]

            custom_dataloader.sample_probabilities = sample_probabilities
            batch_counter+=1
        writer.add_scalar('train/loss', loss_tracker, epoch)
        epoch_losses_train.append(loss_tracker /batch_counter)

        net.eval()
        y_true_list = []
        y_scores_list = []
        loss_tracker = 0.0
        batch_counter = 0
        with torch.no_grad():
            for x, y in validation_dataloader:
                _, y_pred=net.predict(x)
                y_true_list.append(y.cpu().numpy())
                y_scores_list.append(y_pred.cpu().numpy())
                
                outputs = net(x)
                loss_dict = net.loss_function_train(outputs, y)
                loss = loss_dict['Loss']
                loss_tracker += loss.item()
                batch_counter += 1
        y_true = np.vstack(y_true_list)
        y_scores = np.vstack(y_scores_list)

        auc = macro_averaging_auc(y_true,y_scores, y_scores)
#         writer.add_scalar('val/auc', auc, epoch)

        if auc > best_auc:
            best_auc = auc
            best_epoch=epoch
            best_model_state = net.state_dict().copy()
    net.load_state_dict(best_model_state)
    mets = eval_metrics(net, [macro_f1, micro_f1, macro_averaging_auc, ranking_loss, hamming_loss, one_error], test_dataset_new, configs['out_index'],configs['batch_size'],torch.device('cuda'))

    return mets


# In[14]:


path_to_arff_files = ["emotions","scene","yeast", "Corel5k","rcv1subset1","rcv1subset2","rcv1subset3","yahoo-Business1","yahoo-Arts1","bibtex",'tmc2007','enron','cal500','LLOG-F']
label_counts = [6, 6,14,374,101,101,101,28,25,159,22,53,174,75]
select_feature=[1,1,1,1,0.02,0.02,0.02,0.05,0.05,1,0.01,1,1,1]

path_to_arff_files = ["emotions"]
label_counts = [6]
select_feature=[1]

for idx, dataname in enumerate(path_to_arff_files):
    path_to_arff_file = f"/home/tt/{dataname}.arff"
    X, y, feature_names, label_names = load_from_arff(
    path_to_arff_file,
    label_count=label_counts[idx],
    label_location="end",
    load_sparse=False,
    return_attribute_definitions=True
    )
    X,feature_names=FeatureSelect(X,select_feature[idx])  
    y,label_names=LabelSelect(y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    warm_epoch=3
    print(dataname)
    k_fold = IterativeStratification(n_splits=5,order=1,random_state=42)
    dicts=[]
    for idx,(train,test) in enumerate(k_fold.split(X,y)):
        train_dataset = TensorDataset(torch.tensor(X[train], device=device, dtype=torch.float),torch.tensor(y[train], device=device,dtype=torch.float))
        test_dataset = TensorDataset(torch.tensor(X[test], device=device, dtype=torch.float), torch.tensor(y[test], device=device, dtype=torch.float))    
        configs=CFG('DELA',X[train],y[train]).getconfig()
        dict_1=training(configs,warm_epoch,2)
        dicts.append(dict_1)
    averages_and_stds = {}
    for key in dicts[0].keys():
        values = [d[key] for d in dicts]
        averages_and_stds[key] = {
            'average': round(np.mean(values),4),
            'std': round(np.std(values),4)
        }
    print(averages_and_stds)


# In[ ]:




