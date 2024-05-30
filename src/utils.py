'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- Utils for data preparation and evaluation.

Version: 1.0
***********************************************************************
'''


import math
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader


'''
random seed setup

input:
    * seed: seed number
'''
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
     
'''
numpy array to tensor

input:
    * var: an array
    * device: device for the tensor
returns:
    * a tensor
'''   
def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()


'''
data partition in to train, valid, and test dataset

input:
    * dataset: full dataset
returns:
    * user_train: train dataset
    * user_valid: valid dataset
    * user_test: test dataset
    * user_count: number of users
    * item_count: number of items
'''
def data_partition(dataset):
    user_count = 0
    item_count = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open('data/%s' % dataset, 'r')
    count = 0
    for line in f:
        count += 1
        splited = line.rstrip().split(' ')
        u = int(splited[0])
        i = int(splited[1])
        if len(splited) == 2:
            l = 0
        else:
            if splited[2] == 'none':
                l = -1
            else:
                l = int(splited[2])
        user_count = max(u, user_count)
        item_count = max(i, item_count)
        User[u].append(i)
    print("users:", user_count)
    print("items:", item_count)
    print('interactions:', count)
    
    for user in User:
        user_train[user] = User[user][:-2]
        user_valid[user] = User[user][:-1]
        user_test[user] = User[user]
        
    return [user_train, user_valid, user_test, user_count, item_count]


'''
data iterator class

inputs:
    * user_count: number of users
    * item_count: number of items
    * batch_size: size of the batch
    * seq_len: length of sequence
    * train_flag: indicator of train dataset
    * valid_flag: indicator of valid dataset
returns:
    * user_id_list: list of users' ids
    * item_id_list: list of positive items' ids
    * hist_item_list: list of user history
    * hist_mask_list: list of item mask
'''
class DataIterator(torch.utils.data.IterableDataset):
    
    # initialization for data iterator
    def __init__(self, data,
                 user_count,
                 item_count,
                 batch_size=128,
                 seq_len=100,
                 train_flag=1,
                 valid_flag=0,
                ):
                
        self.data = data
        self.user_count = user_count
        self.item_count = item_count
        self.batch_size = batch_size 
        self.eval_batch_size = batch_size 
        self.train_flag = train_flag
        self.valid_flag = valid_flag
        self.seq_len = seq_len
        self.index = 1


    def __iter__(self):
        return self
   
    # get next batch
    def __next__(self):
        if self.train_flag == 1:
            user_id_list = np.random.randint(1, self.user_count+1, self.batch_size)
        else:
            if self.valid_flag == 1:
                if self.index > 1000:
                    self.index = 1
                    raise StopIteration
                user_id_list = np.random.randint(1, self.user_count+1, self.batch_size)
                self.index += self.eval_batch_size
            else:
                if self.index > self.user_count:
                    self.index = 1
                    raise StopIteration
                user_id_list = range(self.index, self.index+self.eval_batch_size)
                self.index += self.eval_batch_size

        item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        
        for user_id in user_id_list:
            item_list = self.data[user_id]
            if self.train_flag == 1:
                while len(item_list) < 2:
                    user_id = np.random.randint(1, self.user_count+1)
                    item_list = self.data[user_id]
                k = random.choice(range(1, len(item_list)))
                
            else:
                if len(item_list) < 1:
                    return -1, -1, -1, -1, -1
                k = len(item_list) - 1
            item_id_list.append(item_list[k])
            if k >= self.seq_len:
                hist_item_list.append(item_list[k-self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
        
        return user_id_list, item_id_list, hist_item_list, hist_mask_list


'''
gets data iterator

inputs:
    * source: dataset
    * user_count: number of users
    * item_count: number of items
    * batch_size: size of the batch
    * seq_len: length of sequence
    * train_flag: indicator of train dataset
    * valid_flag: indicator of valid dataset
returns:
    * data iterator
'''
def get_DataLoader(source, user_count, item_count, batch_size, seq_len, train_flag=1, valid_flag=0):
    dataIterator = DataIterator(source, user_count, item_count, batch_size, seq_len, train_flag=train_flag, valid_flag=valid_flag)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)


'''
evaluation for the recommendation

inputs:
    * model: model to evaluate
    * test_loader: data iterator of test dataset
    * device: device to use
    * k: number for top-k
returns:
    * recall
    * nDCG
'''
def evaluate(model, test_loader, device, k=[10,20]):
        
    total = 0
    total_recall = [0.0] * len(k)
    total_ndcg = [0.0] * len(k)
    
    for users, pos_items, items, mask in tqdm(test_loader, desc='test'):
        
        total += 1
        
        user_eb,_  = model(to_tensor(items, device), None, to_tensor(mask, device), train=False)
        score = model.calculate_score(user_eb)[0]
        score[:, items] = 0
        s, _ = torch.max(score, dim=0)
        
        for i in range(len(k)):
            _, top = torch.topk(s, k[i])
            if pos_items[0] in top:
                total_recall[i] += 1
                total_ndcg[i] += 1.0 / math.log(torch.where(top==pos_items[0])[0].item()+2, 2)
    
    return {'recall': np.array(total_recall)/total, 'nDCG': np.array(total_ndcg)/total}


'''
saves model

inputs:
    * model: model to save
    * Path: path to save the parameters at
'''
def save_model(model, Path):
    torch.save(model.state_dict(), Path + 'model.pt')


'''
loads model

inputs:
    * model: model structure
    * Path: path to load the parameters from
'''
def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'))
    print('model loaded from %s' % path)