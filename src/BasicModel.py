'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: BasicModel.py
- A basic model framework for all models.
- The function 'select_interest()' implements the proposed idea 'Soft-selection' of BaM.

Version: 1.0
***********************************************************************
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from SSLoss import SSLoss

'''
Basic model framework for all models

input:
    * item_num: number of items
    * hidden_size: size of hidden layers
    * batch_size: size of the batch
    * seq_len: length of the sequence
'''
class BasicModel(nn.Module):

    # initialization of model
    def __init__(self, item_num, hidden_size, batch_size, seq_len=50):
        super(BasicModel, self).__init__()
        self.name = 'base'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.item_num = item_num
        self.seq_len = seq_len
        self.embeddings = nn.Embedding(self.item_num+1, self.hidden_size, padding_idx=0)
        self.interest_num = 0

    # sets used device
    def set_device(self, device):
        self.device = device

    # sets sampler for negative sample
    def set_sampler(self, sampled_n, device=None):
        
        self.is_sampler = True
        if sampled_n == 0:
            self.is_sampler = False
            return
        
        self.sampled_n = sampled_n
        
        noise = self.build_noise(self.item_num+1)
        self.sample_loss = SSLoss(noise=noise,
                                       noise_ratio=self.sampled_n,
                                       device=device
                                       )

    # initialization of weights
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    # parameters reset
    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)

    '''
    PROPOSED: Soft-selection
    
    input:
        * interest_eb: multi-interest representation
        * pos_eb: embedding of positive item
    returns:
        * selection: selected interest
    '''
    def select_interest(self, interest_eb, pos_eb):
        
        # BaM: soft selection of interest based on correlation score
        if self.selection == 'bam':
            corr = torch.matmul(interest_eb, torch.reshape(pos_eb, (-1, self.hidden_size, 1))) # inner-product of multi-interest representation and embediding of positive item (Equation (8))
            prob = F.softmax(torch.reshape(corr, (-1, self.interest_num)), dim=-1) # probability of interest selection 
            selected_index = torch.multinomial(prob, 1).flatten() # selecting an interest based on the probability
            selection = torch.reshape(interest_eb, (-1, self.hidden_size))[
                        (selected_index + torch.arange(pos_eb.shape[0], device=interest_eb.device) * self.interest_num).long()]
          
        # hard selection from previous methods 
        else: 
            corr = torch.matmul(interest_eb, torch.reshape(pos_eb, (-1, self.hidden_size, 1))) # inner-product of multi-interest representation and embediding of positive item
            prob = F.softmax(torch.reshape(corr, (-1, self.interest_num)), dim=-1) # probability of interest selection
            selected_index = torch.argmax(prob, dim=-1) # hard interest selection using argmax
            selection = torch.reshape(interest_eb, (-1, self.hidden_size))[
                        (selected_index + torch.arange(pos_eb.shape[0], device=interest_eb.device) * self.interest_num).long()]
            
        return selection

    # calculates recommendation score for all items given multi-interest representation
    def calculate_score(self, interest_eb):
        all_items = self.embeddings.weight
        scores = torch.matmul(interest_eb, all_items.transpose(1, 0)) # [b, n]
        return scores

    # caculcates sampled softmax loss
    def calculate_sampled_loss(self, selection, pos_items):
        return self.sample_loss(pos_items.unsqueeze(-1), selection, self.embeddings.weight)

    # generates noise for negative sampling
    def build_noise(self, number):
        total = number
        freq = torch.Tensor([1.0] * number).to(self.device)
        noise = freq / total 
        assert abs(noise.sum() - 1) < 0.001
        return noise