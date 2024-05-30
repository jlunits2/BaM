'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: MIND.py
- A backbone model MIND (Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19). Association for Computing Machinery, New York, NY, USA, 2615–2623. https://doi.org/10.1145/3357384.3357814).
- This code is based on the implementation of https://github.com/ShiningCosmos/pytorch_ComiRec.

Version: 1.0
***********************************************************************
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicModel import BasicModel


'''
Capsule network layer for MIND

input:
    * hidden_size: size of hidden layers
    * seq_len: the length of the sequence
    * interest_num: number of interests
    * routing_times: the number of routings in capsule network
    * relu_layer: whether to use a relu_layer for the model output 
returns:
    * interest_capsule: multi-interest representation
'''
class CapsuleNetwork(nn.Module):

    # initialization of model
    def __init__(self, hidden_size, seq_len, interest_num=4, routing_times=3, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size 
        self.seq_len = seq_len 
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.ReLU()
            )
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    # forward propagation of the model
    def forward(self, item_eb, mask, device):
        item_eb_hat = self.linear(item_eb)
        item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
    
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        capsule_weight = torch.randn(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device, requires_grad=False)

        for i in range(self.routing_times): 
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight) 
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))

        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)
        
        return interest_capsule


'''
MIND model

input:
    * item_num: number of items
    * hidden_size: size of hidden layers
    * batch_size: size of the batch
    * interest_num: number of interests
    * seq_len: the length of the sequence
    * routing_times: the number of routings in capsule network
    * relu_layer: whether to use a relu_layer for the model output 
    * selection: the method of interest selection
returns:
    * interest_eb: multi-interst representation
    * selection: selected interest
'''
class MIND(BasicModel):

    # initialization of model
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, routing_times=3, relu_layer=True, selection='hard'):
        super(MIND, self).__init__(item_num, hidden_size, batch_size, seq_len)
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.selection = selection

        self.capsule_network = CapsuleNetwork(self.hidden_size, self.seq_len, interest_num=self.interest_num, 
                                            routing_times=self.routing_times, relu_layer=relu_layer)
        self.reset_parameters()
        
    # forward propagation of the model
    def forward(self, item_list, pos_list, mask, train=True):

        item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            pos_eb = self.embeddings(pos_list)
        interest_eb = self.capsule_network(item_eb, mask, self.device)
        
        if not train:
            return interest_eb, None

        selection  = self.select_interest(interest_eb, pos_eb)

        return interest_eb, selection