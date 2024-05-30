'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: ComiRec.py
- A backbone model ComiRec (Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 2942–2951. https://doi.org/10.1145/3394486.3403344).
- This code is based on the implementation of https://github.com/ShiningCosmos/pytorch_ComiRec.


Version: 1.0
***********************************************************************
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicModel import BasicModel


'''
ComiRec model

input:
    * item_num: number of items
    * hidden_size: size of hidden layers
    * batch_size: size of the batch
    * selection: the method of interest selection
    * interest_num: number of interests
    * seq_len: the length of the sequence
    * add_pos: whether to use position embedding or not
returns:
    * interest_emb: multi-interst representation
    * selection: selected interest
'''
class ComiRec(BasicModel):
    
    # initialization of model
    def __init__(self, item_num, hidden_size, batch_size, selection=True, interest_num=4, seq_len=50, add_pos=True):
        super(ComiRec, self).__init__(item_num, hidden_size, batch_size, seq_len)
        self.interest_num = interest_num
        self.num_heads = interest_num
        self.selection = selection
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.reset_parameters()

    # forward propagation of the model
    def forward(self, item_list, pos_list, mask, train=True):
        item_eb = self.embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            pos_eb = self.embeddings(pos_list)

        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        item_hidden = self.linear1(item_eb_add_pos)
        item_att_w  = self.linear2(item_hidden)
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous() 

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) 
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1) 

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1)

        interest_emb = torch.matmul(item_att_w, item_eb) # interest_emb M_u


        if not train:
            return interest_emb, None
        
        # selects a interest vector M_u^n from multi-interest
        selection = self.select_interest(interest_emb, pos_eb)

        return interest_emb, selection
