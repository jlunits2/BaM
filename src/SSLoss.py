'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Training Schemes for Balanced Interest Learning

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: SSLoss.py
- A class of sampled softmax loss.
- This code is based on the implementation of https://github.com/Stonesjtu/Pytorch-NCE/.

Version: 1.0
***********************************************************************
'''

import torch
import torch.nn as nn
BACKOFF_PROB = 1e-10


'''
Alias sampling method to speedup multinomial sampling
The alias method treats multinomial sampling as a combination of uniform sampling and bernoulli sampling. It achieves significant acceleration when repeatedly sampling from the save multinomial distribution.
Refs: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

input:
    * probs: the probability density of desired multinomial distribution
returns:
    * samples from the probability
    '''
class AliasMultinomial(torch.nn.Module):
    
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        assert abs(probs.sum().item() - 1) < 1e-5, 'The noise distribution must sum to 1'

        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    
    # Draw N(size) samples from multinomial 
    def draw(self, *size):
        
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)
    
    
"""
Sampled softmax approximation

inputs:
    * noise: the distribution of noise
    * noise_ratio: $\frac{#noises}{#real data samples}$
    * target: the supervised training label
returns:
    * the scalar loss ready for backward
"""
class SSLoss(nn.Module):
    
    # initialization
    def __init__(self,
                 noise,
                 noise_ratio=100,
                 device=None
                 ):
        super(SSLoss, self).__init__()
        self.device = device
        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        self.update_noise(noise)

        self.noise_ratio = noise_ratio
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    # udate the noise
    def update_noise(self, noise):
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()
        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

    # compute the loss with output and the desired target
    def forward(self, target, input, embs):
        batch = target.size(0)
        max_len = target.size(1)

        noise_samples = torch.arange(embs.size(0)).to(self.device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1) if self.noise_ratio == 1 else self.get_noise(batch, max_len)

        logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
        logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

        logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
        logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

        logit_target_in_model, logit_noise_in_model = self.get_score(target, noise_samples, input, embs)

        loss = self.sampled_softmax_loss(
                logit_target_in_model, logit_noise_in_model,
                logit_noise_in_noise, logit_target_in_noise,
            )

        return loss.mean()

    # Generate noise samples from noise distribution
    def get_noise(self, batch_size, max_len):
        noise_size = (batch_size, max_len, self.noise_ratio)
        noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)
        noise_samples = noise_samples.contiguous()
        return noise_samples

    # Get the target and noise score.
    def get_score(self, target_idx, noise_idx, input, embs):
        original_size = target_idx.size()

        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)
        target_batch = embs[target_idx]
        target_score = torch.sum(input * target_batch, dim=1) # N X E * N X E

        noise_batch = embs[noise_idx]  # Nr X H
        noise_score = torch.matmul(
            input, noise_batch.t()
        )
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    # Compute the sampled softmax loss based on the tensorflow's impl
    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        ori_logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)
        logits = ori_logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        loss = self.ce(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view_as(labels)

        return loss