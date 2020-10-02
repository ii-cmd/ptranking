#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description

"""

import torch
import torch.nn.functional as F

from ptranking.base.ranker import NeuralRanker

from ptranking.ltr_global import global_gpu as gpu, global_device as device

EPS = 1e-20

class Fast_MDPRank(NeuralRanker):
    '''
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None):
        super(Fast_MDPRank, self).__init__(id='FastMDPRank', sf_para_dict=sf_para_dict)
        self.temperature = model_para_dict['temperature']

    def inner_train(self, batch_preds, batch_stds, **kwargs):
        '''
        The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        :return:
        '''

        unif = torch.rand(batch_preds.size())  # [num_samples_per_query, ranking_size]
        if gpu: unif = unif.to(device)

        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

        batch_preds = (batch_preds + gumbel) / self.temperature

        # todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
        # c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        batch_loss = torch.sum(-torch.sum(F.softmax(batch_stds, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss
