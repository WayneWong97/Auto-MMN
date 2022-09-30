# -*- coding:utf-8 -*-
"""
@author: W4yne
@file: loss_count.py
@time: 2021/7/8 0008
"""

import torch
from scipy.optimize import linear_sum_assignment
from dataloader.data_generator import image_pair_generator


def loss_count(model, loss_fn, batch_size, dataset, ls_raw, class_count, state, device):
    correct = 0
    loss_batch = torch.zeros((1)).to(device)
    global_spt, global_qry, patch_list_spt, patch_list_qry, label_list_spt, label_list_qry, mask_list_spt, mask_list_qry = image_pair_generator(dataset, state,
                                                                                          batch_size, ls_raw,
                                                                                          device)
    desc0, desc1 = model.forword(global_spt, global_qry, patch_list_spt, patch_list_qry,  mask_list_spt, mask_list_qry)

    desc0 = desc0.view(batch_size, class_count, -1)
    desc1 = desc1.view(batch_size, class_count, -1)

    for k in range(batch_size):
        sim = []
        for i in range(len(desc1[k])):
            sim.append(torch.cosine_similarity(desc1[k][i].unsqueeze(0), desc0[k], dim=1))
        sim = torch.stack(sim).to(device)
        cost = -sim
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        result = [label_list_spt[k][i] for i in col_ind]
        # result = col_ind
        for i in range(len(col_ind)):
            if int(label_list_qry[k][i]) == int(result[i]):
                correct = correct + 1
        loss_batch = loss_batch + loss_fn(sim, label_list_qry[k])
    acc = correct / (len(label_list_qry) * len(label_list_qry[k]))

    return loss_batch, acc
