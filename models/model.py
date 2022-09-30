# -*- coding:utf-8 -*-
"""
@author: W4yne
@file: model.py
@time: 2021/4/28 0028
"""
import torch
from copy import deepcopy
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, desc0_g, desc1_g, weights_h):
        for i in range(6):
            delta00, delta10 = self.layers[3 * i](desc0, desc0_g), self.layers[3 * i](desc1, desc1_g)
            delta01, delta11 = self.layers[3 * i + 1](desc0, desc0), self.layers[3 * i + 1](desc1, desc1)
            delta02, delta12 = self.layers[3 * i + 2](desc0, desc1), self.layers[3 * i + 2](desc1, desc0)
            delta0 = weights_h[i][0] * delta00 + weights_h[i][1] * delta01 + weights_h[i][2] * delta02
            delta1 = weights_h[i][0] * delta10 + weights_h[i][1] * delta11 + weights_h[i][2] * delta12
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1
        # for layer, name in zip(self.layers, self.names):
        #     if name == 'cross':
        #         src0, src1 = desc1, desc0
        #     elif name == 'global':
        #         src0, src1 = desc0_g, desc1_g
        #     else:  # if name == 'self':
        #         src0, src1 = desc0, desc1
        #     delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
        #     desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        # return desc0, desc1


class SuperNet(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['global', 'self', 'cross'] * 6,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    def __init__(self, device, batch_size, class_count, loss_fn):
        super().__init__()

        self.device = device

        self.loss_fn = loss_fn

        self.config = {**self.default_config}

        self.batch_size = batch_size

        self.class_count = class_count

        self.norm1d = nn.BatchNorm1d(2048).to(self.device)

        self.net1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        ).to(self.device)

        self.net_location = nn.Sequential(
            nn.Linear(224 * 224, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256)
        ).to(self.device)

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers']).to(self.device)

        # initialize architect parameters: alphas
        self.alpha = nn.ParameterList([nn.Parameter(torch.ones(3) / 3) for i in range(6)]).to(self.device)

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

    def Resnet_backbone(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 2048)
        model = model.eval()
        model.to(self.device)
        return model

    def extract_feature(self, model, patch):
        net = self.net1
        norm = self.norm1d
        result = model(patch)
        result = norm(result)
        result = torch.flatten(result, 1)
        result = net(result)
        return result

    def location_feature(self, location):
        result = self.net_location(location)
        return result

    def feature_fusion(self, feature, location):
        return torch.cat((feature, location), 1)

    def forword(self, global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry):
        weights_h = [F.softmax(a, dim=-1) for a in self.alpha]
        vis0 = self.extract_feature(self.Resnet_backbone(), patch_spt)
        vis1 = self.extract_feature(self.Resnet_backbone(), patch_qry)
        vis0_global = self.extract_feature(self.Resnet_backbone(), global_spt)
        vis1_global = self.extract_feature(self.Resnet_backbone(), global_qry)
        loc0 = self.location_feature(mask_spt)
        loc1 = self.location_feature(mask_qry)
        desc0 = vis0 + loc0
        desc1 = vis1 + loc1
        desc0_g = vis0_global.view(self.batch_size, 256, -1)
        desc1_g = vis1_global.view(self.batch_size, 256, -1)
        desc0 = desc0.view(self.batch_size, 256, -1)
        desc1 = desc1.view(self.batch_size, 256, -1)

        desc0, desc1 = self.gnn(desc0, desc1, desc0_g, desc1_g, weights_h)

        desc0 = desc0.view(self.batch_size, self.class_count, -1)
        desc1 = desc1.view(self.batch_size, self.class_count, -1)

        return desc0, desc1

    def loss(self, global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry):
        desc0, desc1 = self.forword(global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry)
        loss = torch.zeros((1)).to(self.device)
        correct = 0
        for k in range(self.batch_size):
            sim = []
            for i in range(len(desc1[k])):
                sim.append(torch.cosine_similarity(desc1[k][i].unsqueeze(0), desc0[k], dim=1))
            sim = torch.stack(sim).to(self.device)
            loss = loss + self.loss_fn(sim, label_list_qry[k])
            cost = -sim
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            result = [label_list_spt[k][i] for i in col_ind]
            # result = col_ind
            for i in range(len(col_ind)):
                if int(label_list_qry[k][i]) == int(result[i]):
                    correct = correct + 1
        acc = correct / (len(label_list_qry) * len(label_list_qry[k]))
        return loss, acc

    def weights(self):
        return self.parameters()

    def named_weights(self):
        return self.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def print_alphas(self):
        for alpha in self.alphas():
            print(F.softmax(alpha, dim=-1))





