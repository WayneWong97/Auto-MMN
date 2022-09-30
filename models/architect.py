# -*- coding:utf-8 -*-
"""
@author: W4yne
@file: architect.py
@time: 2021/11/4 0004
"""

""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from dataloader.data_utils import get_pair_ls

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, args, model, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.model = model
        self.v_model = copy.deepcopy(model)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.batch_size = args.batch_size
        self.dataset = args.dataset


    def virtual_step(self, global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss, _ = self.model.loss(global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.weights(), allow_unused=True)

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.model.weights(), self.v_model.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.model.alphas(), self.v_model.alphas()):
                va.copy_(a)

    def unrolled_backward(self, global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry, xi, w_optim,global_spt_val, global_qry_val, patch_spt_val, patch_qry_val, mask_spt_val, mask_qry_val, label_list_spt_val, label_list_qry_val):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry, xi, w_optim)

        # calc unrolled loss
        loss, _ = self.v_model.loss(global_spt_val, global_qry_val, patch_spt_val, patch_qry_val, mask_spt_val, mask_qry_val, label_list_spt_val, label_list_qry_val) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.model.alphas())
        v_weights = tuple(self.v_model.weights())
        v_grads = torch.autograd.grad(loss, v_weights, allow_unused=True)
        dalpha = v_grads[len(v_grads) - len(v_alphas):]
        dw = v_weights

        hessian = self.compute_hessian(dw, global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.model.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.model.weights(), dw):
                p += eps * d
        loss, _ = self.model.loss(global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry)
        dalpha_pos = torch.autograd.grad(loss, self.model.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.model.weights(), dw):
                p -= 2. * eps * d
        loss, _ = self.model.loss(global_spt, global_qry, patch_spt, patch_qry, mask_spt, mask_qry, label_list_spt, label_list_qry)
        dalpha_neg = torch.autograd.grad(loss, self.model.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.model.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian