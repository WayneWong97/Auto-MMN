# -*- coding:utf-8 -*-
"""
@author: W4yne
@file: train_search.py.py
@time: 2021/11/4 0004
"""

from models.model import SuperNet
import torch
from torch import optim
import argparse
from dataloader.data_utils import get_pair_ls
import numpy as np
from models.architect import Architect
from dataloader.data_generator import image_pair_generator
from utils.log import get_logger

def main(args):
    logger = get_logger("{}_search.log".format(args.dataset))
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = args.dataset
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size

    if dataset == 'dipart':
        class_count = 10
    elif dataset == 'cross_dipart_ppm':
        class_count = 4
    elif dataset == 'ppm':
        class_count = 10


    # split data to train/validation
    ls_raw = get_pair_ls(args, 'train')
    ls_raw_val = get_pair_ls(args, 'val')

    loss_fn = torch.nn.CrossEntropyLoss()
    model = SuperNet(device, batch_size, class_count, loss_fn).to(device)


    #weights optimizer
    w_optim = optim.SGD(model.parameters(), lr = args.lr, momentum=args.w_momentum,
                              weight_decay=args.w_weight_decay)

    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), args.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=args.alpha_weight_decay)

    architect = Architect(args, model, args.w_momentum, args.w_weight_decay)

    # training loop
    logger.info("Logger is set - searching start")
    for epoch in range(epochs):
        #training
        train(model, architect, w_optim, alpha_optim, args.lr, epoch, dataset, batch_size, ls_raw, ls_raw_val, logger, device)

        # validation
        validate(model, epoch, dataset, batch_size, ls_raw_val, logger, device)


def train(model, architect, w_optim, alpha_optim, lr, epoch, dataset, batch_size, ls_raw, ls_raw_val, logger, device):
    model.train()
    global_spt, global_qry, patch_list_spt, patch_list_qry, label_list_spt, label_list_qry, mask_list_spt, mask_list_qry \
        = image_pair_generator(dataset, 'train', batch_size, ls_raw, device)
    global_spt_val, global_qry_val, patch_list_spt_val, patch_list_qry_val, label_list_spt_val, label_list_qry_val, mask_list_spt_val, mask_list_qry_val\
        = image_pair_generator(dataset, 'val', batch_size, ls_raw_val, device)


    # phase 2. architect step (alpha)
    alpha_optim.zero_grad()
    architect.unrolled_backward(global_spt, global_qry, patch_list_spt, patch_list_qry, mask_list_spt, mask_list_qry, label_list_spt, label_list_qry, lr, w_optim, global_spt_val,
                                global_qry_val, patch_list_spt_val, patch_list_qry_val, mask_list_spt_val, mask_list_qry_val, label_list_spt_val, label_list_qry_val)
    alpha_optim.step()

    # phase 1. child network step (w)
    w_optim.zero_grad()
    loss, acc = model.loss(global_spt, global_qry, patch_list_spt, patch_list_qry, mask_list_spt, mask_list_qry, label_list_spt, label_list_qry)
    loss.backward()
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.weights(), args.w_grad_clip)
    w_optim.step()

    Loss = torch.zeros((1))
    Arc = ['global', 'self', 'cross']
    if epoch % args.print_freq == 0:
        Loss = Loss + loss.clone().detach().cpu()
        Acc = acc
        logger.info("Train Step {}/{} Loss: {} Acc: {}, ".format(epoch + 1, args.epochs, float(Loss), float(Acc)))
        a = model.alphas()
        arc = []
        for ten in a:
            arc.append(Arc[ten.argmax(-1)])
        model.print_alphas()
        logger.info("Architect: %s", str(arc))

def validate(model, epoch, dataset, batch_size, ls_raw_val, logger, device):
    model.eval()
    global_spt_val, global_qry_val, patch_list_spt_val, patch_list_qry_val, label_list_spt_val, label_list_qry_val, mask_list_spt_val, mask_list_qry_val\
        = image_pair_generator(dataset, 'val', batch_size, ls_raw_val, device)

    loss, acc = model.loss(global_spt_val, global_qry_val, patch_list_spt_val, patch_list_qry_val, mask_list_spt_val, mask_list_qry_val, label_list_spt_val, label_list_qry_val)
    if epoch % args.print_freq == 0:
        Loss = loss.clone().detach().cpu()
        Acc = acc
        logger.info("Validation Step {}/{} Loss: {} Acc: {}".format(epoch + 1, args.epochs, float(Loss), float(Acc)))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, help='device', default='cuda:0')
    argparser.add_argument('--epochs', type=int, help='epoch', default=86985)
    argparser.add_argument('--dataset', type=str, help='dataset', default='ppm')
    argparser.add_argument('--batch_size', type=int, help='batch size', default=16)
    argparser.add_argument('--seed', type=int, help='seed', default=999)
    argparser.add_argument('--lr', type=float, help='weigts learning rate', default=0.01)
    argparser.add_argument('--alpha_lr', type=float, default=0.01, help='lr for alpha')
    argparser.add_argument('--alpha_weight_decay', type=float, default=1e-3)
    argparser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
    argparser.add_argument('--w_weight_decay', type=float, default=3e-4,
                        help='weight decay for weights')
    argparser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
    argparser.add_argument('--print_freq', type=int, default=10,
                           help='print_freq')

    args = argparser.parse_args()

    main(args)
