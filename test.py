# -*- coding:utf-8 -*-
"""
@author: W4yne
@file: test.py
@time: 2021/4/28 0028
"""
from models.model import SuperNet
import torch
from torch import optim
import argparse
from dataloader.data_utils import get_pair_ls
from utils.loss_count import loss_count
import time


def main(args):
    dataset = args.dataset
    device = args.device
    epoch = args.epoch
    batch_size = args.batch_size
    if dataset == 'dipart':
        class_count = 10
    elif dataset == 'cross_dipart_ppm':
        class_count = 4
    elif dataset == 'ppm':
        class_count = 10

    ls_raw = get_pair_ls(args, 'train')
    ls_raw_test = get_pair_ls(args, 'test')
    model = SuperNet(device, batch_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    Acc = 0
    Loss = torch.zeros((1))
    MAXACC = 0
    test_ACC = []
    nummm = 10
    for j in range(epoch):
        start = time.perf_counter()
        loss_batch, acc = loss_count(model, loss_fn, batch_size, dataset, ls_raw, class_count, 'train', device)
        Loss = Loss + loss_batch.clone().detach().cpu()
        Acc = Acc + acc
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        end = time.perf_counter()

        if (j % nummm == 0):
            print('Step:', j, '\t Meta_Training_Accuracy:', (Acc / nummm), '\tLoss:', (Loss / nummm), '\tTime:', round(end - start))
            Loss = torch.zeros((1))
            Acc = 0
            ### TEST
            torch.save(model.state_dict(), 'pl_model.pkl')
            model_trained = SuperNet(device, batch_size)
            model_trained.load_state_dict(model.state_dict())
            model_trained.eval()
            temp = 0
            for t in range(10):
                _, acc_test = loss_count(model_trained, loss_fn, batch_size, dataset, ls_raw_test, class_count, 'test', device)
                temp = temp + acc_test
            acc_test = temp/10
            test_ACC.append(acc_test)
            if (acc_test > MAXACC):
                MAXACC = acc_test
            print(MAXACC)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, help='device', default='cuda:2')
    argparser.add_argument('--epoch', type=int, help='epoch', default=999999999)
    argparser.add_argument('--dataset', type=str, help='dataset', default='cross_dipart_ppm')
    argparser.add_argument('--batch_size', type=int, help='batch size', default=16)

    args = argparser.parse_args()

    main(args)
