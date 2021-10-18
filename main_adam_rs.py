"""
#Filename:	main_adam.py
#Author:	Wang Xing
#Date:
"""

import numpy as np
import torch
import torch.utils.data as Data

from params.params_init import *
from dataset.data_proc import *
from dataset.data_choose import *

from model.baseline_awn_uci import AWLeNet5_uci
from model.baseline_awn_wisdm import AWLeNet5_wisdm
from model.baseline_awn_pamap2 import AWLeNet5_pamap2
from model.baseline_awn_unimib import AWLeNet5_unimib
from model.baseline_awn_oppo import AWLeNet5_oppo

from train_test.train_rs import *
from data_plot.matplotlib_plot import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


if __name__ == "__main__":
    args = parse_param()
    print_param(args)

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    valid_time_elapsed_list = []

    train_x_list, train_y_list, valid_x_list, valid_y_list = dataset_choose(args.dataset)
    print(train_x_list, train_y_list, valid_x_list, valid_y_list)

    HAR_List = dataset_label(args.dataset)

    data_train = HAR(train_x_list, train_y_list)
    har_train_tensor = data_train.HAR_data()
    data_valid = HAR(valid_x_list, valid_y_list)
    har_valid_tensor = data_valid.HAR_data()

    train_loader = Data.DataLoader(dataset=har_train_tensor, batch_size=args.batch_size, shuffle=True, num_workers=5,)
    valid_loader = Data.DataLoader(dataset=har_valid_tensor, batch_size=args.batch_size, shuffle=True, num_workers=5,)

    # model = AWLeNet5_uci()
    # model = AWLeNet5_wisdm()
    # model = AWLeNet5_pamap2()
    # model = AWLeNet5_unimib()
    model = AWLeNet5_oppo()

    loss_func = torch.nn.CrossEntropyLoss()

    if args.gpu == True:
        model = model.cuda()
        # model = nn.DataParallel(model)
        # loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    acc_best = 0

    # for e in range(0, 20):
    run_one_experiment(model, train_loader, valid_loader, 200)
    # torch.save(model, "./model_save/rs_oppo.pth")

    # model = torch.load('./model_save/rs_wisdm.pth')
    # model.set_width_mult(1)






