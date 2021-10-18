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
from utils.model_profiling import *


from model.baseline_uci import ConvNet_2d_uci
from model.baseline_uci import Cnn_uci
from model.baseline_wisdm import ConvNet_2d_wisdm
from model.baseline_wisdm import Cnn_wisdm
from model.baseline_unimib import ConvNet_2d_unimib
from model.baseline_unimib import ConvNet_2d_unimib88
from model.baseline_pamap2 import ConvNet_2d_pamap2
from model.baseline_pamap2 import Cnn_pamap2
from model.baseline_oppo import ConvNet_2d_oppo
# from model.baseline_pamap2 import Cnn

from model.baseline_awn import AWLeNet5
from model.baseline_awn_uci import AWLeNet5_uci
from model.baseline_awn_wisdm import AWLeNet5_wisdm
from model.baseline_awn_unimib import AWLeNet5_unimib
from model.baseline_awn_pamap2 import AWLeNet5_pamap2
from model.baseline_awn_oppo import AWLeNet5_oppo

from model.baseline_awn_unimib import convnet

from train_test.train_test_proc import *
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


    # model = AWLeNet5()

    # model = ConvNet_uci()
    # model = AWLeNet5_uci()
    # model = AWLeNet5_pamap2()
    # model = AWLeNet5_unimib()
    # model = AWLeNet5_wisdm()
    # model = AWLeNet5_oppo()

    # model = ConvNet_2d()
    # model = ConvNet_2d_uci()
    # model = ConvNet_2d_unimib()
    model = ConvNet_2d_unimib88()
    # model = ConvNet_2d_pamap2()
    # model = Cnn_pamap2()
    # model = Cnn_wisdm()
    # model = Cnn_uci()
    # model = ConvNet_2d_wisdm()
    # model = ConvNet_2d_oppo()
    # model = Cnn()

    loss_func = torch.nn.CrossEntropyLoss()

    if args.gpu == True:
        model = model.cuda()
        model = nn.DataParallel(model)
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # profiling(model, 'cpu', [128, 9],  # #######################################
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)

    acc_best = 0

    for e in range(0, args.epoch):
        # [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        # model.module.set_width_mult(0.125)  # Set active width  ####################
        # print(model)
        train_loss, train_acc, time_elapsed = train(model, train_loader, loss_func, optimizer, args.lr, e)
        train_acc = train_acc / len(train_loader.dataset)



        print("Training set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(e, train_loss / len(
            train_loader.dataset), train_acc, time_elapsed))

        train_loss_list.append(train_loss / len(train_loader.dataset))
        train_acc_list.append(train_acc)

        # model.module.set_width_mult(0.25)  # Set active width    ####################
        valid_loss, valid_acc, valid_time_elapsed, cm_valid = test(model, valid_loader, loss_func)
        valid_acc = float(valid_acc) / len(valid_loader.dataset)

        # if acc_best < valid_acc:
        #     acc_best = valid_acc
        #     torch.save(model, "./model_save/oppo/16/net" + str(valid_acc) + "_" + str(e) + ".pth")
        #
        # if e == 130:
        #     torch.save(model, "./model_save/oppo/16/net" + str(valid_acc) + "_" + str(e) + ".pth")
        #
        # if e > (args.epoch-10):
        #     torch.save(model, "./model_save/oppo/16/net" + str(valid_acc) + "_" + str(e) + ".pth")

        print("Valid set: Epoch {}, Loss {}, Accuracy {}, Best Accuracy {}, Time Elapsed {}".format(e, valid_loss / len(
            valid_loader.dataset), valid_acc, acc_best, valid_time_elapsed))
        valid_loss_list.append(valid_loss / len(valid_loader.dataset))
        valid_acc_list.append(valid_acc)
        valid_time_elapsed_list.append(valid_time_elapsed)

    # np.save("./oppo_0125_valid_acc_list_" + str(valid_acc_list[-1])+".npy", valid_acc_list)
    # np.save("./oppo_0125_valid_loss_list_" + str(valid_loss_list[-1]) + ".npy", valid_loss_list)
    # np.save("./oppo_0125_train_loss_list_" + str(train_loss_list[-1]) + ".npy", train_loss_list)
    # np.save("./oppo_0125_valid_time_elapsed_list_" + str(valid_time_elapsed_list[-1]) + ".npy", valid_time_elapsed_list)
    #
    plot_acc_plot("acc", train_acc_list, valid_acc_list, 'train_acc', 'valid_acc')
    # # plot_confusion_matrix(cm_valid, HAR_List, "uci-har-cm")
    # plot_confusion_matrix(cm_valid, HAR_List, args.dataset)






