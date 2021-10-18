from dataset.data_proc import *


import torch.nn as nn
from utils.model_profiling import *
# from utils.model_profiling_base import *
from train_test.train_test_proc import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device_ids = [0, 1]
if __name__ == "__main__":
    test_loss_list = []
    test_acc_list = []
    test_time_elapsed_list = []


    model = torch.load("./model_save/wisdm/rs_1/net0.9854413102820746_199.pth")
    model.module.set_width_mult(0.125)





    # profiling(model, 'gpu', [128, 9],  # UCI
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)
    # profiling(model, 'gpu', [128, 9],  # UCI
    #           1, [1,1,1,1],
    #           True)

    profiling(model, 'gpu', [200, 3],  # wisdm
              1, [1,1,1,1],
              True)

    # profiling(model, 'gpu', [171, 40],  # pamap2
    #           1, [1,1,1,1],
    #           True)
    #
    # profiling(model, 'gpu', [151, 3],  # unimib
    #           1, [1,1,1,1],
    #           True)
    # #
    # profiling(model, 'gpu', [40, 113],  # oppo
    #           1, [1,1,1,1,1,1],
    #           True)


