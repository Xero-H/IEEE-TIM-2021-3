from params.params_init import *
from dataset.data_choose import *
from dataset.data_proc import *
from sklearn.metrics import f1_score


import torch.nn as nn

from train_test.train_test_proc import *
from data_plot.matplotlib_plot import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
dataset_y_label = ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
                       'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping']
def plot(cm, title, class_data, thresh, path,color):
    # plt.figure(figsize=(9, 9))

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(color))
    plt.title(label=title, fontsize=10)
    plt.colorbar()

    classes = class_data
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=6)
    plt.yticks(tick_marks, classes, rotation=45, fontsize=6)
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)

    x = 0
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            x += cm[i][j]
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j]) > 0:
                plt.text(j, i, format((cm[i][j]/x)*100, '.2f') + '%' + '\n' + str(cm[i][j]),
                ha="center", va="center", size=6,
                color="white" if cm[i][j] >= thresh else "black")

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率
    # plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    args = parse_param()
    print_param(args)

    test_loss_list = []
    test_acc_list = []
    test_time_elapsed_list = []

    test_x_list = "./dataset/" + str(args.dataset) + "/x_test.npy"
    test_y_list = "./dataset/" + str(args.dataset) + "/y_test.npy"
    # test_x_list = "./dataset/wisdm/raw/x_test.npy"
    # test_y_list = "./dataset/wisdm/raw/y_test.npy"

    HAR_List = dataset_label(args.dataset)
    print("HAR_List", HAR_List)

    data_test = HAR(test_x_list, test_y_list)
    har_test_tensor = data_test.HAR_data()

    test_loader = Data.DataLoader(dataset=har_test_tensor, batch_size=args.batch_size, shuffle=True, num_workers=5, )


    model = torch.load('./model_save/oppo/revision/net0.9215946843853821_199.pth')
    # model = torch.load('./model_save/rs_uci.pth')
# # -------------------------
#     model.module.set_width_mult(0.5)
#     #
#     loss_func = torch.nn.CrossEntropyLoss()
#
#     if args.gpu == True:
#         model = model.cuda()
#         # model = nn.DataParallel(model)
#         loss_func = loss_func.cuda()
#
#     for e in range(1):
#
#         test_loss, test_acc, cm_test = test_with_cm(model, test_loader, loss_func)
#         test_acc = float(test_acc) / len(test_loader.dataset)
#
#         test_loss_list.append(test_loss / len(test_loader.dataset))
#         test_acc_list.append(test_acc)
#         print("Test set: Epoch {}, Loss {}, Accuracy {}".format(e, test_loss / len(
#             test_loader.dataset), test_acc))
#         # plot_confusion_matrix(cm_test, HAR_List, args.dataset)
#         print("test_acc", test_acc)

#######----------------------------------------------
    for i in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]:
        model.module.set_width_mult(i)
        print(i)

        loss_func = torch.nn.CrossEntropyLoss()

        if args.gpu == True:
            model = model.cuda()
            # model = nn.DataParallel(model)
            loss_func = loss_func.cuda()

        for e in range(1):
            test_loss, test_acc, cm_test = test_with_cm(model, test_loader, loss_func)
            test_acc = float(test_acc) / len(test_loader.dataset)

            test_loss_list.append(test_loss / len(test_loader.dataset))
            test_acc_list.append(test_acc)
            print("Test set: Epoch {}, Loss {}, Accuracy {}".format(e, test_loss / len(
                test_loader.dataset), test_acc))
            print("Test set: Epoch {}, Loss {}, Accuracy {}".format(e, test_loss / len(
                test_loader.dataset), round(test_acc*100, 2)))
            # plot_confusion_matrix(cm_test, HAR_List, args.dataset)
            # plot(cm=cm_test, title='PAMAP2 (CE)', class_data=dataset_y_label, thresh=168,
            #      path='image/matrix/pamap2(CE).png', color='Blues')
    #######----------------------------------------------





