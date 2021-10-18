from torch.autograd import Variable
import torch
import time
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score



def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))  # 30  40
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, epoch, learning_rate):
#     lr = learning_rate * (0.2 ** (epoch // 35))  # 30  40
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def train(train_model, train_loader, criterion, optimizer, learning_rate, epoch):
    adjust_learning_rate(optimizer, epoch, learning_rate)

    train_model.train()
    start = time.time()
    # print(model)
    epoch_loss = 0
    epoch_acc = 0
    loss_list = []

    # layer1_bn_mean_list = []
    # layer1_bn_var_list = []
    #
    # layer2_bn_mean_list = []
    # layer2_bn_var_list = []
    #
    # layer3_bn_mean_list = []
    # layer3_bn_var_list = []


    for step, (batch_x, batch_y) in enumerate(train_loader):
        inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
        labels_one_hot = to_one_hot(labels).cuda()

        optimizer.zero_grad()
        # outputs = train_model(inputs, labels, labels_one_hot, True)  ###
        outputs = train_model(inputs)
        _, preds = torch.max(outputs.data, 1)

        # # 用于统计baseline的各通道mean和var的均值
        # with open('BN_running_mean_0_025_1', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer1[1].running_mean.cpu())), f)
        # with open('BN_running_var_0_025_1', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer1[1].running_var.cpu())), f)
        # with open('BN_running_mean_0_025_2', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer2[1].running_mean.cpu())), f)
        # with open('BN_running_var_0_025_2', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer2[1].running_var.cpu())), f)
        # with open('BN_running_mean_0_025_3', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer3[1].running_mean.cpu())), f)
        # with open('BN_running_var_0_025_3', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer3[1].running_var.cpu())), f)

        # # # # 用于统计baseline的第0 1 2 3个通道的mean和var
        # with open('BN_running_mean_0_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_mean.cpu())[0], f)
        # with open('BN_running_var_0_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_var.cpu())[0], f)
        # with open('BN_running_mean_0_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_mean.cpu())[0], f)
        # with open('BN_running_var_0_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_var.cpu())[0], f)
        # with open('BN_running_mean_0_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_mean.cpu())[0], f)
        # with open('BN_running_var_0_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_var.cpu())[0], f)
        #
        # with open('BN_running_mean_1_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_mean.cpu())[1], f)
        # with open('BN_running_var_1_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_var.cpu())[1], f)
        # with open('BN_running_mean_1_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_mean.cpu())[1], f)
        # with open('BN_running_var_1_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_var.cpu())[1], f)
        # with open('BN_running_mean_1_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_mean.cpu())[1], f)
        # with open('BN_running_var_1_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_var.cpu())[1], f)
        #
        # with open('BN_running_mean_2_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_mean.cpu())[2], f)
        # with open('BN_running_var_2_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_var.cpu())[2], f)
        # with open('BN_running_mean_2_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_mean.cpu())[2], f)
        # with open('BN_running_var_2_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_var.cpu())[2], f)
        # with open('BN_running_mean_2_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_mean.cpu())[2], f)
        # with open('BN_running_var_2_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_var.cpu())[2], f)
        #
        # with open('BN_running_mean_3_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_mean.cpu())[3], f)
        # with open('BN_running_var_3_025_1', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer1[1].running_var.cpu())[3], f)
        # with open('BN_running_mean_3_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_mean.cpu())[3], f)
        # with open('BN_running_var_3_025_2', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer2[1].running_var.cpu())[3], f)
        # with open('BN_running_mean_3_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_mean.cpu())[3], f)
        # with open('BN_running_var_3_025_3', 'ab') as f:
        #     pickle.dump(list(train_model.module.layer3[1].running_var.cpu())[3], f)

        epoch_acc += torch.sum(preds == labels.data).item()
        loss = criterion(outputs, labels)
        epoch_loss += loss.detach().item()

        loss.backward()
        optimizer.step()

        print("train iteration {}, loss {}, acc {}, lr {}".format(step, loss.item(),
                                                                  torch.sum(preds == labels.data).item() / len(batch_x),
                                                                  optimizer.param_groups[0]['lr']))

    end = time.time()
    time_elapsed = end - start

    # np.save("BN_running_mean_0_025_1", layer1_bn_mean_list)
    # np.save("BN_running_var_0_025_1", layer1_bn_var_list)
    # np.save("BN_running_mean_0_025_2", layer2_bn_mean_list)
    # np.save("BN_running_var_0_025_2", layer2_bn_var_list)
    # np.save("BN_running_mean_0_025_3", layer3_bn_mean_list)
    # np.save("BN_running_var_0_025_3", layer3_bn_var_list)

    return epoch_loss, epoch_acc, time_elapsed


def test(test_model, test_loader, criterion):
    # total_num = sum(p.numel() for p in test_model.parameters())
    # trainable_num = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    # print('test_model Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

    start = time.time()
    epoch_loss = 0
    epoch_acc = 0

    raw_cm = np.array(0)

    test_model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            outputs = test_model(inputs)
            # print(outputs.data.shape)
            # print(torch.nn.functional.softmax(outputs.data, dim=1))
            _, preds = torch.max(outputs.data, 1)
            # print(preds)
            epoch_acc += torch.sum(preds == labels.data).item()
            loss = criterion(outputs, labels)
            epoch_loss += loss.detach().item()

            # raw_cm = raw_cm + confusion_matrix(labels.data.cpu(), preds.data.cpu())
            # print(confusion_matrix(labels.data.cpu(), preds.data.cpu()))

    end = time.time()
    time_elapsed = end - start

    # cm = confusion_matrix(labels.data.cpu(), preds.data.cpu())

    return epoch_loss, epoch_acc, time_elapsed, raw_cm


# def test_with_cm(test_model, test_loader, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#
#     raw_cm = np.array(0)
#
#     test_model.eval()
#     with torch.no_grad():
#         for step, (batch_x, batch_y) in enumerate(test_loader):
#             inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
#             outputs = test_model(inputs)
#             # print(outputs.data.shape)
#             # print(torch.nn.functional.softmax(outputs.data, dim=1))
#             _, preds = torch.max(outputs.data, 1)
#             epoch_acc += torch.sum(preds == labels.data).item()
#             loss = criterion(outputs, labels)
#             epoch_loss += loss.detach().item()
#             print("raw_cm.shape")
#             print(raw_cm.shape)
#             print("cm.shape")
#             print(confusion_matrix(labels.data.cpu(), preds.data.cpu()).shape)
#             raw_cm = np.add(raw_cm, confusion_matrix(labels.data.cpu(), preds.data.cpu()))
#
#             # raw_cm = raw_cm + confusion_matrix(labels.data.cpu(), preds.data.cpu())
#             # print(confusion_matrix(labels.data.cpu(), preds.data.cpu()))
#
#     # cm = confusion_matrix(labels.data.cpu(), preds.data.cpu())
#
#     return epoch_loss, epoch_acc, raw_cm


def test_with_cm(test_model, test_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    sum_preds = torch.empty(0)
    sum_labels = torch.empty(0)

    test_model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            outputs = test_model(inputs)
            # print(outputs.data.shape)
            # print(torch.nn.functional.softmax(outputs.data, dim=1))
            _, preds = torch.max(outputs.data, 1)

            epoch_acc += torch.sum(preds == labels.data).item()
            # f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro')
            # print("epoch_acc",epoch_acc)
            loss = criterion(outputs, labels)
            epoch_loss += loss.detach().item()
            sum_preds = torch.cat((sum_preds, preds.cpu()), 0)
            sum_labels = torch.cat((sum_labels, labels.cpu()), 0)
            # print("f1",f1)
            # print(torch.sum(sum_preds == sum_labels).item())

    cm = confusion_matrix(sum_labels.data.cpu(), sum_preds.data.cpu())
    # f1 = f1_score(sum_labels.cpu().numpy(), sum_preds.cpu().numpy(), average='macro')
    f1 = f1_score(sum_labels.cpu().numpy(), sum_preds.cpu().numpy(), average='weighted')
    # f1 = f1_score(sum_labels.cpu().numpy(), sum_preds.cpu().numpy(), average='micro')
    print("f1_score", f1)


    return epoch_loss, epoch_acc, cm
