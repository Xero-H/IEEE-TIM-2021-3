from torch.autograd import Variable
import torch
import time
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
import random
import math
from utils.getters import get_meters, get_lr_scheduler, get_optimizer
from utils.meters import ScalarMeter, flush_wm_meters

class NaNLossError(Exception):
    def __init__(self, value='ERROR: Loss is NaN'):
        self.value = value
    def __str__(self):
        return repr(self.value)

def smallest_greater(thres, list_):
    return min([x for x in list_ if x >= thres])


def forward_loss(model, criterion, input, target, meter, topk=[1, 5],
                 verbose=False):
    """ Perform one forward pass """

    # Forward
    output = model(input)
    # print("criterion(output, target)", criterion(output, target))
    loss = torch.mean(criterion(output, target))
    meter['loss'].cache(loss.item())

    # Record results
    _, preds = output.topk(max(topk))
    pred = preds.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in topk:
        correct_k = correct[:k].float().sum(0)
        error_list = list(1. - correct_k.cpu().detach().numpy())
        meter['top{}_error'.format(k)].cache_list(error_list)

    if math.isnan(loss):
        raise NaNLossError

    if verbose:
        return loss, output, preds
    return loss


def rand_width_mults(num):
    """ Generate randomly sampled width factors """

    mode = "sandwich"
    # mode = "min+n-1"
    # mode = "max+n-1"
    # mode = "n"
    min_w = 0.125  # 0.125
    max_w = 1.0

    # Sample width values using the specified strategy

    if mode == 'sandwich':
        # US-Net's scheme. See https://arxiv.org/abs/1903.05134
        train_widths = ([min_w, max_w] + list(np.random.uniform(
            min_w, max_w, num - 2)))
    elif mode == "min+n-1":
        train_widths = ([min_w] + list(np.random.uniform(
            min_w, 1, num - 1)))
    elif mode == "max+n-1":
        train_widths = ([max_w] + list(np.random.uniform(
            0, max_w, num - 1)))
    elif mode == "n":
        train_widths = (list(np.random.uniform(
            0, 1, num)))
    else:
        raise NotImplementedError('Sampling mode', mode, 'not found!')
    return train_widths


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

# [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]  UCI wisdm
# [1.0, 0.75, 0.625, 0.5, 0.25, 0.125, 0.0625] unimib
def run_one_experiment(model, train_loader, valid_loader, epoch, train_width_mults=[1.0, 0.75, 0.625, 0.5, 0.25, 0.125, 0.0625]):  # [1.0, 0.75, 0.5, 0.375, 0.25, 0.125]
    acc_best = 0

    random.seed(2019)
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = model
    model = torch.nn.DataParallel(model).cuda()

    # Init training devices
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = get_optimizer(model, "adam", 0, 5e-4)

    train_meters = get_meters('train', [1, 5], train_width_mults)
    val_meters = get_meters('val', [1, 5], train_width_mults)
    val_meters['best_val_error1'] = ScalarMeter('best_val_error1')

    time_meter = ScalarMeter('runtime')

    # Perform training
    print('Start training...')
    last_epoch = -1
    best_val_error1 = 1.
    for epoch in range(last_epoch + 1, epoch):
        t_epoch_start = time.time()
        print('\nEpoch {}/{}.'.format(epoch + 1, 200)  #####
              + ' Print format: [width factor, loss, accuracy].'
              + ' Learning rate: {}'.format(optimizer.param_groups[0]['lr']))

        # Train one epoch
        steps_per_epoch = len(train_loader.dataset) / 512
        total_steps = 200 * steps_per_epoch  #####

        train_rs(epoch, num_epochs=200, loader=train_loader, model=model, criterion=criterion, #####
                                 optimizer=optimizer, meters=train_meters, width_mults=train_width_mults,
                                 topk=[1, 5])

        valid_acc = valid(model, valid_loader)
        valid_acc = float(valid_acc) / len(valid_loader.dataset)

        if acc_best <= valid_acc:
            acc_best = valid_acc
            torch.save(model, "./model_save/oppo/revision/net" + str(valid_acc) + "_" + str(epoch) + ".pth")

        if epoch > 190:
            torch.save(model, "./model_save/oppo/revision/net" + str(valid_acc) + "_" + str(epoch) + ".pth")

        print("Valid set: Epoch {}, Accuracy {}, Best Accuracy {}".format(epoch, valid_acc, acc_best))

    return


def train_rs(epoch, num_epochs, loader, model, criterion, optimizer, meters,
          width_mults, topk):
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        input, target = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
        adjust_learning_rate(optimizer, epoch, 5e-4)

        # Train
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()

        meter_width_mults = width_mults
        train_width_mults = rand_width_mults(num=8)  # ###########################修改此处设定三明治法则


        for width_mult in sorted(train_width_mults, reverse=True):
            key = smallest_greater(width_mult, meter_width_mults)
            meter = meters[str(key)]
            # print(width_mult)

            model.module.set_width_mult(width_mult) # Set active width
            loss, output, preds = forward_loss(
                model, criterion, input, target, meter, topk, verbose=True)
            loss.backward()

        optimizer.step()

    return


def train(train_model, train_loader, criterion, optimizer, learning_rate, epoch):
    adjust_learning_rate(optimizer, epoch, learning_rate)

    train_model.train()
    start = time.time()
    # print(model)
    epoch_loss = 0
    epoch_acc = 0
    loss_list = []



    for step, (batch_x, batch_y) in enumerate(train_loader):
        inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
        labels_one_hot = to_one_hot(labels).cuda()

        optimizer.zero_grad()
        # outputs = train_model(inputs, labels, labels_one_hot, True)  ###
        outputs = train_model(inputs)
        _, preds = torch.max(outputs.data, 1)
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

    return epoch_loss, epoch_acc, time_elapsed


def valid(test_model, test_loader):  # 不统计loss
    epoch_acc = 0
    test_model.module.set_width_mult(1.0)
    test_model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            inputs, labels = Variable(batch_x).float().cuda(), Variable(batch_y).long().cuda()
            outputs = test_model(inputs)
            _, preds = torch.max(outputs.data, 1)
            epoch_acc += torch.sum(preds == labels.data).item()
    return epoch_acc


def test(test_model, test_loader, criterion):  #统计loss

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
