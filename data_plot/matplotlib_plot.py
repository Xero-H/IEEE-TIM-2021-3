import matplotlib.pyplot as plt
import numpy as np


data1 = np.random.random(10)
data2 = np.random.random(10)


def plot_line(title, test_acc, label='test_acc'):
    test_acc = np.array(test_acc)
    plt.title(title)
    # plt.ylim(0.8, 1)
    plt.plot(test_acc, label=label)
    plt.legend(loc='best')
    plt.show()


def plot_acc_plot(title, train_acc, test_acc, label1='train_acc', label2='test_acc'):
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    plt.title(title)
    plt.ylim(0.5, 1)
    plt.plot(train_acc, label=label1)
    plt.plot(test_acc, label=label2)
    # plt.legend(loc='lower right')
    plt.legend(loc='best')
    plt.show()


def plot_acc_plot_8(title, line1,line2,line3,line4,line5,line6,line7,line8, #test_acc,
                    label1='UCI_baseline_0.25', label2='UCI_baseline_05',
                    label3='UCI_baseline_0.75', label4='UCI_baseline_1',
                    label5='UCI_awn_0.25', label6='UCI_awn_0.5',
                    label7='UCI_awn_0.75', label8='UCI_awn_1'):
    line1 = np.array(line1)
    line2 = np.array(line2)
    line3 = np.array(line3)
    line4 = np.array(line4)
    line5 = np.array(line5)
    line6 = np.array(line6)
    line7 = np.array(line7)
    line8 = np.array(line8)
    plt.title(title)
    # plt.ylim(0.5, 1)
    plt.plot(line1, label=label1)
    plt.plot(line2, label=label2)
    plt.plot(line3, label=label3)
    plt.plot(line4, label=label4)
    plt.plot(line5, label=label5)
    plt.plot(line6, label=label6)
    plt.plot(line7, label=label7)
    plt.plot(line8, label=label8)
    # plt.legend(loc='lower right')

    # plt.ylim(-0.2, 0.2)
    # plt.ylim(0, 0.03)
    plt.legend(loc='best')
    plt.show()


def plot_bar(title, data_array):
    plt.title(title)
    plt.bar(np.arange(len(data_array)), data_array)
    plt.tight_layout()  # 避免遮挡标题
    # plt.savefig('./plot_image/%s.png' % title)
    # plt.clf()  # 避免图像重叠
    plt.show()  # 先保存，后显示，避免保存空白图像


def plot_bar_3(data, title, cpt1, cpt2, cpt3):
    plt.subplot(411)
    plt.title(title)
    for i in range(len(data)):
        plt.plot(data[i])
    plt.subplot(412)
    plt.title('cpt1')
    plt.bar(np.arange(len(cpt1)), cpt1)
    plt.subplot(413)
    plt.title('cpt2')
    plt.bar(np.arange(len(cpt2)), cpt2)
    plt.subplot(414)
    plt.title('cpt3')
    plt.bar(np.arange(len(cpt3)), cpt3)

    plt.tight_layout()  # 避免遮挡标题
    # plt.savefig('./plot_image/%s.png' % title)
    # plt.clf()  # 避免图像重叠

    plt.show()  # 先保存，后显示，避免保存空白图像


def plot_confusion_matrix(cm, labels_name, title):
    print(cm)
    plt.figure(figsize=(13, 11))
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)    # 在特定的窗口上显示图像
    plt.title('Confusion Matrix for ' + title + ' dataset baseline', fontsize=13)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=315)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.tick_params(labelsize=11)
    iters = np.reshape([[[i, j] for j in range(len(labels_name))] for i in range(len(labels_name))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))  # 显示对应的数字
    plt.ylabel('True label', fontsize=11)
    plt.xlabel('Predicted label', fontsize=11)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_acc_plot('test', data1, data2)
    plot_bar('test2', data1)

