import numpy as np


def one_hot_cancel(load_path, save_path):  # 取消独热编码
    data_y_raw = np.load(load_path)
    num = []
    for x in data_y_raw:
        num.append(np.argmax(x))
    data_y = np.array(num)
    np.save(save_path, data_y)


if __name__ == "__main__":
    data_x_raw = np.load('./UCI/x_test.npy')
    data_y_raw = np.load('./UCI/y_test.npy')
    print(data_x_raw.shape)
    print(data_y_raw.shape)
    # one_hot_cancel('./UCI/y_test.npy', '')


