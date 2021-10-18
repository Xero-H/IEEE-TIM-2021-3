import numpy as np

y_path_list = ['../dataset/UCI/y_train.npy',         # 6
               '../dataset/oppo/y_train.npy',        # 17
               '../dataset/opp/y_train.npy',        # 18
               '../dataset/pamap2/y_train.npy',     # 12
               '../dataset/unimib/y_train.npy',      # 17
               '../dataset/wisdm/y_train.npy',        # 6
               # '../dataset/DSA/y_train.npy',          # 19
               # '../dataset/USC/y_train.npy']          # 12
               ]

x_path_list = ['../dataset/UCI/x_train.npy',      # (7352, 128, 9)
               '../dataset/oppo/x_train.npy',     # (5373, 64, 113)  #(25343, 64, 107) # (4770, 64, 113)
               # '../dataset/opp/train_x.npy',      # (37196, 30, 113)
               '../dataset/pamap2/x_train.npy',  # (5568, 171, 40)
               '../dataset/unimib/x_train.npy',   # (7734, 151, 3)
               '../dataset/wisdm/x_train.npy',     # (7686, 200, 3)
               '../dataset/DSA/x_train.npy',      # (6384, 125, 45)
               '../dataset/USC/x_train.npy'       # (8768, 512, 6)
               ]

x_test_path_list = ['../dataset/UCI/x_test.npy',      # (2947, 128, 9)
                    '../dataset/oppo/x_test.npy',     # (619, 64, 113)  # (5313, 64, 107)
                    '../dataset/opp/x_test.npy',      # (7915, 30, 113)
                    '../dataset/opp/test_x.npy',      # (7915, 30, 113)
                    '../dataset/pamap2/x_test.npy',  # (2048, 171, 40)
                    '../dataset/unimib/x_test.npy',   # (4037, 151, 3)
                    '../dataset/wisdm/x_test.npy',    # (3295, 200, 3)
                    '../dataset/DSA/x_valid.npy',     # (2736, 125, 45)
                    '../dataset/USC/x_valid.npy'       # (2196, 512, 6)
                    ]


def print_dataset_arch():
    for x_path in x_path_list:
        data_x_raw = np.load(x_path)
        print(x_path)
        print(data_x_raw.shape)
    print("*"*15)

def print_dataset_x_test_arch():
    for x_test_path in x_test_path_list:
        data_y_raw = np.load(x_test_path)
        print(x_test_path)
        print(data_y_raw.shape)
    print("*" * 15)

    # (7352, 128, 9)
    # 6
    # 17 Open Door 1 Open Door 2 Close Door 1 Close Door 2 Open Fridge Close Fridge Open Dishwasher
    # Close Dishwasher Open Drawer 1 Close Drawer 1 Open Drawer 2 Close Drawer 2 Open Drawer 3 Close Drawer 3
    # Clean Table Drink from Cup Toggle Switch
    # 12
    # 17
    # 6 Walk Jog Up Down Sit Stand


def y_label_num():
    for y_path in y_path_list:
        data_y_raw = np.load(y_path)
        flatten_data_y_raw = data_y_raw.flatten()
        list_data_y_raw = flatten_data_y_raw.tolist()
        set_data_y_raw = set(list_data_y_raw)
        print(y_path)
        print(set_data_y_raw)
        print(len(set_data_y_raw))


def num_of_every_class(path='../dataset/oppo/y_train.npy', class_num=18):
    label_y = np.load(path).tolist()
    for i in range(class_num):
        print(str(i), label_y.count(i))




if __name__ == "__main__":
    # print_dataset_arch()
    # y_label_num()
    # print_dataset_x_test_arch()
    # num_of_every_class()

    # aa = np.array([1,2,3,4,5,6]).reshape(2,3)
    # print(aa)
    # print(aa[:][0])
    # print(aa[0][:])
    #
    dataset_num = np.load("../dataset/wisdm/x_train.npy")

    print(dataset_num.shape)
    print("###0")
    print(len(dataset_num[0]))

    for i in range(len(dataset_num[0])):
        print(i)
        print(dataset_num[0][i])

    print("###1")
    for i in range(len(dataset_num[0])):
        print(i)
        print(dataset_num[1][i])







