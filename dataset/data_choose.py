

def dataset_choose(dataset_name):
    train_x_list_path = './dataset/' + dataset_name + '/x_train.npy'
    train_y_list_path = './dataset/' + dataset_name + '/y_train.npy'
    valid_x_list_path = './dataset/' + dataset_name + '/x_valid.npy'
    valid_y_list_path = './dataset/' + dataset_name + '/y_valid.npy'

    return train_x_list_path, train_y_list_path, valid_x_list_path, valid_y_list_path


def dataset_label(dataset_name):
    if dataset_name == 'UCI':
        dataset_y_label = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

    elif dataset_name == 'oppo':
        dataset_y_label = ['Null', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2', 'Open Fridge',
                           'Close Fridge', 'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1',
                           'Open Drawer 2', 'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3',
                           'Clean Table', 'Drink from Cup', 'Toggle Switch']
    # elif dataset_name == 'oppo':
    #     dataset_y_label = ['Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2', 'Open Fridge',
    #                        'Close Fridge', 'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1',
    #                        'Open Drawer 2', 'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3',
    #                        'Clean Table', 'Drink from Cup', 'Toggle Switch']
    elif dataset_name == 'pamap2':
        dataset_y_label = ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
                           'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping']
    elif dataset_name == 'unimib':
        dataset_y_label = ['StandingUpFS', 'StandingUpFL', 'Walking', 'Runing', 'GoingUpS', 'Jumping', 'GoingDownS',
                           'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack',
                           'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft']
    elif dataset_name == 'wisdm':
        # dataset_y_label = ['Walk', 'Jog', 'Up', 'Down', 'Sit', 'Stand']
        dataset_y_label = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
    elif dataset_name == 'USC':
        dataset_y_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    elif dataset_name == 'DSA':
        dataset_y_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                           '13', '14', '15', '16', '17', '18']

    return dataset_y_label


