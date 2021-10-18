import argparse


def parse_param():
    """
    parse the arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=bool, default=True, help="Use gpu to accelerate")
    parser.add_argument("-dataset", type=str, default="UCI", help="Use gpu to accelerate")
    # parser.add_argument("-dataset", type=str, default="pamap2", help="Use gpu to accelerate")
    # parser.add_argument("-dataset", type=str, default="wisdm", help="Use gpu to accelerate")
    # parser.add_argument("-dataset", type=str, default="unimib", help="Use gpu to accelerate")
    # parser.add_argument("-dataset", type=str, default="oppo", help="Use gpu to accelerate")
    # parser.add_argument("-dataset", type=str, default="DSA", help="Use gpu to accelerate")
    # parser.add_argument("-dataset", type=str, default="USC", help="Use gpu to accelerate")

    parser.add_argument("-cls", type=int, default=12, help="dataset classes")
    parser.add_argument("-batch_size", type=int, default=512, help="batch size for dataloader")
    parser.add_argument("-lr", type=float, default=5e-4, help="initial learning rate") # 5e-4
    parser.add_argument("-epoch", type=int, default=200, help="training epoch")  # pamap画图用137
    parser.add_argument("-optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-active-width", type=float, default="0.25", help="active_width 0.25 0.5 0.75 1.0")
    # parser.add_argument("-decay", nargs='+', type=int, default=[50, 70, 90], help="epoch for weight decay")
    parser.add_argument('-no-prefetcher', action='store_true', default=False)
    args = parser.parse_args()

    return args


def print_param(args):
    """
    print the arguments
    """
    print("-" * 15 + "training configuration" + "-" * 15)
    print("gpu used:{}".format(args.gpu))
    print("dataset used:{}".format(args.dataset))
    print("class number:{}".format(args.cls))
    print("batch size:{}".format(args.batch_size))
    print("learning rate:{}".format(args.lr))
    # print("branch network used:{}".format(args.branch))
    print("training epoch:{}".format(args.epoch))
    # print("attention or not:{}".format(args.attention))
    print("optimizer used:{}".format(args.optimizer))
    # print("weights decay:{}-{}-{}".format(args.decay[0], args.decay[1], args.decay[2]))
    print(args.no_prefetcher)
    print(args.active_width)
    print("-" * 53)


if __name__ == "__main__":
    args = parse_param()
    print_param(args)
