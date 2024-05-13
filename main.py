import argparse
import os
import importlib
import utils.models as models, utils.datasets as datasets
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack',
                        help='name of attack approach;',
                        type=str,
                        choices=['sign', 'cluster', 'reconstruction', 'completion'],
                        default='sign')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=datasets.datasets_choices,
                        default='mnist')
    parser.add_argument('--epochs',
                        help='number of epochs;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size;',
                        type=int,
                        default=128)
    parser.add_argument('--lr_passive',
                        help='learning rate for passive party;',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_active',
                        help='learning rate for active party;',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_attack',
                        help='learning rate for attacker;',
                        type=float,
                        default=0.01)
    parser.add_argument('--set_attack_epoch',
                        help='whether to set attack epoch;',
                        action='store_true',
                        default=False)
    parser.add_argument('--attack_epoch',
                        help='epoch at which attack happens;',
                        type=int,
                        default=1)
    parser.add_argument('--attack_id',
                        help='id of the attacker;',
                        type=int,
                        default=0)
    parser.add_argument('--num_passive',
                        help='number of passive parties;',
                        type=int,
                        default=1)
    parser.add_argument('--use_emb',
                        help='whether to use embedding, if not, will use gradients;',
                        action='store_true',
                        default=False)
    parser.add_argument('--attack_every_n_iter',
                        help='attack every n iterations;',
                        type=int,
                        default=100)
    parser.add_argument('--simple',
                        help='use simple model',
                        action='store_true',
                        default=False)
    parser.add_argument('--padding_mode',
                        help='using the extreme assumption that only one passive party has data and the rest are padded with random data in [0, 1);',
                        action='store_true',
                        default=False)
    parser.add_argument('--division_mode',
                        help='choose the data division mode;',
                        type=str,
                        choices=['vertical', 'random', 'imbalanced'],
                        default='vertical')
    parser.add_argument('--tsne',
                        help='whether to use tsne for attack;',
                        action='store_true',
                        default=False)
    parser.add_argument('--as_order',
                        help='whether to use order dataset;',
                        action='store_true',
                        default=False)
    parser.add_argument('--attack_model_epochs',
                        help='number of epochs for attack model (reconstruction or completion model);',
                        type=int,
                        default=5)
    parser.add_argument('--lr_attack_model',
                        help='learning rate for attack model (reconstruction or completion model);',
                        type=float,
                        default=0.1)
    parser.add_argument('--balanced',
                        help='use balanced Criteo dataset;',
                        action='store_true',
                        default=False)
    parser.add_argument('--defense',
                        help='use defense;',
                        action='store_true',
                        default=False)
    parser.add_argument('--defense_all',
                        help='use defense for all passive parties;',
                        action='store_true',
                        default=False)
    parser.add_argument('--epsilon',
                        help='epsilon for defense;',
                        type=float,
                        default=0.01)
    parser.add_argument('--round',
                        help='round for log;',
                        type=int,
                        default=0)
    parser.add_argument('--dispersion',
                        help='calculate dispersion',
                        action='store_true',
                        default=False)
    parser.add_argument('--our',
                        help='use our method',
                        action='store_true',
                        default=False)
    
    # determine whether the arguments are legal or not
    args = parser.parse_args()
    if args.set_attack_epoch and args.attack_epoch > args.epochs:
        raise ValueError('--attack_epoch should be smaller than or equals to --epochs')
    if not args.set_attack_epoch and args.attack_epoch != 1:
        raise ValueError('--attack_epoch should be 1 if not use `--set_attack_epoch`')
    if args.attack_id >= args.num_passive:
        raise ValueError('--attack_id should be smaller than --num_passive')
    if args.padding_mode and args.num_passive < 2:
        raise ValueError('--padding_mode should be used with --num_passive >= 2')
    # if args.attack_every_n_iter % 10 != 0:
    #     raise ValueError('--attack_every_n_iter should be a multiple of 10')
    if args.padding_mode and args.dataset == "criteo":
        raise ValueError("Dataset Criteo can not use padding_mode.")
    if args.num_passive != 1 and not args.padding_mode:
        if args.dataset in ['mnist', 'fashionmnist'] and args.num_passive not in [2, 4, 7]:
            raise ValueError("The number of passive parties for {} must be 1, 2, 4 or 7.".format(datasets.datasets_name[args.dataset]))
        elif args.dataset in ['cifar10', 'cifar100'] and args.num_passive not in [2, 4, 8]:
            raise ValueError("The number of passive parties for {} must be 1, 2, 4 or 8.".format(datasets.datasets_name[args.dataset]))
        elif args.dataset == "criteo" and args.num_passive != 3:
            raise ValueError("The number of passive parties for {} must be 1 or 3.".format(datasets.datasets_name[args.dataset]))
    if args.balanced and args.dataset != "criteo":
        raise ValueError("{} dataset should not use --balanced.".format(datasets.datasets_name[args.dataset]))
    if args.tsne and args.attack != 'cluster':
        raise ValueError("--tsne should be used with --attack='cluster'")
    if args.use_emb and args.attack != 'cluster':
        raise ValueError("--use_emb should be used with --attack='cluster'")
    if args.division_mode in ['random', 'imbalanced'] and args.dataset not in ['mnist', 'cifar10']:
        raise ValueError("Dataset {} can not use division_mode={}.".format(datasets.datasets_name[args.dataset], args.division_mode))
    if args.defense and args.defense_all:
        raise ValueError("Can not use both --defense and --defense_all.")

    
    # change the arguments to dictionary and print
    print('Arguments:')
    args_vars = vars(args)
    format_args = '\t%' + str(max([len(i) for i in args_vars.keys()])) + 's : %s'
    for pair in sorted(args_vars.items()): print(format_args % pair)

    # create a log directory
    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    if args.dataset == "criteo":
        filedir = "balanced" if args.balanced else "imbalanced"
        log_dir = os.path.join(dir, "log", args.attack, args.dataset, filedir)
    else:
        log_dir = os.path.join(dir, "log", args.attack, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create a data directory
    data_dir = os.path.join(dir, "data", args.attack, args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # create a label directory
    data_dir = os.path.join(dir, "label_guess", args.attack, args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # load dataset
    dataset_path = os.path.join(dir, 'dataset')
    if args.dataset == "criteo":
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, balanced=args.balanced)
    else:
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True, transform=datasets.transforms_default[args.dataset])
    if args.as_order:
        data_train = order_dataset(args.dataset, data_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)

    if args.dataset == "criteo":
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, balanced=args.balanced)
    else:
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, transform=datasets.transforms_default[args.dataset])
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    # NOTE: not change test dataset format, it still out of order.

    # load model
    if args.simple:
        entire_model = models.entire_simple[args.dataset](num_passive=args.num_passive, padding_mode=args.padding_mode, division_mode=args.division_mode)
    else:
        entire_model = models.entire[args.dataset](num_passive=args.num_passive, padding_mode=args.padding_mode, division_mode=args.division_mode)

    # load attacker
    attacker_path = 'attackers.%s' % args.attack
    attacker = getattr(importlib.import_module(attacker_path), 'Attacker')

    # call trainer
    t = attacker(args, entire_model, dataloader_train, dataloader_test)  # passive_model, active_model,
    t.train()
    # t.test()


def order_dataset(dataset, data):
    # order dataset by classes
    num_classes = datasets.datasets_classes[dataset]
    data_ordered = []
    if dataset in ['mnist', 'fashionmnist', 'cifar10', 'cifar100']:
        class_idx = [[] for _ in range(num_classes)]
        for i in range(len(data)):
            class_idx[data[i][1]].append(i)

        while True:
            num_empty = 0
            for i in range(num_classes):
                if len(class_idx[i]) == 0:
                    num_empty += 1
            if num_empty == num_classes:
                break

            for i in range(num_classes):
                if len(class_idx[i]) == 0:
                    continue
                else:
                    data_ordered.append(data[class_idx[i][0]])
                    class_idx[i].pop(0)
    else:
        raise ValueError('this dataset should not use --as_order.')
    
    return data_ordered


if __name__ == '__main__':
    main()