import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='split_cifar10_100', type=str, required=False,
                        choices=['omniglot',
                                 'split_cifar10_100',
                                 'split_cifar100',
                                 'mixture'],
                        help='(default=%(default)s)')
    parser.add_argument('--approach', default='gs', type=str, required=False,
                        choices=['ewc',
                                 'si',
                                 'lrp',
                                 'rwalk',
                                 'mas',
                                 'hat',],
                        help='(default=%(default)s)')
    
    
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--rho', default=0.3, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--eta', default=0.8, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--nu', default='0.1', type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=0, type=float, help='groupsparse parameter')

    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')
    
    parser.add_argument('--is_split', type=bool, default=False, help='(default=%(default)s)')
    parser.add_argument('--is_split_cub', type=bool, default=False, help='(default=%(default)s)')
    
    parser.add_argument('--ogd', type=bool, default=False, help='(default=%(default)s)')
    parser.add_argument('--ogd_plus', type=bool, default=True, help='(default=%(default)s)')
    
    parser.add_argument('--gpu', type=bool, default=False, help='(default=%(default)s)')

    args=parser.parse_args()
    return args

