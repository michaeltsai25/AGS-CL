import argparse
from tap import Tap
import os

# class Config(Tap):
#     run_name: str
#     group_id: str

#     gpu: bool = True
#     workers: int = 4

#     # repeat : int = 5
#     start_seed : int = 0
#     end_seed : int = 5
#     run_seed : int = 0
#     val_size: int = 256
#     lr: float = 1e-3
#     scheduler : bool = False
#     nepoch: int = 5
#     val_check_interval: int = 300
#     batch_size: int = 256
#     train_percent_check: float = 1.

#     # 1 layer is either a weight layer of biais layer, the count starts from the out side of the DNN
#     # For multihead DNNs the heads are not taken into account in the count, the count starts from the layers below
#     ogd_start_layer : int = 0
#     ogd_end_layer : int = 1e6

#     memory_size: int = 100
#     hidden_dim: int = 100
#     pca: bool = False
#     subset_size : float = None

#     # AGEM
#     agem_mem_batch_size : int = 256
#     no_transfer : bool = False

#     n_permutation : int = 0
#     n_rotate : int = 0
#     rotate_step : int = 0
#     is_split : bool = False
#     data_seed : int = 2

#     toy: bool = False
#     ogd: bool = False
#     ogd_plus: bool = False

#     no_random_name : bool = False

#     project : str = "iclr-2021-cl-prod"
#     wandb_dryrun : bool = False
#     wandb_dir : str = "scratch"
#     # wandb_dir : str = "/scratch/thang/iclr-2021/wandb-offline"
#     dataroot : str = os.path.join("scratch", "datasets")

#     is_split_cub : bool = False

#     # Regularisation methods
#     reg_coef : float = 0.

#     agent_type : str = "ogd_plus"
#     agent_name : str = "OGD"
#     model_name : str = "MLP"
#     model_type : str = "mlp"

#     # Stable SGD
#     dropout : float = 0.
#     gamma : float = 1.
#     is_stable_sgd : bool = False


#     # Other :
#     momentum : float = 0.
#     weight_decay : float = 0.
#     print_freq : float = 100

#     no_val : bool = False
        
def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument("--rotations", type=list, default=[])
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
    parser.add_argument('--batch_size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--rho', default=0.3, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--eta', default=0.8, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--nu', default='0.1', type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=0, type=float, help='groupsparse parameter')
    
    parser.add_argument('--memory_size', default=100, type=int, help='memory size')
    parser.add_argument('--pca', default=False, type=bool, help='Use PCA')

    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')
    
    parser.add_argument('--is_split', type=bool, default=False, help='(default=%(default)s)')
    parser.add_argument('--is_split_cub', type=bool, default=False, help='(default=%(default)s)')
    
    parser.add_argument('--ogd', type=bool, default=False, help='(default=%(default)s)')
    parser.add_argument('--ogd_plus', type=bool, default=True, help='(default=%(default)s)')
    
    parser.add_argument('--gpu', type=bool, default=False, help='(default=%(default)s)')
    
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                            help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)
    parser.add_argument('--force_out_dim', type=int, default=2,
                        help="Set 0 to let the task decide the required output dimension", required=False)
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...", required=False)
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100", required=False)
    parser.add_argument('--first_split_size', type=int, default=2, required=False)
    parser.add_argument('--other_split_size', type=int, default=2, required=False)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,"
                            "6 ...] -> [0,1,2 ...]", required=False)
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training", required=False)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits", required=False)
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits", required=False)
    parser.add_argument('--schedule', nargs="+", type=int, default=[5],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number "
                            "is the end epoch", required=False)
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).", required=False)
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set", required=False)
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring "
                            "the upperbound performance.", required=False)
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new "
                            "categories.", required=False)
    
    args = parser.parse_args()
    return args