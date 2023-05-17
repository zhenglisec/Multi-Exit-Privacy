import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import deeplearning.aux_funcs  as af
from deeplearning.train_networks import train_models
from deeplearning.early_exit_experiments import early_exit_experiments
import torch

from membership.label_only_purchase import label_only_attack_random_nosie
from membership.label_only import label_only_attack
from membership.white_box import white_box_attack
from membership.black_box import black_box_attack, black_box_attack_randomdefense
from membership.build_dataset import build_membership_dataset
def train_networks(args): 
    device = af.get_pytorch_device()
    models_path = 'networks/{}/{}'.format(args.seed, args.mode)
    af.create_path(models_path)
    af.set_logger('outputs/train_models'.format(args.seed))
    train_models(args, models_path, device)

def search_threshold(args):
    device = af.get_pytorch_device()
    trained_models_path = 'networks/{}/{}'.format(args.seed, args.mode)
    early_exit_experiments(args, trained_models_path, device)

def membership_analysis(args):
    print(f'--------------{args.mia_type}-------------')
    device = af.get_pytorch_device()

    trained_models_path = 'networks/{}/{}'.format(args.seed, args.mode)
    if args.mia_type == 'build-dataset':
        build_membership_dataset(args, trained_models_path, device)
    trained_models_path = 'networks/{}'.format(args.seed)
    if args.mia_type == 'white-box':
        white_box_attack(args, trained_models_path, device)
    elif args.mia_type == 'black-box':
        black_box_attack(args, trained_models_path, device)
    elif args.mia_type == 'label-only':
        if args.task in ['cifar10', 'cifar100', 'tinyimagenet']:
            label_only_attack(args, trained_models_path, device)
        elif args.task in ['purchase', 'texas', 'location']:
            label_only_attack_random_nosie(args, trained_models_path, device)
    elif args.mia_type == 'random-delay':
        black_box_attack_randomdefense(args, trained_models_path, device)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Multi-Exit Toy Example') 
    parser.add_argument('--action', type=int, default=2, metavar='S', help='0:train_networks, 1:search_threshold, 2:membership')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--mode', type=str, default='target', help=['target', 'shadow'])
    parser.add_argument('--model', type=str, default='mlp_3', help=['vgg', 'resnet', 'wideresnet', 'mobilenet', 'fcn_1', 'fcn_2', 'fcn_3', 'fcn_4'])
    parser.add_argument('--task', type=str, default='location', help=['cifar10', 'cifar100', 'tinyimagenet',  'purchase', 'texas', 'location'])
    parser.add_argument('--training_type', type=str, default='sdn_training')
    parser.add_argument('--add_ic', nargs='+',
                        default=[[1, 3, 5, 7, 10],                                 
                                [[3, 7], [2, 6], [2]],                             
                                [[2, 4], [1, 3], [1]],                                        
                                [1, 3, 5, 7, 10],                                  
                                [0,1,2,3,4]              
                                ]) 

    parser.add_argument('--mia_type', type=str, default='black-box', help=['build-dataset', 'white-box', 'black-box', 'label-only', 'random-delay'])
    parser.add_argument('--gpu', type=str, default='7')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('GPU ID:', args.gpu)
    af.set_random_seeds(args.seed)
    print('Random Seed: {}'.format(args.seed))
    if args.action == 0:
        train_networks(args)
    elif args.action == 1:
        search_threshold(args)
    elif args.action == 2:
        membership_analysis(args)


