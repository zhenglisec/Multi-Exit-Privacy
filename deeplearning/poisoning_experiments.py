import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import itertools
import deeplearning.aux_funcs  as af
import deeplearning.model_funcs as mf
from deeplearning import network_architectures as arcs
from deeplearning.profiler import profile_sdn, profile
import numpy as np
import time
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
from deeplearning import data as DATA
from sklearn import metrics
from myart.attacks.evasion import HopSkipJump, ProjectedGradientDescent
from myart.estimators.classification import PyTorchClassifier
from myart.utils import compute_success
from foolbox.distances import l0, l1, l2, linf
def poisoning_attack(args, models_path, device='cpu'):
    sdn_training_type = 'ic_only' # IC-only training
    #sdn_training_type = 'sdn_training' # SDN training

    cnns_sdns = []

    if args.model == 'vgg':
        add_ic = args.add_ic[0]
        model_name = '{}_vgg16bn'.format(args.task)
    elif args.model == 'resnet':
        add_ic = args.add_ic[1]
        model_name = '{}_resnet56'.format(args.task)
    elif args.model == 'wideresnet':
        add_ic = args.add_ic[2]
        model_name = '{}_wideresnet32_4'.format(args.task)
    elif args.model == 'mobilenet':
        add_ic = args.add_ic[3]
        model_name = '{}_mobilenet'.format(args.task)

    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))
    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        cnns_sdns.append('None')
        
    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

    thds_path = 'networks/0/target/' + model_name + '_sdn/threshold/' + sdn_training_type + '/thds.txt'
    with open(thds_path, "r") as f:  # 打开文件
        confidence_thresholds = f.read()  # 读取文件
        confidence_thresholds = str(confidence_thresholds).split()
        confidence_thresholds = [float(x) for x in confidence_thresholds]
    print(confidence_thresholds)

    for model_idx, model_name in  enumerate(cnns_sdns):
        print(f'------------------model: {model_name}, num_exits:{model_idx+1}-------------------')
        if 'cnn' in model_name:
            cnn_model, cnn_params = arcs.load_model(models_path, model_name, epoch=-1)
            MODEL = cnn_model.to(device)
            #print(cnn_params)

            dataset = af.get_dataset(cnn_params['task'], batch_size=100)
            if args.mode == 'target':
                print('load target_dataset ... ')
                train_loader = dataset.target_train_loader
                test_loader = dataset.target_test_loader
            elif args.mode == 'shadow':
                print('load shadow_dataset ...')
                train_loader = dataset.shadow_train_loader
                test_loader = dataset.shadow_test_loader

        elif 'sdn' in model_name:
            sdn_model, sdn_params = arcs.load_model(models_path, model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            MODEL = sdn_model.to(device)    
            #print(sdn_params)

            # to test early-exits with the SDN
            dataset = af.get_dataset(sdn_params['task'], 100)
            if args.mode == 'target':
                print('load target_dataset ...')
                one_batch_train_loader = dataset.target_train_loader
                one_batch_test_loader = dataset.target_test_loader
                train_loader, test_loader = one_batch_train_loader, one_batch_test_loader
            elif args.mode == 'shadow':
                print('load shadow_dataset ...')
                one_batch_train_loader = dataset.shadow_train_loader
                one_batch_test_loader = dataset.shadow_test_loader
                train_loader, test_loader = one_batch_train_loader, one_batch_test_loader

            
            #confidence_thresholds = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] # search for the confidence threshold for early exits
            MODEL.forward = MODEL.early_exit
            MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds
            print(f'--------------theshold:{MODEL.confidence_threshold}----------------')
        else:
            continue
        

        #######################
        ARTclassifier = PyTorchClassifier(
                model=MODEL.eval(),
                clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=(3, dataset.img_size, dataset.img_size),
                nb_classes=dataset.num_classes,
            )
        L0_dist, L1_dist, L2_dist, Linf_dist, success = [], [], [], [], []
        DATA_ADV = None
        if args.adv_type == 'PGD':
            Attack = ProjectedGradientDescent(estimator=ARTclassifier, norm=2, eps=0.3, eps_step=0.1, max_iter=100, num_random_init=0, batch_size=100, targeted=False)
        elif args.adv_type == 'HopSkipJump':
            Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, norm=2, max_iter=10, max_eval=100, init_size = 50, init_eval = 25)
        for loader_idx, data_loader in enumerate([train_loader, test_loader]):
            for data_idx, (data, target) in enumerate(data_loader):
                print(data.size)
                data = np.array(data)  

                data_adv = Attack.generate(x=data, y=None) 
                data_adv = np.array(data_adv)

                success = compute_success(ARTclassifier, data, [target.item()], data_adv)
                l0_dist = l0(data, data_adv)
                print(success)
                print(l0_dist)
                exit()
                
                DATA_ADV = data_adv if loader_idx == 0 and data_idx == 0 else np.concatenate((DATA_ADV, data_adv), axis=0)

                if data_idx == 9:
                    break
                # success.append(compute_success(ARTclassifier, data, [target.item()], data_adv))

                # L0_dist.append(l0(data, data_adv))
                # L1_dist.append(l1(data, data_adv))
                # L2_dist.append(l2(data, data_adv))
                # Linf_dist.append(linf(data, data_adv))

    #np.save(self.save_path + f'/{str(eps)}.npy', DATA_ADV)
    #distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=np.inf, axis=1)
    
    # print(distance)
    # print(y_truth)

    # fpr, tpr, _ = metrics.roc_curve(y_truth, distance)
    # auc = metrics.auc(fpr, tpr)
    # auc = auc if auc >= 0.5 else 1 - auc
        #######################
        



