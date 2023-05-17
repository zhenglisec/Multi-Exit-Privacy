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
# from myart.attacks.evasion import HopSkipJump, ProjectedGradientDescent
# from myart.estimators.classification import PyTorchClassifier
# from myart.utils import compute_success
import pandas as pd
#from foolbox.distances import l0, l1, l2, linf
def prediction(x):

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    max_index = np.argmax(softmax, axis=-1)
    min_index = np.argmin(softmax, axis=-1)
    max_index2 = np.argsort(softmax, axis=1)[:,-2:-1]
    max_index2 = np.reshape(max_index2, (-1))
    return softmax, max_index, max_index2, min_index#, sec_index
def adversarial_attack(args, models_path, device='cpu', mode='target'):
    sdn_training_type = 'ic_only' # IC-only training
    #sdn_training_type = 'sdn_training' # SDN training

    cnns_sdns = []

    if args.model == 'vgg':
        add_ic = args.add_ic[0]
        ori_model_name = '{}_vgg16bn'.format(args.task)
    elif args.model == 'resnet':
        add_ic = args.add_ic[1]
        ori_model_name = '{}_resnet56'.format(args.task)
    elif args.model == 'wideresnet':
        add_ic = args.add_ic[2]
        ori_model_name = '{}_wideresnet32_4'.format(args.task)
    elif args.model == 'mobilenet':
        add_ic = args.add_ic[3]
        ori_model_name = '{}_mobilenet'.format(args.task)

    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))
    if sdn_training_type == 'ic_only':
        cnns_sdns.append(ori_model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        cnns_sdns.append('None')
        
    for i in range(num_exits):
        cnns_sdns.append(ori_model_name + '_sdn')

    thds_path = 'networks/0/target/' + ori_model_name + '_sdn/threshold/' + sdn_training_type + '/thds.txt'
    with open(thds_path, "r") as f:  # 打开文件
        confidence_thresholds = f.read()  # 读取文件
        confidence_thresholds = str(confidence_thresholds).split()
        confidence_thresholds = [float(x) for x in confidence_thresholds]
    print(confidence_thresholds)
    L_inf = []
    dataset = af.get_dataset(args.task, batch_size=200 if args.adv_type == 'HopSkipJump' else 1000)
    #if args.mode == 'target':
    print('load shadow_test_dataset ... ')
    #train_loader = dataset.target_train_loader
    test_loader = dataset.shadow_test_loader

    for model_idx, model_name in  enumerate(cnns_sdns):
        print(f'------------------model: {model_name}, num_exits:{model_idx+1}-------------------')
        if 'cnn' in model_name:
            cnn_model, cnn_params = arcs.load_model(models_path, model_name, epoch=-1)
            MODEL = cnn_model.to(device)
            #print(cnn_params)

            # dataset = af.get_dataset(cnn_params['task'], batch_size=10)
            #if args.mode == 'target':
            # print('load target_dataset ... ')
            # train_loader = dataset.target_train_loader
            # test_loader = dataset.target_test_loader
            # elif args.mode == 'shadow':
            #     print('load shadow_dataset ...')
            #     train_loader = dataset.shadow_train_loader
            #     test_loader = dataset.shadow_test_loader

        elif 'sdn' in model_name:
            sdn_model, sdn_params = arcs.load_model(models_path, model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            MODEL = sdn_model.to(device)    
            #print(sdn_params)

            # to test early-exits with the SDN
            # dataset = af.get_dataset(sdn_params['task'], 10)
            # #if args.mode == 'target':
            # print('load target_dataset ...')
            # one_batch_train_loader = dataset.target_train_loader
            # one_batch_test_loader = dataset.target_test_loader
            # train_loader, test_loader = one_batch_train_loader, one_batch_test_loader
            # elif args.mode == 'shadow':
            #     print('load shadow_dataset ...')
            #     one_batch_train_loader = dataset.shadow_train_loader
            #     one_batch_test_loader = dataset.shadow_test_loader
            #     train_loader, test_loader = one_batch_train_loader, one_batch_test_loader

            
            #confidence_thresholds = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] # search for the confidence threshold for early exits
            MODEL.forward = MODEL.early_exit_only
            MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds
            print(f'--------------theshold:{MODEL.confidence_threshold}----------------')
        else:
            continue
        
        
        #######################
        loss_fn = nn.CrossEntropyLoss()
        ARTclassifier = PyTorchClassifier(
                model=MODEL.eval(),
                #clip_values=(0, 1),
                loss=loss_fn,
                input_shape=(3, dataset.img_size, dataset.img_size),
                nb_classes=dataset.num_classes,
                preprocessing = (dataset.mean, dataset.std)
            )
        L0_dist, L1_dist, L2_dist, Linf_dist, success = [], [], [], [], []
        DATA_ADV = None
        if args.adv_type == 'PGD':
            Attack_targeted = ProjectedGradientDescent(estimator=ARTclassifier, norm=np.inf, eps=args.eps, eps_step=0.1, max_iter=100, num_random_init=0, batch_size=1000, targeted=True, verbose=False)
            Attack_untargeted = ProjectedGradientDescent(estimator=ARTclassifier, norm=np.inf, eps=args.eps, eps_step=0.1, max_iter=100, num_random_init=0, batch_size=1000, targeted=False, verbose=False)
        elif args.adv_type == 'HopSkipJump':
            Attack_targeted = HopSkipJump(classifier=ARTclassifier, targeted =True, norm=np.inf, max_iter=50, max_eval=1000, init_size = 50, init_eval = 25, verbose=False)
            Attack_untargeted = HopSkipJump(classifier=ARTclassifier, targeted =False, norm=np.inf, max_iter=50, max_eval=1000, init_size = 50, init_eval = 25, verbose=False)
        #for loader_idx, data_loader in enumerate([train_loader, test_loader]):
        
        
        for data_idx, (data, target) in enumerate(test_loader):
            data = np.array(data)  

            logit = ARTclassifier.predict(data)
    
            softmax, max_pred, max_pred2, min_pred = prediction(logit)
            #print(softmax, max_pred, max_pred2, min_pred)
            
            data_adv_Most = Attack_targeted.generate(x=data, y=max_pred2) 
            data_adv_Least = Attack_targeted.generate(x=data, y=min_pred) 

            data_adv_untargeted = Attack_untargeted.generate(x=data, y=None) 
            #logit = ARTclassifier.predict(data_adv)
            #softmax_adv, pred_adv  = prediction(logit)

            success_Most = compute_success(ARTclassifier, data, max_pred2, data_adv_Most, batch_size=1000)
            success_Least = compute_success(ARTclassifier, data, min_pred, data_adv_Least, batch_size=1000)
            success_untargeted = compute_success(ARTclassifier, data, max_pred, data_adv_untargeted, batch_size=1000)
            
            
            print(f'dataset_size: 1000, ASR_most_likely_case: {success_Most},  ASR_least_likely_case: {success_Least}, ASR_untargeted: {success_untargeted}')
            #print(L_inf)
            #exit()
            #DATA_ADV = data_adv if loader_idx == 0 and data_idx == 0 else np.concatenate((DATA_ADV, data_adv), axis=0)
            if args.adv_type == 'HopSkipJump':
                l_inf_most = np.linalg.norm((data_adv_Most - data).reshape((data.shape[0], -1)), ord=np.inf, axis=1)
                l_inf_least = np.linalg.norm((data_adv_Least - data).reshape((data.shape[0], -1)), ord=np.inf, axis=1)
                l_inf_untargeted = np.linalg.norm((data_adv_untargeted - data).reshape((data.shape[0], -1)), ord=np.inf, axis=1)
                for tdx in range(len(l_inf_most)):
                    L_inf.append({'model_name':model_name, 'data_idx':tdx, 'l_inf_most': l_inf_most[tdx], 'l_inf_least': l_inf_least[tdx], 'l_inf_untargeted': l_inf_untargeted[tdx]})

            break
            # success.append(compute_success(ARTclassifier, data, [target.item()], data_adv))

                # L0_dist.append(l0(data, data_adv))
                # L1_dist.append(l1(data, data_adv))
                # L2_dist.append(l2(data, data_adv))
                # Linf_dist.append(linf(data, data_adv))
    if args.adv_type == 'HopSkipJump':
        df = pd.DataFrame()
        #if args.advOne_metric == 'AUC':
        L_inf = df.append(L_inf, ignore_index=True)
        L_inf.to_csv(f'plots/attack/evasion/' + sdn_training_type + '/' + ori_model_name + f'_{mode}.csv')
        #np.save(self.save_path + f'/{str(eps)}.npy', DATA_ADV)
    #distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=np.inf, axis=1)
    
    # print(distance)
    # print(y_truth)

    # fpr, tpr, _ = metrics.roc_curve(y_truth, distance)
    # auc = metrics.auc(fpr, tpr)
    # auc = auc if auc >= 0.5 else 1 - auc
        #######################
        



