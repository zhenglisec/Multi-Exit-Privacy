from __future__ import print_function
import itertools, collections

from joblib import MemorizedResult
import deeplearning.aux_funcs  as af
import deeplearning.model_funcs as mf
from deeplearning import network_architectures as arcs
from deeplearning.profiler import profile_sdn, profile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from membership import check_and_transform_label_format, MLP_BLACKBOX, posteriorAttackModel, model_feature_len, train_mia_attack_model, test_mia_attack_model
from sklearn import metrics
# from myart.attacks.evasion import HopSkipJump, ProjectedGradientDescent
# from myart.estimators.classification import PyTorchClassifier
# from myart.utils import compute_success
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from myart.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
# from myart.attacks.evasion import HopSkipJump
import os
import pickle
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
def Attack_Dataloader(models_path, model_name, miatype, bandwidth):
    
    # if 'idx' not in miatype:
    #     filename = '1_process_4_gpu_core'
    # elif 'exit_idx' in miatype:
    #     filename = '1_process_4_gpu_core'
    # elif '1_process_4_gpu_core' in miatype:
    #     filename = '1_process_4_gpu_core'
    # elif '4_process_4_gpu_core' in miatype:
    #     filename = '4_process_4_gpu_core'
    # elif '4_process_1_gpu_core' in miatype:
    #     filename = '4_process_1_gpu_core'
    # elif 'posterior' in miatype:
    #     filename = '1_process_4_gpu_core'
    #print(filename)
    AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
    AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
    if type(AttackModelTrainSet) == np.ndarray:
        AttackModelTrainSet = AttackModelTrainSet.item()
        AttackModelTestSet = AttackModelTestSet.item()

    num_exits = AttackModelTrainSet['num_exits']
    nb_classes = AttackModelTestSet['nb_classes']

    

    number_point = int(AttackModelTrainSet['exit_idx'].shape[0]/2)
    shadow_train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['model_scores'][:number_point], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['model_loss'][:number_point], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'][:number_point], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'][:number_point], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTrainSet['exit_idx'][:number_point])).type(torch.long),
            )
    shadow_test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['model_scores'][number_point:], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['model_loss'][number_point:], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'][number_point:], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'][number_point:], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTrainSet['exit_idx'][number_point:])).type(torch.long),
            )
    target_train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTestSet['model_scores'][:number_point], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['model_loss'][:number_point], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'][:number_point], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'][:number_point], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTestSet['exit_idx'][:number_point])).type(torch.long),
            )
    target_test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTestSet['model_scores'][number_point:], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['model_loss'][number_point:], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'][number_point:], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'][number_point:], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTestSet['exit_idx'][number_point:])).type(torch.long),
            )
    shadow_train_exit_idx_loader = torch.utils.data.DataLoader(shadow_train_set, batch_size=1, shuffle=False)
    shadow_test_exit_idx_loader = torch.utils.data.DataLoader(shadow_test_set, batch_size=1, shuffle=False)
    target_train_exit_idx_loader = torch.utils.data.DataLoader(target_train_set, batch_size=1, shuffle=False)
    target_test_exit_idx_loader = torch.utils.data.DataLoader(target_test_set, batch_size=1, shuffle=False)

    return target_train_exit_idx_loader, target_test_exit_idx_loader, shadow_train_exit_idx_loader, shadow_test_exit_idx_loader, num_exits, nb_classes, 1

def label_only_attack_random_nosie(args, models_path, device='cpu', supervised=True):
    save_path = f'/.../.../'
    os.makedirs(save_path, exist_ok=True)
    max_iter = 10
    sdn_training_type = args.training_type
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
    elif args.model == 'fcn_1':
        add_ic = args.add_ic[4]
        model_name = '{}_fcn_1'.format(args.task)
    elif args.model == 'fcn_2':
        add_ic = args.add_ic[4]
        model_name = '{}_fcn_2'.format(args.task)
    elif args.model == 'fcn_3':
        add_ic = args.add_ic[4]
        model_name = '{}_fcn_3'.format(args.task)
    elif args.model == 'fcn_4':
        add_ic = args.add_ic[4]
        model_name = '{}_fcn_4'.format(args.task)

    csv_name = model_name
    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))

    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        # cnns_sdns.append('None')
        cnns_sdns.append(model_name + '_cnn')

    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

    thds_path = 'networks/'+str(args.seed)+'/target/' + model_name + '_sdn/threshold/' + sdn_training_type + '/thds.txt'
    with open(thds_path, "r") as f:  # 打开文件
        confidence_thresholds = f.read()  # 读取文件
        confidence_thresholds = str(confidence_thresholds).split()
        confidence_thresholds = [float(x) for x in confidence_thresholds]
    print(confidence_thresholds)

    Distance = []
    Final_Results = []
    for model_idx, model_name in  enumerate(cnns_sdns):

        print(f'------------------model: {model_name}, num_exits:{model_idx+1}-------------------')
        if 'cnn' in model_name:
            target_cnn_model, target_cnn_params = arcs.load_model(models_path + '/target', model_name, epoch=-1)
            target_MODEL = target_cnn_model.to(device)
            shadow_cnn_model, shadow_cnn_params = arcs.load_model(models_path + '/shadow', model_name, epoch=-1)
            target_MODEL = target_cnn_model.to(device)
            shadow_MODEL = shadow_cnn_model.to(device)

            dataset = af.get_dataset(target_cnn_params['task'], batch_size=1)
    
            print('load target_dataset ... ')
            target_train_loader = dataset.target_train_loader
            target_test_loader = dataset.target_test_loader
            print('load shadow_dataset ...')
            shadow_train_loader = dataset.shadow_train_loader
            shadow_test_loader = dataset.shadow_test_loader
            
        elif 'sdn' in model_name:
            target_sdn_model, target_sdn_params = arcs.load_model(models_path + '/target', model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            target_MODEL = target_sdn_model.to(device)    
            shadow_sdn_model, shadow_sdn_params = arcs.load_model(models_path + '/shadow', model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            shadow_MODEL = shadow_sdn_model.to(device) 

            # to test early-exits with the SDN
            dataset = af.get_dataset(target_sdn_params['task'], batch_size=1)
           
            print('load target_dataset ...')
            target_train_loader = dataset.target_train_loader
            target_test_loader = dataset.target_test_loader
            
            print('load shadow_dataset ...')
            shadow_train_loader = dataset.shadow_train_loader
            shadow_test_loader = dataset.shadow_test_loader

            # if metric == 'exit_index':
            #     target_MODEL.forward = target_MODEL.early_exit_only_index
            # elif metric == 'prediction':
            target_MODEL.forward = target_MODEL.early_exit_only_score
            target_MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds
            shadow_MODEL.forward = shadow_MODEL.early_exit_only_score
            shadow_MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds
            print(f'--------------theshold:{target_MODEL.confidence_threshold}----------------')

            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
        else:
            continue
    

        target_MODEL.eval()
        shadow_MODEL.eval()

        # MIA_list = ['label_only', 'label_only+exit_idx', 'label_only+predict_idx+1_process_4_gpu_core', 'label_only+predict_idx+4_process_4_gpu_core',
        #             'label_only+predict_idx+4_process_1_gpu_core', #'label_only+predict_idx+24_process_1_cpu_device'                 
        #             ]
        MIA_list = ['label_only', 'label_only+exit_idx']

        target_Mem_Each_Exit = [[] for _ in range(model_idx+1)]
        target_NonMem_Each_Exit = [[] for _ in range(model_idx+1)]
        target_Mem_Label_Each_Exit = [[] for _ in range(model_idx+1)]
        target_NonMem_Label_Each_Exit = [[] for _ in range(model_idx+1)]

        shadow_Mem_Each_Exit = [[] for _ in range(model_idx+1)]
        shadow_NonMem_Each_Exit = [[] for _ in range(model_idx+1)]
        shadow_Mem_Label_Each_Exit = [[] for _ in range(model_idx+1)]
        shadow_NonMem_Label_Each_Exit = [[] for _ in range(model_idx+1)]

        if model_idx == 0: 
            Num_Each_Exit = 300 
        elif model_idx in [1,2]:
            Num_Each_Exit = 300
        else:
            Num_Each_Exit = 300

        
        for mia in MIA_list:
            print(f'-------------------{mia}------------------')
            
            target_train_exit_idx_loader, target_test_exit_idx_loader, shadow_train_exit_idx_loader, shadow_test_exit_idx_loader, num_exits, nb_classes, exit_acc = Attack_Dataloader(models_path, model_name, mia, bandwidth=0.15)
            
            for idx, ((target_mem_data, target_mem_label), (target_mem_scores, target_mem_loss, target_mem_exit), (target_nonmem_data, target_nonmem_label), (target_nonmem_scores, target_nonmem_loss, target_nonmem_exit),
                        (shadow_mem_data, shadow_mem_label), (shadow_mem_scores, shadow_mem_loss, shadow_mem_exit), (shadow_nonmem_data, shadow_nonmem_label), (shadow_nonmem_scores, shadow_nonmem_loss, shadow_nonmem_exit)) in \
                            enumerate(zip(target_train_loader, target_train_exit_idx_loader, target_test_loader, target_test_exit_idx_loader, 
                            shadow_train_loader, shadow_train_exit_idx_loader, shadow_test_loader, shadow_test_exit_idx_loader)):

                # (target_mem_data, target_mem_label), (target_mem_scores, target_mem_loss, target_mem_exit), (target_nonmem_data, target_nonmem_label), (target_nonmem_scores, target_nonmem_loss, target_nonmem_exit), \
                #         (shadow_mem_data, shadow_mem_label), (shadow_mem_scores, shadow_mem_loss, shadow_mem_exit), (shadow_nonmem_data, shadow_nonmem_label), (shadow_nonmem_scores, shadow_nonmem_loss, shadow_nonmem_exit) = (target_mem_data.to(device), target_mem_label.to(device)), (target_mem_scores.to(device), target_mem_loss.to(device), target_mem_exit.to(device)), (target_nonmem_data.to(device), target_nonmem_label.to(device)), (target_nonmem_scores.to(device), target_nonmem_loss.to(device), target_nonmem_exit.to(device)), \
                #         (shadow_mem_data.to(device), shadow_mem_label.to(device)), (shadow_mem_scores.to(device), shadow_mem_loss.to(device), shadow_mem_exit.to(device)), (shadow_nonmem_data.to(device), shadow_nonmem_label.to(device)), (shadow_nonmem_scores.to(device), shadow_nonmem_loss.to(device), shadow_nonmem_exit.to(device))
                #print(idx)
               
                if len(target_Mem_Each_Exit[target_mem_exit.item()]) < Num_Each_Exit:
                    #target_MEM_data = target_mem_data.cpu().numpy() if idx == 0 else np.concatenate((target_MEM_data, target_mem_data.cpu().numpy()), axis=0)
                    #target_MEM_label = target_mem_label.cpu().numpy() if idx == 0 else np.concatenate((target_MEM_label, target_mem_label.cpu().numpy()), axis=0)
                    target_Mem_Each_Exit[target_mem_exit.item()].append(target_mem_data.cpu().numpy())
                    target_Mem_Label_Each_Exit[target_mem_exit.item()].append(target_mem_label.cpu().numpy())
                if len(target_NonMem_Each_Exit[target_nonmem_exit.item()]) < Num_Each_Exit:
                    #target_NonMEM_data = target_nonmem_data.cpu().numpy() if idx == 0 else np.concatenate((target_NonMEM_data, target_nonmem_data.cpu().numpy()), axis=0)
                    #target_NonMEM_label = target_nonmem_label.cpu().numpy() if idx == 0 else np.concatenate((target_NonMEM_label, target_nonmem_label.cpu().numpy()), axis=0)
                    #target_NoMem_Each_Exit[target_nonmem_exit.item()] = target_NoMem_Each_Exit[target_nonmem_exit.item()] + 1
                    target_NonMem_Each_Exit[target_nonmem_exit.item()].append(target_nonmem_data.cpu().numpy())
                    target_NonMem_Label_Each_Exit[target_nonmem_exit.item()].append(target_nonmem_label.cpu().numpy())
                #target_set = np.concatenate((target_MEM_data, target_NonMEM_data), axis=0)
                #train_set_labels = np.concatenate((target_MEM_label, target_NonMEM_label), axis=0)

                if len(shadow_Mem_Each_Exit[shadow_mem_exit.item()]) < Num_Each_Exit:
                    # shadow_MEM_data = shadow_mem_data.cpu().numpy() if idx == 0 else np.concatenate((shadow_MEM_data, shadow_mem_data.cpu().numpy()), axis=0)
                    # shadow_MEM_label = shadow_mem_label.cpu().numpy() if idx == 0 else np.concatenate((shadow_MEM_label, shadow_mem_label.cpu().numpy()), axis=0)
                    # shadow_Mem_Each_Exit[shadow_mem_exit.item()] = shadow_Mem_Each_Exit[shadow_mem_exit.item()] + 1
                    shadow_Mem_Each_Exit[shadow_mem_exit.item()].append(shadow_mem_data.cpu().numpy())
                    shadow_Mem_Label_Each_Exit[shadow_mem_exit.item()].append(shadow_mem_label.cpu().numpy())

                if len(shadow_NonMem_Each_Exit[shadow_nonmem_exit.item()]) < Num_Each_Exit:
                    # shadow_NonMEM_data = shadow_nonmem_data.cpu().numpy() if idx == 0 else np.concatenate((shadow_NonMEM_data, shadow_nonmem_data.cpu().numpy()), axis=0)
                    # shadow_NonMEM_label = shadow_nonmem_label.cpu().numpy() if idx == 0 else np.concatenate((shadow_NonMEM_label, shadow_nonmem_label.cpu().numpy()), axis=0)
                    # shadow_NoMem_Each_Exit[shadow_nonmem_exit.item()] = shadow_NoMem_Each_Exit[shadow_nonmem_exit.item()] + 1
                    shadow_NonMem_Each_Exit[shadow_nonmem_exit.item()].append(shadow_nonmem_data.cpu().numpy())
                    shadow_NonMem_Label_Each_Exit[shadow_nonmem_exit.item()].append(shadow_nonmem_label.cpu().numpy())

            sigma, threshold = -1, -1
            if mia == 'label_only':
            
                # shadow_MEM_data = np.array(sum(shadow_Mem_Each_Exit,[]))
                # shadow_MEM_label = np.array(sum(shadow_Mem_Label_Each_Exit, []))
                # shadow_NonMEM_data = np.array(sum(shadow_NonMem_Each_Exit, []))
                # shadow_NonMEM_label = np.array(sum(shadow_NonMem_Label_Each_Exit, []))
                
                # shadow_MEM_data = np.squeeze(shadow_MEM_data)
                # shadow_MEM_label = np.squeeze(shadow_MEM_label)
                # shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                # shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)
            
                #sigma, threshold = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label)


                target_MEM_data = np.array(sum(target_Mem_Each_Exit,[]))
                target_MEM_label = np.array(sum(target_Mem_Label_Each_Exit, []))
                target_NonMEM_data = np.array(sum(target_NonMem_Each_Exit, []))
                target_NonMEM_label = np.array(sum(target_NonMem_Label_Each_Exit, []))
                
                target_MEM_data = np.squeeze(target_MEM_data)
                target_MEM_label = np.squeeze(target_MEM_label)
                target_NonMEM_data = np.squeeze(target_NonMEM_data)
                target_NonMEM_label = np.squeeze(target_NonMEM_label)


                # accuracy, auc = inference(target_MODEL, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, sigma, threshold)
                accuracy, auc = inference_auc(dataset, target_MODEL, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label)
                print(f'Exit_index All,  Accuracy {accuracy}, AUC {auc}')
                Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':0, 'threshold_flag':True, 'accuracy_flag':True, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia})
            #elif 'sdn' in model_name:
            elif mia == 'label_only+exit_idx':
                # previous_distance_threshold_tau = 0
                previous_sigma, previous_threshold = 0, 0
                avg_acc, avg_auc = 0, 0
                for idx in range(len(target_Mem_Each_Exit)):
                    threshold_flag = True
                    accuracy_flag = True
                    # shadow_MEM_data = np.array(shadow_Mem_Each_Exit[idx])
                    # shadow_MEM_label = np.array(shadow_Mem_Label_Each_Exit[idx])
                    # shadow_NonMEM_data = np.array(shadow_NonMem_Each_Exit[idx])
                    # shadow_NonMEM_label = np.array(shadow_NonMem_Label_Each_Exit[idx])
                    
                    # shadow_MEM_data = np.squeeze(shadow_MEM_data)
                    # shadow_MEM_label = np.squeeze(shadow_MEM_label)
                    # shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                    # shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)

                    # if (len(shadow_MEM_data) == 0) or (len(shadow_NonMEM_data) == 0):
                    #     sigma, threshold = previous_sigma, previous_threshold
                    #     threshold_flag = False
                    # else:
                    #     sigma, threshold = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label)
                    #     previous_sigma, previous_threshold = sigma, threshold
                        
                    target_MEM_data = np.array(target_Mem_Each_Exit[idx])
                    target_MEM_label = np.array(target_Mem_Label_Each_Exit[idx])
                    target_NonMEM_data = np.array(target_NonMem_Each_Exit[idx])
                    target_NonMEM_label = np.array(target_NonMem_Label_Each_Exit[idx])

                    target_MEM_data = np.squeeze(target_MEM_data)
                    target_MEM_label = np.squeeze(target_MEM_label)
                    target_NonMEM_data = np.squeeze(target_NonMEM_data)
                    target_NonMEM_label = np.squeeze(target_NonMEM_label)

                    if (len(target_MEM_data) == 0) or (len(target_NonMEM_data) == 0):
                        accuracy, auc = 1, 1
                        accuracy_flag = False
                    else:
                        # accuracy, auc = inference(target_MODEL, sigma, threshold, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, sigma, threshold)
                        accuracy, auc = inference_auc(dataset, target_MODEL, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label)
                    print(f'Exit_index {idx},  Accuracy {accuracy}, AUC {auc}, threshold_flag {threshold_flag}, accuracy_flag {accuracy_flag}')
                    
                    avg_acc += accuracy
                    avg_auc += auc
                avg_acc = avg_acc/len(target_Mem_Each_Exit)
                avg_auc = avg_auc/len(target_Mem_Each_Exit)
                Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':idx, 'threshold_flag':threshold_flag, 'accuracy_flag':accuracy_flag, 'accuracy':avg_acc, 'auc':avg_auc, 'mia_type':mia})
                    
                    # fh = open(save_path + f'/{args.task}_{args.model}.txt', 'a', encoding='utf-8')
                    # straa = f'dataset:{args.task}, model:{args.model}, num_exits: {num_exits}, training_type:{sdn_training_type}, exit_index:{idx}, threshold_flag:{threshold_flag}, accuracy_flag:{accuracy_flag}, accuracy:{accuracy}, auc:{auc}, mia_type:{mia}'
                    # fh.write(straa)
                    # fh.close()
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(save_path + f'/{args.task}_{args.model}.csv')   

def robust_accs(model, inputs, labels, p, noise_samples=1000):
    robust_dist = []

    inputs_ts = torch.from_numpy(inputs)
    inputs_ts = inputs_ts.cuda()

    scores = model(inputs_ts)
    preds = scores.max(1, keepdim=True)[1]
    preds = preds.view(-1).cpu().numpy()

    correct = preds == labels

    for i in range(len(correct)):
        if correct[i]:
            noise = np.random.binomial(1, p, [noise_samples, inputs[i: i + 1].shape[-1]])
            inputs_sampled = np.tile(np.copy(inputs[i:i + 1]), (noise_samples, 1))
            inputs_noisy = np.invert(inputs[i: i + 1].astype(np.bool), out=inputs_sampled,
                                    where=noise.astype(np.bool)).astype(np.float32)
            preds = []

            bsize = 100
            num_batches = noise_samples // bsize
            for j in range(num_batches):
                inputs_noisy_bs = torch.from_numpy(inputs_noisy[j * bsize:(j + 1) * bsize])
                inputs_noisy_bs = inputs_noisy_bs.cuda()
                # print(inputs_noisy_bs)
                scores_noisy_bs = model(inputs_noisy_bs)
                preds_noisy_bs = scores_noisy_bs.max(1, keepdim=True)[1]
                preds_noisy_bs = preds_noisy_bs.view(-1).cpu().numpy()

                preds_noisy = preds_noisy_bs if j == 0 else np.concatenate((preds_noisy, preds_noisy_bs), axis=0)
                # preds.extend(sess.run(output, feed_dict={x: x_noisy[j * bsize:(j + 1) * bsize]}))

            # for idx, n in enumerate(num):
            #     if n == 0:
            #     robust_accs[idx].append(1)
            #     else:
            robust_dist.append(np.mean(preds_noisy == labels[i]))
        else:
        #   for idx in range(len(num)):
            robust_dist.append(0)
    return robust_dist
    
def search_threshold(dataset, shadow_MODEL, mem, mem_labels, non_mem, non_mem_labels):
    best_acc = -1
    best_sigmas = -1
    best_threshold = -1
    sigmas = [1. / dataset.input_size, 2. / dataset.input_size, 3. / dataset.input_size, 5. / dataset.input_size, 10. / dataset.input_size]
    for sigma in sigmas:
        mem_robust_dist = robust_accs(shadow_MODEL, mem, mem_labels, sigma)
        mem_robust_dist = np.array(mem_robust_dist)
        # print('Member:', sigma, robust_dist, len(robust_dist))

        nonmem_robust_dist = robust_accs(shadow_MODEL, non_mem, non_mem_labels, sigma)
        nonmem_robust_dist = np.array(nonmem_robust_dist)
        # print('Non-Member:', sigma, robust_dist, len(robust_dist))
    
        robust_dist = np.concatenate((mem_robust_dist, nonmem_robust_dist), axis=0)
        mem_ground = np.array([1]*len(mem_robust_dist))
        nonmem_ground = np.array([0]*len(nonmem_robust_dist))
        ground = np.concatenate((mem_ground, nonmem_ground), axis=0)
        
        for threshold in range(0, 100, 1):
            threshold = threshold/100.0
            pred = (robust_dist>threshold).astype(int)
            accuracy = accuracy_score(ground, pred)

            if accuracy > best_acc:
                best_sigmas = sigma
                best_threshold = threshold 
                best_acc = accuracy
            # print(sigma, threshold, accuracy)
        
    print(best_sigmas, best_threshold, best_acc)
    return best_sigmas, best_threshold

def inference(target_MODEL, mem, mem_labels, non_mem, non_mem_labels, sigma, threshold):
    print('Start inference...')
       
    mem_robust_dist = robust_accs(target_MODEL, mem, mem_labels, sigma)
    mem_robust_dist = np.array(mem_robust_dist)
    # print('Member:', sigma, robust_dist, len(robust_dist))

    nonmem_robust_dist = robust_accs(target_MODEL, non_mem, non_mem_labels, sigma)
    nonmem_robust_dist = np.array(nonmem_robust_dist)
    # print('Non-Member:', sigma, robust_dist, len(robust_dist))

    robust_dist = np.concatenate((mem_robust_dist, nonmem_robust_dist), axis=0)
    mem_ground = np.array([1]*len(mem_robust_dist))
    nonmem_ground = np.array([0]*len(nonmem_robust_dist))
    ground = np.concatenate((mem_ground, nonmem_ground), axis=0)
    
   
    pred = (robust_dist>threshold).astype(int)
    accuracy = accuracy_score(ground, pred)

      
    return accuracy, 1

def inference_auc(dataset, target_MODEL, mem, mem_labels, non_mem, non_mem_labels):
    # sigma = 5. / dataset.input_size
    best_acc = -1
    best_auc = -1
    print('Start inference...')
    sigmas = [1. / dataset.input_size, 2. / dataset.input_size, 3. / dataset.input_size, 5. / dataset.input_size, 10. / dataset.input_size]
    for sigma in sigmas:
        mem_robust_dist = robust_accs(target_MODEL, mem, mem_labels, sigma)
        mem_robust_dist = np.array(mem_robust_dist)
        # print('Member:', sigma, robust_dist, len(robust_dist))

        nonmem_robust_dist = robust_accs(target_MODEL, non_mem, non_mem_labels, sigma)
        nonmem_robust_dist = np.array(nonmem_robust_dist)
        # print('Non-Member:', sigma, robust_dist, len(robust_dist))

        robust_dist = np.concatenate((mem_robust_dist, nonmem_robust_dist), axis=0)

        mem_ground = np.array([1]*len(mem_robust_dist))
        nonmem_ground = np.array([0]*len(nonmem_robust_dist))
        ground = np.concatenate((mem_ground, nonmem_ground), axis=0)
        
        # print(ground)
        # print(robust_dist)
        # accuracy = accuracy_score(ground, robust_dist)
        accuracy = highest_acc(ground, robust_dist)
        auc = round(roc_auc_score(ground, robust_dist), 4)
        # print(accuracy, auc)
        if accuracy > best_acc:
            best_acc = accuracy
        if auc > best_auc:
            best_auc = auc
    return best_acc, best_auc

def highest_acc(ground, robust_dist):
    best_acc = -1
    for threshold in range(0, 100, 1):
            threshold = threshold/100.0
            pred = (robust_dist>threshold).astype(int)
            accuracy = accuracy_score(ground, pred)

            if accuracy > best_acc:
                best_acc = accuracy
    return best_acc