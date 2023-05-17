import itertools, collections
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
from myart.attacks.evasion import HopSkipJump, ProjectedGradientDescent
from myart.estimators.classification import PyTorchClassifier
# from myart.utils import compute_success
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from myart.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from myart.attacks.evasion import HopSkipJump
import os
import pickle
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
def Attack_Dataloader(models_path, model_name, miatype, bandwidth):
    
    if 'idx' not in miatype:
        filename = '1_process_4_gpu_core'
    elif 'exit_idx' in miatype:
        filename = '1_process_4_gpu_core'
    elif '1_process_4_gpu_core' in miatype:
        filename = '1_process_4_gpu_core'
    elif '4_process_4_gpu_core' in miatype:
        filename = '4_process_4_gpu_core'
    elif '4_process_1_gpu_core' in miatype:
        filename = '4_process_1_gpu_core'
    elif 'posterior' in miatype:
        filename = '1_process_4_gpu_core'
    #print(filename)
    AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset_{filename}.pkl', 'rb'))#.item()
    AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset_{filename}.pkl', 'rb'))#.item()
    if type(AttackModelTrainSet) == np.ndarray:
        AttackModelTrainSet = AttackModelTrainSet.item()
        AttackModelTestSet = AttackModelTestSet.item()

    num_exits = AttackModelTrainSet['num_exits']
    nb_classes = AttackModelTestSet['nb_classes']

    #print(model_name)
    acc = -1
    if 'sdn' in model_name and '4_gpu_core' in miatype:
        test_time = np.array(AttackModelTestSet['infer_time'], dtype='f')
        test_idx = AttackModelTestSet['exit_idx']
       
        test_time = np.array([np.mean(np.sort(it, axis=0)[:1]) for it in test_time])
        s = np.linspace(0, np.max(test_time)+0.5)
        a = test_time.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(a)
        
        e = kde.score_samples(s.reshape(-1, 1))
        # plt.plot(s, e)
        # plt.show()
        mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
        
        Minima = s[mi]
        Maxima = s[ma]

        Minima = Minima[0:num_exits-1]
        #print(Minima)
        #print(Minima)
        pred_idx = test_idx * -1
        for input_idx, infer_time in enumerate(test_time):
            for idx, mini in enumerate(Minima):
                if infer_time < mini:
                    pred_idx[input_idx] = idx
                    break
                elif idx == len(Minima)-1:
                    pred_idx[input_idx] = idx + 1
                    break
            # if pred_idx[input_idx] == 3:
            #     print(idx, mini)        
        correct = pred_idx == test_idx
        correct = correct.sum()
        acc = correct/(test_time.shape[0]+0.0)
        print('predict exit_idx accuracy:', acc,  'bandwidth',bandwidth)
        AttackModelTestSet['exit_idx'] = pred_idx
    elif 'sdn' in model_name and 'posterior' in miatype:
        device = 'cpu'
        Epochs = 100
        attack_model = posteriorAttackModel(nb_classes, num_exits)
        attack_optimizer = torch.optim.SGD(attack_model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
        attack_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attack_optimizer, T_max=Epochs)
        attack_model = attack_model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        attack_train_set = torch.utils.data.TensorDataset(
                        torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
                        torch.from_numpy(np.array(AttackModelTrainSet['exit_idx'])).type(torch.long),)
        attack_train_loader = torch.utils.data.DataLoader(attack_train_set, batch_size=100, shuffle=False)

        attack_test_set = torch.utils.data.TensorDataset(
                        torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
                        torch.from_numpy(np.array(AttackModelTestSet['exit_idx'])).type(torch.long),)
        attack_test_loader = torch.utils.data.DataLoader(attack_test_set, batch_size=100, shuffle=False)
        best_accuracy = 0.0
        for epoch in range(Epochs):
            train_loss = 0
            correct = 0
            attack_model.train()
            for batch_idx, (data, target) in enumerate(attack_train_loader):  
                data, target = data.to(device), target.to(device)
                output = attack_model(data)
                loss = loss_fn(output, target)
                attack_optimizer.zero_grad()
                loss.backward()
                attack_optimizer.step()
                        
                train_loss += loss.item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss /= len(attack_train_loader.dataset)
            train_accuracy = 100. * correct / len(attack_train_loader.dataset)

            attack_model.eval()
            test_loss = 0
            correct = 0

            Pred_Exit = []
            for batch_idx, (data, target) in enumerate(attack_test_loader): 
               
                data, target = data.to(device), target.to(device)
                output = attack_model(data)
                loss = loss_fn(output, target)
                
                test_loss += loss.item()
                pred = output.max(1, keepdim=True)[1]
                
                correct += pred.eq(target.view_as(pred)).sum().item()

                pred_exit = pred.view(-1).cpu().numpy()
                Pred_Exit = np.concatenate((Pred_Exit, pred_exit), axis=0)
                
            test_loss /= len(attack_train_loader.dataset)
            test_accuracy = 100. * correct / len(attack_test_loader.dataset)

            is_best_accuracy= test_accuracy > best_accuracy
            if is_best_accuracy:
                best_accuracy = test_accuracy
                AttackModelTestSet['exit_idx'] = Pred_Exit
            if epoch>1000:
                print(('epoch:{} \t train_loss:{:.4f} \t test_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t best_accuracy:{:.4f}')
                                    .format(epoch, train_loss, test_loss,
                                            train_accuracy, test_accuracy, best_accuracy))
            
            attack_scheduler.step()
        acc = best_accuracy
        print('predict exit_idx accuracy:', acc,  'bandwidth',bandwidth)

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

    return target_train_exit_idx_loader, target_test_exit_idx_loader, shadow_train_exit_idx_loader, shadow_test_exit_idx_loader, num_exits, nb_classes, acc


def label_only_attack_random_nosie(args, models_path, device='cpu', supervised=True):
    save_path = '/.../.../'
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
    
    for model_idx, model_name in  enumerate(cnns_sdns):
        # if model_idx in [0,1,2,3,4]:
        #     continue
        # if model_idx < (args.finish_exits - 1):
        #     continue
        # if (model_idx + 1) == args.finish_exits:
        #     print('ok')
        # else:
        #     continue
        # logoname = f'{args.task}_{args.model}.txt'
        # with open(f'/p/project/hai_unganable/projects/multi-exit/multiexit-CCS/logsave/1/MIA/label-only/{sdn_training_type}/' + logoname, 'r') as f:
        #             while True:
        #                 lines = f.readline()
        #                 if not lines:
        #                     break

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
            Num_Each_Exit = 150 
        elif model_idx in [1,2]:
            Num_Each_Exit = 40
        else:
            Num_Each_Exit = 30

        Final_Results = []
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

            if mia == 'label_only':
                
                shadow_MEM_data = np.array(sum(shadow_Mem_Each_Exit,[]))
                shadow_MEM_label = np.array(sum(shadow_Mem_Label_Each_Exit, []))
                shadow_NonMEM_data = np.array(sum(shadow_NonMem_Each_Exit, []))
                shadow_NonMEM_label = np.array(sum(shadow_NonMem_Label_Each_Exit, []))
                
                shadow_MEM_data = np.squeeze(shadow_MEM_data)
                shadow_MEM_label = np.squeeze(shadow_MEM_label)
                shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)
                
                # shadow_MEM_data = shadow_MEM_data[35:40]
                # shadow_MEM_label = shadow_MEM_label[35:40]

                distance_threshold_tau = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label, max_iter=max_iter)


                target_MEM_data = np.array(sum(target_Mem_Each_Exit,[]))
                target_MEM_label = np.array(sum(target_Mem_Label_Each_Exit, []))
                target_NonMEM_data = np.array(sum(target_NonMem_Each_Exit, []))
                target_NonMEM_label = np.array(sum(target_NonMem_Label_Each_Exit, []))
                
                target_MEM_data = np.squeeze(target_MEM_data)
                target_MEM_label = np.squeeze(target_MEM_label)
                target_NonMEM_data = np.squeeze(target_NonMEM_data)
                target_NonMEM_label = np.squeeze(target_NonMEM_label)


                accuracy, auc = inference(dataset, target_MODEL, distance_threshold_tau, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, max_iter=max_iter)
                print(f'Exit_index {0},  Accuracy {accuracy}, AUC {auc}')
                Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':0, 'threshold_flag':True, 'accuracy_flag':True, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia})
            elif 'sdn' in model_name:
                previous_distance_threshold_tau = 0
                for idx in range(len(target_Mem_Each_Exit)):
                    threshold_flag = True
                    accuracy_flag = True
                    shadow_MEM_data = np.array(shadow_Mem_Each_Exit[idx])
                    shadow_MEM_label = np.array(shadow_Mem_Label_Each_Exit[idx])
                    shadow_NonMEM_data = np.array(shadow_NonMem_Each_Exit[idx])
                    shadow_NonMEM_label = np.array(shadow_NonMem_Label_Each_Exit[idx])
                    
                    shadow_MEM_data = np.squeeze(shadow_MEM_data)
                    shadow_MEM_label = np.squeeze(shadow_MEM_label)
                    shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                    shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)

                    if (len(shadow_MEM_data) == 0) or (len(shadow_NonMEM_data) == 0):
                        distance_threshold_tau = previous_distance_threshold_tau
                        threshold_flag = False
                    else:
                        distance_threshold_tau = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label, max_iter=max_iter)
                        previous_distance_threshold_tau = distance_threshold_tau
                        
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
                        accuracy, auc = inference(dataset, target_MODEL, distance_threshold_tau, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, max_iter=max_iter)
                    print(f'Exit_index {idx},  Accuracy {accuracy}, AUC {auc}, threshold_flag {threshold_flag}, accuracy_flag {accuracy_flag}')
                    Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':idx, 'threshold_flag':threshold_flag, 'accuracy_flag':accuracy_flag, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia})
                    
                    fh = open(save_path + f'/{sdn_training_type}/{args.task}_{args.model}.txt', 'w', encoding='utf-8')
                    straa = f'dataset:{args.task}, model:{args.model}, num_exits: {num_exits}, training_type:{sdn_training_type}, exit_index:{idx}, threshold_flag:{threshold_flag}, accuracy_flag:{accuracy_flag}, accuracy:{accuracy}, auc:{auc}, mia_type:{mia}'
                    fh.write(straa)
                    fh.close()
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(save_path + f'/{sdn_training_type}/{args.task}_{args.model}.csv')   

def label_only_attack(args, models_path, device='cpu', supervised=True):
    save_path = '/.../.../'
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
            Num_Each_Exit = 150 
        elif model_idx in [1,2]:
            Num_Each_Exit = 40
        else:
            Num_Each_Exit = 30

        Final_Results = []
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

            if mia == 'label_only':
                
                shadow_MEM_data = np.array(sum(shadow_Mem_Each_Exit,[]))
                shadow_MEM_label = np.array(sum(shadow_Mem_Label_Each_Exit, []))
                shadow_NonMEM_data = np.array(sum(shadow_NonMem_Each_Exit, []))
                shadow_NonMEM_label = np.array(sum(shadow_NonMem_Label_Each_Exit, []))
                
                shadow_MEM_data = np.squeeze(shadow_MEM_data)
                shadow_MEM_label = np.squeeze(shadow_MEM_label)
                shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)
                
                # shadow_MEM_data = shadow_MEM_data[35:40]
                # shadow_MEM_label = shadow_MEM_label[35:40]

                distance_threshold_tau = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label, max_iter=max_iter)


                target_MEM_data = np.array(sum(target_Mem_Each_Exit,[]))
                target_MEM_label = np.array(sum(target_Mem_Label_Each_Exit, []))
                target_NonMEM_data = np.array(sum(target_NonMem_Each_Exit, []))
                target_NonMEM_label = np.array(sum(target_NonMem_Label_Each_Exit, []))
                
                target_MEM_data = np.squeeze(target_MEM_data)
                target_MEM_label = np.squeeze(target_MEM_label)
                target_NonMEM_data = np.squeeze(target_NonMEM_data)
                target_NonMEM_label = np.squeeze(target_NonMEM_label)


                accuracy, auc = inference(dataset, target_MODEL, distance_threshold_tau, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, max_iter=max_iter)
                print(f'Exit_index {0},  Accuracy {accuracy}, AUC {auc}')
                Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':0, 'threshold_flag':True, 'accuracy_flag':True, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia})
            elif 'sdn' in model_name:
                previous_distance_threshold_tau = 0
                for idx in range(len(target_Mem_Each_Exit)):
                    threshold_flag = True
                    accuracy_flag = True
                    shadow_MEM_data = np.array(shadow_Mem_Each_Exit[idx])
                    shadow_MEM_label = np.array(shadow_Mem_Label_Each_Exit[idx])
                    shadow_NonMEM_data = np.array(shadow_NonMem_Each_Exit[idx])
                    shadow_NonMEM_label = np.array(shadow_NonMem_Label_Each_Exit[idx])
                    
                    shadow_MEM_data = np.squeeze(shadow_MEM_data)
                    shadow_MEM_label = np.squeeze(shadow_MEM_label)
                    shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                    shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)

                    if (len(shadow_MEM_data) == 0) or (len(shadow_NonMEM_data) == 0):
                        distance_threshold_tau = previous_distance_threshold_tau
                        threshold_flag = False
                    else:
                        distance_threshold_tau = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label, max_iter=max_iter)
                        previous_distance_threshold_tau = distance_threshold_tau
                        
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
                        accuracy, auc = inference(dataset, target_MODEL, distance_threshold_tau, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, max_iter=max_iter)
                    print(f'Exit_index {idx},  Accuracy {accuracy}, AUC {auc}, threshold_flag {threshold_flag}, accuracy_flag {accuracy_flag}')
                    Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':idx, 'threshold_flag':threshold_flag, 'accuracy_flag':accuracy_flag, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia})
                    
                    fh = open(save_path + f'/{sdn_training_type}/{args.task}_{args.model}.txt', 'w', encoding='utf-8')
                    straa = f'dataset:{args.task}, model:{args.model}, num_exits: {num_exits}, training_type:{sdn_training_type}, exit_index:{idx}, threshold_flag:{threshold_flag}, accuracy_flag:{accuracy_flag}, accuracy:{accuracy}, auc:{auc}, mia_type:{mia}'
                    fh.write(straa)
                    fh.close()
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(save_path + f'/{sdn_training_type}/{args.task}_{args.model}.csv')       

    # df = pd.DataFrame()
    # Distance = df.append(Distance, ignore_index=True)
    #Distance.to_csv(models_path + f'/attack/label-only/{sdn_training_type}/{csv_name}_{metric}.csv') 
def label_only_attack_practicality(args, models_path, device='cpu', supervised=True):
    save_path = '/.../.../'
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
    
    for model_idx, model_name in  enumerate(cnns_sdns):
        if model_idx in [0]:
            continue


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
        MIA_list = ['label_only+predict_idx+4_process_4_gpu_core']

        target_Mem_Each_Exit = [[] for _ in range(model_idx+1)]
        target_NonMem_Each_Exit = [[] for _ in range(model_idx+1)]
        target_Mem_Label_Each_Exit = [[] for _ in range(model_idx+1)]
        target_NonMem_Label_Each_Exit = [[] for _ in range(model_idx+1)]

        shadow_Mem_Each_Exit = [[] for _ in range(model_idx+1)]
        shadow_NonMem_Each_Exit = [[] for _ in range(model_idx+1)]
        shadow_Mem_Label_Each_Exit = [[] for _ in range(model_idx+1)]
        shadow_NonMem_Label_Each_Exit = [[] for _ in range(model_idx+1)]

        if model_idx == 0: 
            Num_Each_Exit = 150 
        elif model_idx in [1,2]:
            Num_Each_Exit = 40
        else:
            Num_Each_Exit = 30

        Final_Results = []
        for mia in MIA_list:
            print(f'-------------------{mia}------------------')
            for deviation in [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1]:
                for query_num in [1, 10, 30, 50, 80, 100, 300, 500, 800, 1000]:
                    target_train_exit_idx_loader, target_test_exit_idx_loader, shadow_train_exit_idx_loader, shadow_test_exit_idx_loader, num_exits, nb_classes, exit_acc = Attack_Dataloader_MR(models_path, model_name, mia, bandwidth=0.15, deviation=deviation, query_num=query_num)
                    
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

                    if mia == 'label_only':
                        
                        shadow_MEM_data = np.array(sum(shadow_Mem_Each_Exit,[]))
                        shadow_MEM_label = np.array(sum(shadow_Mem_Label_Each_Exit, []))
                        shadow_NonMEM_data = np.array(sum(shadow_NonMem_Each_Exit, []))
                        shadow_NonMEM_label = np.array(sum(shadow_NonMem_Label_Each_Exit, []))
                        
                        shadow_MEM_data = np.squeeze(shadow_MEM_data)
                        shadow_MEM_label = np.squeeze(shadow_MEM_label)
                        shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                        shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)
                        
                        # shadow_MEM_data = shadow_MEM_data[35:40]
                        # shadow_MEM_label = shadow_MEM_label[35:40]

                        distance_threshold_tau = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label, max_iter=max_iter)

                        target_MEM_data = np.array(sum(target_Mem_Each_Exit,[]))
                        target_MEM_label = np.array(sum(target_Mem_Label_Each_Exit, []))
                        target_NonMEM_data = np.array(sum(target_NonMem_Each_Exit, []))
                        target_NonMEM_label = np.array(sum(target_NonMem_Label_Each_Exit, []))
                        
                        target_MEM_data = np.squeeze(target_MEM_data)
                        target_MEM_label = np.squeeze(target_MEM_label)
                        target_NonMEM_data = np.squeeze(target_NonMEM_data)
                        target_NonMEM_label = np.squeeze(target_NonMEM_label)


                        accuracy, auc = inference(dataset, target_MODEL, distance_threshold_tau, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, max_iter=max_iter)
                        print(f'Exit_index {0},  Accuracy {accuracy}, AUC {auc}')
                        Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':0, 'threshold_flag':True, 'accuracy_flag':True, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia, 'deviation':deviation, 'query_num':query_num})
                    elif 'sdn' in model_name:
                        previous_distance_threshold_tau = 0
                        for idx in range(len(target_Mem_Each_Exit)):
                            threshold_flag = True
                            accuracy_flag = True
                            shadow_MEM_data = np.array(shadow_Mem_Each_Exit[idx])
                            shadow_MEM_label = np.array(shadow_Mem_Label_Each_Exit[idx])
                            shadow_NonMEM_data = np.array(shadow_NonMem_Each_Exit[idx])
                            shadow_NonMEM_label = np.array(shadow_NonMem_Label_Each_Exit[idx])
                            
                            shadow_MEM_data = np.squeeze(shadow_MEM_data)
                            shadow_MEM_label = np.squeeze(shadow_MEM_label)
                            shadow_NonMEM_data = np.squeeze(shadow_NonMEM_data)
                            shadow_NonMEM_label = np.squeeze(shadow_NonMEM_label)

                            if (len(shadow_MEM_data) == 0) or (len(shadow_NonMEM_data) == 0):
                                distance_threshold_tau = previous_distance_threshold_tau
                                threshold_flag = False
                            else:
                                distance_threshold_tau = search_threshold(dataset, shadow_MODEL, shadow_MEM_data, shadow_MEM_label, shadow_NonMEM_data, shadow_NonMEM_label, max_iter=max_iter)
                                previous_distance_threshold_tau = distance_threshold_tau
                                
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
                                accuracy, auc = inference(dataset, target_MODEL, distance_threshold_tau, target_MEM_data, target_MEM_label, target_NonMEM_data, target_NonMEM_label, max_iter=max_iter)
                            print(f'Exit_index {idx},  Accuracy {accuracy}, AUC {auc}, threshold_flag {threshold_flag}, accuracy_flag {accuracy_flag}')
                            Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_index':idx, 'threshold_flag':threshold_flag, 'accuracy_flag':accuracy_flag, 'accuracy':accuracy, 'auc':auc, 'mia_type':mia, 'deviation':deviation, 'query_num':query_num})
                            
                            # fh = open(save_path + f'/{sdn_training_type}/{args.task}_{args.model}.txt', 'w', encoding='utf-8')
                            # straa = f'dataset:{args.task}, model:{args.model}, num_exits: {num_exits}, training_type:{sdn_training_type}, exit_index:{idx}, threshold_flag:{threshold_flag}, accuracy_flag:{accuracy_flag}, accuracy:{accuracy}, auc:{auc}, mia_type:{mia}'
                            # fh.write(straa)
                            # fh.close()
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(save_path + f'/{sdn_training_type}/{args.task}_{args.model}.csv')       

    # df = pd.DataFrame()
    # Distance = df.append(Distance, ignore_index=True)
    #Distance.to_csv(models_path + f'/attack/label-only/{sdn_training_type}/{csv_name}_{metric}.csv') 


def search_threshold(dataset, shadow_MODEL, mem, mem_shadow_labels, non_mem, non_mem_shadow_labels, max_iter):

    print('searchThreshold...')
    loss_fn = nn.CrossEntropyLoss()
    ARTclassifier = PyTorchClassifier(
            model=shadow_MODEL.eval(),
            #clip_values=(0, 1),
            loss=loss_fn,
            input_shape=(3, dataset.img_size, dataset.img_size),
            nb_classes=dataset.num_classes,
            #preprocessing = (dataset.mean, dataset.std)
            )
    LabelonlyAttack = LabelOnlyDecisionBoundary(
            estimator=ARTclassifier, distance_threshold_tau=None)

    #threshold_supervised = True
    #if threshold_supervised:
    LabelonlyAttack.calibrate_distance_threshold(x_train=mem, y_train=mem_shadow_labels,
                                                     x_test=non_mem, y_test=non_mem_shadow_labels, max_iter=max_iter, batch_size=1, verbose=True)
    #else:
    #    LabelonlyAttack.calibrate_distance_threshold_unsupervised(num_samples=100, normalize=(dataset.mean, dataset.std), img_size=dataset.img_size, max_iter=max_iter, verbose=False)

    distance_threshold_tau = LabelonlyAttack.distance_threshold_tau
    return distance_threshold_tau

def inference(dataset, target_MODEL, distance_threshold_tau, mem, mem_target_labels, non_mem, non_mem_target_labels, max_iter):
    print('Start inference...')
    loss_fn = nn.CrossEntropyLoss()
    ARTclassifier = PyTorchClassifier(
            model=target_MODEL.eval(),
            #clip_values=(0, 1),
            loss=loss_fn,
            input_shape=(3, dataset.img_size, dataset.img_size),
            nb_classes=dataset.num_classes,
            #preprocessing = (dataset.mean, dataset.std)
            )

    LabelonlyAttack = LabelOnlyDecisionBoundary(
            estimator=ARTclassifier, distance_threshold_tau=distance_threshold_tau)

    train_set = np.concatenate(
            (mem, non_mem), axis=0)
    train_target_labels = np.concatenate(
            (mem_target_labels, non_mem_target_labels), axis=0)

    member_ground_truth = [1 if idx < len(mem_target_labels) else 0 for idx in range(len(train_target_labels))]

    member_predictions = LabelonlyAttack.infer(
            x=train_set, y=train_target_labels, max_iter=max_iter, batch_size=1, verbose=True)
    #distance = LabelonlyAttack.distance
    acc = accuracy_score(member_ground_truth, member_predictions)
    auc = roc_auc_score(member_ground_truth, member_predictions)
    return acc, auc#, distance

def inference_auc(dataset, target_MODEL, distance_threshold_tau, train_set, train_set_labels, Exit_labels, Mem_Each_Exit, max_iter):
    print('Start inference...')
    loss_fn = nn.CrossEntropyLoss()
    ARTclassifier = PyTorchClassifier(
            model=target_MODEL.eval(),
            #clip_values=(0, 1),
            loss=loss_fn,
            input_shape=(3, dataset.img_size, dataset.img_size),
            nb_classes=dataset.num_classes,
            #preprocessing = (dataset.mean, dataset.std)
            )

    LabelonlyAttack = LabelOnlyDecisionBoundary(
            estimator=ARTclassifier, distance_threshold_tau=distance_threshold_tau)

    member_predictions = LabelonlyAttack.infer(
            x=train_set, y=train_set_labels, max_iter=max_iter, verbose=False, batch_size=1), 
    distance = LabelonlyAttack.distance

    mid_idx = sum(Mem_Each_Exit)
    member_ground_truth = [1 if idx < mid_idx else 0 for idx in range(len(train_set))]

    auc_exit = None
    # if metric == 'prediction':
    #     auc_overall = round(roc_auc_score(member_ground_truth, distance), 4)
    # else:
    auc_overall = round(roc_auc_score(member_ground_truth, distance), 4)


    member_status_ground_truth_exit = [[] for _ in range(len(Mem_Each_Exit))]
    distance_exit = [[] for _ in range(len(Mem_Each_Exit))]
    # print(distance)
    # print(Exit_labels)
    # print(distance_exit)
    for dist_idx in range(len(distance)):
        distance_exit[Exit_labels[dist_idx]].append(distance[dist_idx])
        if dist_idx < mid_idx:
            member_status_ground_truth_exit[Exit_labels[dist_idx]].append(1)
        else:
            member_status_ground_truth_exit[Exit_labels[dist_idx]].append(0)
    # print(member_status_ground_truth_exit)
    # print(distance_exit)
    auc_exit = [] 
    for idx in range(len(distance_exit)):
        
        if len(member_status_ground_truth_exit[idx])==0:
            auc_exit.append(-1)
        elif max(member_status_ground_truth_exit[idx]) == min(member_status_ground_truth_exit[idx]):
            auc_exit.append(-2)
        else:
            auc_exit.append(round(roc_auc_score(member_status_ground_truth_exit[idx], distance_exit[idx]), 4))
    #auc_exit = [1 if (max(member_status_ground_truth_exit[idx]) == min(member_status_ground_truth_exit[idx])) or (len(member_status_ground_truth_exit[idx])==0) else round(roc_auc_score(member_status_ground_truth_exit[idx], distance_exit[idx]), 4) ]
    
    return auc_overall, auc_exit, distance

