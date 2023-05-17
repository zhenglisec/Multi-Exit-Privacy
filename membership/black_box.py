import itertools, collections
import deeplearning.aux_funcs  as af
import deeplearning.model_funcs as mf
from deeplearning import network_architectures as arcs
from deeplearning.profiler import profile_sdn, profile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from membership import check_and_transform_label_format, MLP_BLACKBOX, posteriorAttackModel, model_feature_len, train_black_mia_attack_model, test_black_mia_attack_model
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pandas as pd
import os, heapq
import imagehash
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
    if 'mlp' in model_name:
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
    else:
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset_{filename}.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset_{filename}.pkl', 'rb'))#.item()
    
    if type(AttackModelTrainSet) == np.ndarray:
        AttackModelTrainSet = AttackModelTrainSet.item()
    if type(AttackModelTestSet) == np.ndarray:
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
        attack_model = posteriorAttackModel(3, num_exits)
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
                data, _ = torch.topk(data, k=3, dim=1, largest=True, sorted=False)
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

            Shadow_Pred = []
            Target_Pred = []
            for batch_idx, ((shadow_data, shadow_target), (target_data, target_target)) in enumerate(zip(attack_train_loader,attack_test_loader)): 
                shadow_data, _ = torch.topk(shadow_data, k=3, dim=1, largest=True, sorted=False)
                target_data, _ = torch.topk(target_data, k=3, dim=1, largest=True, sorted=False)
                shadow_data, shadow_target = shadow_data.to(device), shadow_target.to(device)
                target_data, target_target = target_data.to(device), target_target.to(device)

                shadow_output = attack_model(shadow_data)
                target_output = attack_model(target_data)
                #loss = loss_fn(output, target)
                
                #test_loss += loss.item()
                shadow_pred = shadow_output.max(1, keepdim=True)[1]
                target_pred = target_output.max(1, keepdim=True)[1]
                correct += target_pred.eq(target_target.view_as(pred)).sum().item()

                shadow_pred = shadow_pred.view(-1).cpu().numpy()
                Shadow_Pred = np.concatenate((Shadow_Pred, shadow_pred), axis=0)
                target_pred = target_pred.view(-1).cpu().numpy()
                Target_Pred = np.concatenate((Target_Pred, target_pred), axis=0)
            #test_loss /= len(attack_train_loader.dataset)
            test_accuracy = 100. * correct / len(attack_test_loader.dataset)

            is_best_accuracy= test_accuracy > best_accuracy
            if is_best_accuracy:
                best_accuracy = test_accuracy
                
                AttackModelTestSet['exit_idx'] = Target_Pred
            if epoch>0:
                print(('epoch:{} \t train_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t best_accuracy:{:.4f}')
                                    .format(epoch, train_loss ,
                                            train_accuracy, test_accuracy, best_accuracy))
            
            attack_scheduler.step()
        
        AttackModelTrainSet['exit_idx'] = Shadow_Pred
        AttackModelTestSet['exit_idx'] = Target_Pred
        acc = best_accuracy
        print('predict exit_idx accuracy:', acc,  'bandwidth',bandwidth)

    train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),
            )
    test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),
            )
    attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)

    return attack_train_loader, attack_test_loader, num_exits, nb_classes, acc
def Attack_Dataloader_MR(models_path, model_name, miatype, bandwidth, deviation=None, query_num=None):
    
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
    # #print(filename)
    if 'mlp' in model_name:
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
    else:
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset_{filename}.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset_{filename}.pkl', 'rb'))#.item()

    # print(AttackModelTrainSet.keys())
    if type(AttackModelTrainSet) == np.ndarray:
        AttackModelTrainSet = AttackModelTrainSet.item()
    if type(AttackModelTestSet) == np.ndarray:
        AttackModelTestSet = AttackModelTestSet.item()

    AttackModelTrainSet['model_features'] = [0]*AttackModelTrainSet['model_scores'].shape[0]
    AttackModelTrainSet['model_gradient'] = [0]*AttackModelTrainSet['model_scores'].shape[0]
    AttackModelTestSet['model_features'] = [0]*AttackModelTrainSet['model_scores'].shape[0]
    AttackModelTestSet['model_gradient'] = [0]*AttackModelTrainSet['model_scores'].shape[0]
    # print(AttackModelTestSet['member_status'].shape)
    num_exits = AttackModelTrainSet['num_exits']
    nb_classes = AttackModelTestSet['nb_classes']
    data_seed = AttackModelTestSet['data_seed']
    #print(model_name)
    acc = -1
    if 'sdn' in model_name and '4_gpu_core' in miatype:
        test_time = np.array(AttackModelTestSet['infer_time'], dtype='f')
        test_idx = AttackModelTestSet['exit_idx']

        # atest_time = np.array([np.mean(np.sort(it, axis=0)[:1]) for it in test_time])

        # print(atest_time[:10])
        # print(atest_time.shape)
        

        btest_time = []
        for it in test_time:
            min_time = np.sort(it, axis=0)[0]
            btest_time.append(min_time)
        test_time = np.array(btest_time)
        # print(test_time[:10])
        # print(test_time.shape)

        noise = 0
        for _ in range(query_num):
            noise += np.abs(np.random.normal(2, deviation, test_time.shape[0]))
        noise = noise/query_num
        test_time = test_time + noise
        # print(test_time[:10])
        # exit()
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
        print('predict exit_idx accuracy:', acc,  'bandwidth:',bandwidth, 'deviation:', deviation, 'query_num:',query_num)
        AttackModelTestSet['exit_idx'] = pred_idx
    
    train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),
            )
    test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),
            )
    attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)

    return attack_train_loader, attack_test_loader, num_exits, nb_classes, acc
def Attack_Dataloader_RandomDefense(models_path, model_name, miatype, bandwidth, deviation=None):
    
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
    flag = False
    for dataset in ['purchase', 'texas', 'location']:
        if dataset in model_name:
            flag = True
    if flag:
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
    else:
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset_{filename}.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset_{filename}.pkl', 'rb'))#.item()
    # print(AttackModelTrainSet.keys())
    if type(AttackModelTrainSet) == np.ndarray:
        AttackModelTrainSet = AttackModelTrainSet.item()
    if type(AttackModelTestSet) == np.ndarray:
        AttackModelTestSet = AttackModelTestSet.item()

    num_samples = int(AttackModelTrainSet['model_loss'].shape[0])
    AttackModelTrainSet['model_features'] = [0]*num_samples
    AttackModelTrainSet['model_gradient'] = [0]*num_samples
    AttackModelTestSet['model_features'] = [0]*num_samples
    AttackModelTestSet['model_gradient'] = [0]*num_samples
    # print(AttackModelTestSet['member_status'].shape)
    num_exits = AttackModelTrainSet['num_exits']
    nb_classes = AttackModelTestSet['nb_classes']
    data_seed = AttackModelTestSet['data_seed']
    #print(model_name)
    acc = -1
    sum_time_origin = -1
    sum_time_new = -1
    sum_time_max = -1
    if 'sdn' in model_name and '4_gpu_core' in miatype:
        test_time = np.array(AttackModelTestSet['infer_time'], dtype='f')
        test_idx = AttackModelTestSet['exit_idx']
        # print(test_time[:100])
        # print(test_idx[:100])
        # atest_time = np.array([np.mean(np.sort(it, axis=0)[:1]) for it in test_time])

        # print(atest_time[:10])
        # print(atest_time.shape)
        

        btest_time = []
        for it in test_time:
            min_time = np.sort(it, axis=0)[0]
            btest_time.append(min_time)
        test_time = np.array(btest_time)
        sum_time_origin = np.sum(test_time)

        top_k = 100
        top_k_idx=test_idx.argsort()[::-1][0:top_k]

   

        max_time = 0
        for idx in top_k_idx:
            max_time += test_time[idx]
        max_time = max_time/top_k
        sum_time_max = max_time*num_samples

        delay_time = []
        for idx, idx_time in enumerate(test_time):
            if test_idx[idx] < num_exits -1 and idx_time < max_time:
                # print(test_idx[idx], num_exits, idx_time, max_time)
                np.random.seed(data_seed[idx])  # here "idx" can be replaced by imagehash(image), if the input x is non-image, we can reshape x to a 3D array.
                noise = np.random.normal(idx_time, deviation, 1)[0]  # noise = np.random.normal(idx_time, portion * (max_time-idx_time), 1)[0]
                noise = np.abs(idx_time-noise)
                delay = idx_time + noise 
                if delay > max_time:
                    delay = max_time
                delay_time.append(delay)
            else:
                delay_time.append(idx_time)

        test_time = np.array(delay_time)
        sum_time_new = np.sum(test_time)
        # print(test_time[:10])
        # exit()
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
        print('predict exit_idx accuracy:', acc,  'bandwidth:',bandwidth, 'deviation:', deviation, 'sum_time_origin:',sum_time_origin, 'sum_time_new:',sum_time_new, 'sum_time_max:', sum_time_max)
        AttackModelTestSet['exit_idx'] = pred_idx
    
    train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),
            )
    test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),
            )
    attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)

    return attack_train_loader, attack_test_loader, num_exits, nb_classes, acc, sum_time_origin, sum_time_new, sum_time_max
def Attack_Dataloader_New(models_path, model_name, miatype, bandwidth):

    #print(filename)
    AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
    AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
    
    if type(AttackModelTrainSet) == np.ndarray:
        AttackModelTrainSet = AttackModelTrainSet.item()
    if type(AttackModelTestSet) == np.ndarray:
        AttackModelTestSet = AttackModelTestSet.item()

    num_exits = AttackModelTrainSet['num_exits']
    nb_classes = AttackModelTestSet['nb_classes']

    train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),
            )
    test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
            torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
            #torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
            #torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),
            )
    attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)

    return attack_train_loader, attack_test_loader, num_exits, nb_classes, 1

def black_box_attack(args, models_path, device='cpu'):

    Final_Results = []

    sdn_training_type = args.training_type
    csv_save_path = f'/.../.../'
    os.makedirs(csv_save_path, exist_ok=True)
    #args.model = cur_model
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

    cnns_sdns = []
    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))

    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        # cnns_sdns.append('None')
        cnns_sdns.append(model_name + '_cnn')
    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

    for model_idx, model_name in  enumerate(cnns_sdns):
        print(f'------------------model: {model_name}, num_braches:{model_idx+1}-------------------')
        # if model_idx in [0,1]: continue
        #orgin_model_name = model_name
        if 'cnn' in model_name:
            model_name = model_name
            save_path = models_path + '/attack/' + model_name
        elif 'sdn' in model_name:
            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
            save_path = models_path + '/attack/' + model_name
        else:
            continue
        af.create_path(save_path)

        MIA_list = ['black_box_top3', 'black_box_top3+exit_idx', 'black_box_top3+predict_idx+4_process_4_gpu_core', 'black_box_top3+predict_idx+4_process_1_gpu_core']
        for mia in MIA_list:
            print(f'-------------------{mia}------------------')
            #for bandwidth in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            attack_train_loader, attack_test_loader, num_exits, nb_classes, exit_acc = Attack_Dataloader_New(models_path, model_name, mia, bandwidth=0.15)
            
            if mia == 'black_box_top3':
                attack_model = MLP_BLACKBOX(3)
            elif mia == 'black_box_top3+exit_idx': #and 'sdn' in model_name:
                attack_model = MLP_BLACKBOX(3,  num_exits)
            elif 'black_box_top3+predict_idx' in mia and 'sdn' in model_name:
                attack_model = MLP_BLACKBOX(3,  num_exits)
    
            elif mia == 'black_box_sorted':
                attack_model = MLP_BLACKBOX(nb_classes) 
            elif mia == 'black_box_sorted+exit_idx' and 'sdn' in model_name:
                attack_model = MLP_BLACKBOX(nb_classes, num_exits)
            elif 'black_box_sorted+predict_idx' in mia and 'sdn' in model_name:
                attack_model = MLP_BLACKBOX(nb_classes, num_exits)
            else:
                continue
            epoch = 100
            attack_optimizer = torch.optim.SGD(attack_model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
            attack_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attack_optimizer, T_max=epoch)
            attack_model = attack_model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            best_prec1 = 0.0
            best_auc = 0.0
            for epoch in range(epoch):
                #os.system('nvidia-smi')
                train_loss, train_prec1 = train_black_mia_attack_model(mia, attack_model, attack_train_loader, attack_optimizer, loss_fn, device)
                val_loss, val_prec1, val_auc = test_black_mia_attack_model(mia, attack_model, attack_test_loader, loss_fn, device)
                attack_scheduler.step()
                is_best_prec1 = val_prec1 > best_prec1
                is_best_auc = val_auc > best_auc
                if is_best_prec1:
                    best_prec1 = val_prec1
                if is_best_auc:
                    best_auc = val_auc
                if epoch > 90:
                    print(('epoch:{} \t train_loss:{:.4f} \t test_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t best_prec1:{:.4f} \t best_auc:{:.4f}')
                            .format(epoch, train_loss, val_loss,
                                    train_prec1, val_prec1, best_prec1, best_auc))
            #predict_acc_results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_acc':exit_acc, 'best_prec1':best_prec1, 'mia_type':mia, 'bandwidth':0.15})
            torch.save(attack_model.state_dict(), save_path + '/' + mia + '.pkl')
            Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'accuracy':best_prec1, 'auc':best_auc, 'mia_type':mia})
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(csv_save_path + f'/{args.task}_{args.model}.csv')   
    
def black_box_attack_practicality(args, models_path, device='cpu'):

    predict_acc_results = []
    Final_Results = []

    sdn_training_type = args.training_type
    csv_save_path = f'/.../.../'
    os.makedirs(csv_save_path, exist_ok=True)
    #args.model = cur_model
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

    cnns_sdns = []
    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.abc.Iterable) else [item] for item in add_ic)))

    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        # cnns_sdns.append('None')
        cnns_sdns.append(model_name + '_cnn')
    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

    for model_idx, model_name in  enumerate(cnns_sdns):
        print(f'------------------model: {model_name}, num_braches:{model_idx+1}-------------------')
        if model_idx in [0,1,2,3,4]: continue
        #orgin_model_name = model_name
        if 'cnn' in model_name:
            model_name = model_name
            save_path = models_path + '/attack/' + model_name
        elif 'sdn' in model_name:
            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
            save_path = models_path + '/attack/' + model_name
        else:
            continue
        af.create_path(save_path)

        #MIA_list = ['black_box_top3', 'black_box_top3+exit_idx', 'black_box_top3+predict_idx+4_process_4_gpu_core', 'black_box_top3+predict_idx+4_process_1_gpu_core']

        MIA_list = ['black_box_top3+predict_idx+4_process_4_gpu_core']
        
        
        for mia in MIA_list:
            print(f'-------------------{mia}------------------')
            for deviation in [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1]:
                # perfect_exit_acc = 0
                previous_exit_acc = -1
                previous_acc = -1
                previous_auc = -1
                for query_num in [1,2,3,4,5,6,7,8,9,10, 40, 60, 80, 100]:
                    attack_train_loader, attack_test_loader, num_exits, nb_classes, exit_acc = Attack_Dataloader_MR(models_path, model_name, mia, bandwidth=0.15, deviation=deviation, query_num=query_num)
                    
                    
                    if exit_acc == 1 and previous_exit_acc == 1:
                        # previous_exit_acc = 1
                        best_prec1 = previous_acc
                        best_auc = previous_auc
                        print('predict exit_idx accuracy:', exit_acc,  'bandwidth:',0.15, 'deviation:', deviation, 'query_num:',query_num)
                        Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'accuracy':best_prec1, 'auc':best_auc, 'exit_acc':exit_acc, 'mia_type':mia, 'deviation':deviation, 'query_num':query_num})
                        continue                       


                    if mia == 'black_box_top3':
                        attack_model = MLP_BLACKBOX(3)
                    elif mia == 'black_box_top3+exit_idx' and 'sdn' in model_name:
                        attack_model = MLP_BLACKBOX(3,  num_exits)
                    elif 'black_box_top3+predict_idx' in mia and 'sdn' in model_name:
                        attack_model = MLP_BLACKBOX(3,  num_exits)
            
                    elif mia == 'black_box_sorted':
                        attack_model = MLP_BLACKBOX(nb_classes) 
                    elif mia == 'black_box_sorted+exit_idx' and 'sdn' in model_name:
                        attack_model = MLP_BLACKBOX(nb_classes, num_exits)
                    elif 'black_box_sorted+predict_idx' in mia and 'sdn' in model_name:
                        attack_model = MLP_BLACKBOX(nb_classes, num_exits)
                    else:
                        continue
                    epoch = 100
                    attack_optimizer = torch.optim.SGD(attack_model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
                    attack_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attack_optimizer, T_max=epoch)
                    attack_model = attack_model.to(device)
                    loss_fn = nn.CrossEntropyLoss()
                    best_prec1 = 0.0
                    best_auc = 0.0
                    for epoch in range(epoch):
                        #os.system('nvidia-smi')
                        train_loss, train_prec1 = train_black_mia_attack_model(mia, attack_model, attack_train_loader, attack_optimizer, loss_fn, device)
                        val_loss, val_prec1, val_auc = test_black_mia_attack_model(mia, attack_model, attack_test_loader, loss_fn, device)
                        attack_scheduler.step()
                        is_best_prec1 = val_prec1 > best_prec1
                        is_best_auc = val_auc > best_auc
                        if is_best_prec1:
                            best_prec1 = val_prec1
                        if is_best_auc:
                            best_auc = val_auc
                        if epoch > 90:
                            print(('epoch:{} \t train_loss:{:.4f} \t test_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t best_auc:{:.4f} \t best_prec1:{:.4f}')
                                    .format(epoch, train_loss, val_loss,
                                            train_prec1, val_prec1, best_prec1, best_auc))
                    #predict_acc_results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_acc':exit_acc, 'best_prec1':best_prec1, 'mia_type':mia, 'bandwidth':0.15})
                    #torch.save(attack_model.state_dict(), save_path + '/' + mia + '.pkl')
                    if exit_acc == 1:
                        previous_exit_acc = 1
                        previous_auc = best_auc
                        previous_acc = best_prec1
                    Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'accuracy':best_prec1, 'auc':best_auc, 'exit_acc':exit_acc, 'mia_type':mia, 'deviation':deviation, 'query_num':query_num})
                    # Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_acc':exit_acc, 'mia_type':mia, 'deviation':deviation, 'query_num':query_num})

    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(csv_save_path + f'/{args.task}_{args.model}_mean_02_exit_acc.csv')   

def black_box_attack_randomdefense(args, models_path, device='cpu'):

    predict_acc_results = []
    Final_Results = []

    sdn_training_type = args.training_type
    csv_save_path = f'/home/c01zhli/Workspace/multi-exit-privacy/plots/attack_performance/black_box/sdn_training'
    os.makedirs(csv_save_path, exist_ok=True)
    #args.model = cur_model
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
    elif args.model == 'mlp_0':
        add_ic = args.add_ic[4]
        model_name = '{}_mlp_0'.format(args.task)
    elif args.model == 'mlp_1':
        add_ic = args.add_ic[4]
        model_name = '{}_mlp_1'.format(args.task)
    elif args.model == 'mlp_2':
        add_ic = args.add_ic[4]
        model_name = '{}_mlp_2'.format(args.task)
    elif args.model == 'mlp_3':
        add_ic = args.add_ic[4]
        model_name = '{}_mlp_3'.format(args.task)

    cnns_sdns = []
    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.abc.Iterable) else [item] for item in add_ic)))

    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        # cnns_sdns.append('None')
        cnns_sdns.append(model_name + '_cnn')
    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

    for model_idx, model_name in  enumerate(cnns_sdns):
        print(f'------------------model: {model_name}, num_braches:{model_idx+1}-------------------')
        if model_idx in [0,1,2,3,4]: continue
        #orgin_model_name = model_name
        if 'cnn' in model_name:
            model_name = model_name
            save_path = models_path + '/attack/' + model_name
        elif 'sdn' in model_name:
            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
            save_path = models_path + '/attack/' + model_name
        else:
            continue
        af.create_path(save_path)

        #MIA_list = ['black_box_top3', 'black_box_top3+exit_idx', 'black_box_top3+predict_idx+4_process_4_gpu_core', 'black_box_top3+predict_idx+4_process_1_gpu_core']

        MIA_list = ['black_box_top3+predict_idx+4_process_4_gpu_core']
        portion_list =[[0.01, 0.05, 0.09, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5],
                        [0.01, 0.05, 0.09, 0.12, 0.15, 0.2, 0.3, 0.4],
                        [0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2]]
        if 'mlp' in model_name:
            deviation_list = [0.05, 0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10, 11, 13, 15, 17, 20, 25]
        else:
            deviation_list = [0.05, 0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10, 11, 13, 15, 17, 20, 25]
        for mia in MIA_list:
            print(f'-------------------{mia}------------------')
            for deviation in deviation_list:
                attack_train_loader, attack_test_loader, num_exits, nb_classes, exit_acc, sum_time_origin, sum_time_new, sum_time_max = Attack_Dataloader_RandomDefense(models_path, model_name, mia, bandwidth=0.15, deviation=deviation)
                
                if mia == 'black_box_top3':
                    attack_model = MLP_BLACKBOX(3)
                elif mia == 'black_box_top3+exit_idx' and 'sdn' in model_name:
                    attack_model = MLP_BLACKBOX(3,  num_exits)
                elif 'black_box_top3+predict_idx' in mia and 'sdn' in model_name:
                    attack_model = MLP_BLACKBOX(3,  num_exits)
        
                elif mia == 'black_box_sorted':
                    attack_model = MLP_BLACKBOX(nb_classes) 
                elif mia == 'black_box_sorted+exit_idx' and 'sdn' in model_name:
                    attack_model = MLP_BLACKBOX(nb_classes, num_exits)
                elif 'black_box_sorted+predict_idx' in mia and 'sdn' in model_name:
                    attack_model = MLP_BLACKBOX(nb_classes, num_exits)
                else:
                    continue
                epoch = 100
                attack_optimizer = torch.optim.SGD(attack_model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
                attack_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attack_optimizer, T_max=epoch)
                attack_model = attack_model.to(device)
                loss_fn = nn.CrossEntropyLoss()
                best_prec1 = 0.0
                best_auc = 0.0
                for epoch in range(epoch):
                    #os.system('nvidia-smi')
                    train_loss, train_prec1 = train_black_mia_attack_model(mia, attack_model, attack_train_loader, attack_optimizer, loss_fn, device)
                    val_loss, val_prec1, val_auc = test_black_mia_attack_model(mia, attack_model, attack_test_loader, loss_fn, device)
                    attack_scheduler.step()
                    is_best_prec1 = val_prec1 > best_prec1
                    is_best_auc = val_auc > best_auc
                    if is_best_prec1:
                        best_prec1 = val_prec1
                    if is_best_auc:
                        best_auc = val_auc
                    if epoch > 90:
                        print(('epoch:{} \t train_loss:{:.4f} \t test_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t best_auc:{:.4f} \t best_prec1:{:.4f}')
                                .format(epoch, train_loss, val_loss,
                                        train_prec1, val_prec1, best_prec1, best_auc))
                #predict_acc_results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_acc':exit_acc, 'best_prec1':best_prec1, 'mia_type':mia, 'bandwidth':0.15})
                #torch.save(attack_model.state_dict(), save_path + '/' + mia + '.pkl')
                Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'accuracy':best_prec1, 
                                'auc':best_auc, 'mia_type':mia, 'deviation':deviation, 'sum_time_origin':sum_time_origin, 'sum_time_new':sum_time_new, 'sum_time_max':sum_time_max})
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(csv_save_path + f'/{args.task}_{args.model}_random_delay.csv')   

def black_box_trans_attack(args, models_path, device='cpu'):
    target_model = args.model
    target_task = args.task
    #dataset_list = ['cifar10']
    if target_task in ['purchase', 'texas', 'location',]:
        datasets_list = ['purchase', 'texas', 'location',]
        shadow_models_list = ['fcn_1', 'fcn_2', 'fcn_3', 'fcn_4']
    else:
        datasets_list = ['cifar10', 'cifar100', 'tinyimagenet']
        shadow_models_list = ['vgg', 'resnet', 'mobilenet', 'wideresnet']

    train_type = ['sdn_training']
    predict_acc_results = []
    for sdn_training_type in train_type:
        for shadow_task in datasets_list:
            #args.task = task
            for shadow_model in shadow_models_list:
                # if target_model == shadow_model:
                #     continue 
                if target_model == 'fcn_1':
                    add_ic = args.add_ic[0]
                    target_model_name = '{}_fcn_1'.format(args.task)
                elif target_model == 'fcn_2':
                    add_ic = args.add_ic[1]
                    target_model_name = '{}_fcn_2'.format(args.task)
                elif target_model == 'fcn_3':
                    add_ic = args.add_ic[2]
                    target_model_name = '{}_fcn_3'.format(args.task)
                elif target_model == 'fcn_4':
                    add_ic = args.add_ic[3]
                    target_model_name = '{}_fcn_4'.format(args.task)

                if shadow_model == 'fcn_1':
                    add_ic = args.add_ic[0]
                    shadow_model_name = '{}_fcn_1'.format(shadow_task)
                elif shadow_model == 'fcn_2':
                    add_ic = args.add_ic[1]
                    shadow_model_name = '{}_fcn_2'.format(shadow_task)
                elif shadow_model == 'fcn_3':
                    add_ic = args.add_ic[2]
                    shadow_model_name = '{}_fcn_3'.format(shadow_task)    
                elif shadow_model == 'fcn_4':
                    add_ic = args.add_ic[3]
                    shadow_model_name = '{}_fcn_4'.format(shadow_task)
                num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.abc.Iterable) else [item] for item in add_ic)))

                target_cnns_sdns = []
                if sdn_training_type == 'ic_only':
                    #target_cnns_sdns.append(target_model_name + '_cnn')
                    target_cnns_sdns.append('None')
                elif sdn_training_type == 'sdn_training':
                    target_cnns_sdns.append(target_model_name + '_cnn')
                    # target_cnns_sdns.append('None')
                for i in range(num_exits):
                    target_cnns_sdns.append(target_model_name + '_sdn')

                shadow_cnns_sdns = []
                if sdn_training_type == 'ic_only':
                    #shadow_cnns_sdns.append(shadow_model_name + '_cnn')
                    shadow_cnns_sdns.append('None')
                elif sdn_training_type == 'sdn_training':
                    # shadow_cnns_sdns.append('None')
                    shadow_cnns_sdns.append(shadow_model_name + '_cnn')
                for i in range(num_exits):
                    shadow_cnns_sdns.append(shadow_model_name + '_sdn')
                
                for model_idx, (target_model_name, shadow_model_name) in  enumerate(zip(target_cnns_sdns, shadow_cnns_sdns)):
                    if model_idx in [1,2,3,4]: continue
                    print(f'------------------target_model: {target_model_name}, shadow_model: {shadow_model_name}, num_braches:{model_idx+1}-------------------')
                    if 'cnn' in target_model_name:
                        target_model_name = target_model_name
                        #shadow_model_path = models_path + '/attack/inference/' + shadow_model_name
                    elif 'sdn' in target_model_name:
                        target_model_name = target_model_name + '/' + str(model_idx) + '/' + sdn_training_type
                        shadow_model_name = shadow_model_name + '/' + str(model_idx) + '/' + sdn_training_type
                        #shadow_model_path = models_path + '/attack/inference/' + shadow_model_name
                    else:
                        continue
                    
                    AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{shadow_model_name}/trainset.pkl', 'rb'))#.item()
                    AttackModelTestSet = pickle.load(open(models_path + f'/target/{target_model_name}/testset.pkl', 'rb'))#.item()
                    if type(AttackModelTrainSet) == np.ndarray:
                        AttackModelTrainSet = AttackModelTrainSet.item()
                    if type(AttackModelTestSet) == np.ndarray:
                        AttackModelTestSet = AttackModelTestSet.item()

                    num_exits = AttackModelTrainSet['num_exits']
                    nb_classes = AttackModelTestSet['nb_classes']

                    MIA_list = ['black_box_top3', 'black_box_top3+exit_idx',   ]       

                    #'black_box_sorted', 'black_box_sorted+predict_idx+4_process_4_gpu_core', 
                    
                    #MIA_list = ['black_box_top3+predict_idx+1_process_4_gpu_core', 'black_box_top3+predict_idx+4_process_4_gpu_core', 'black_box_top3+predict_idx+4_process_1_gpu_core']

                    for mia in MIA_list:
                        print(f'-------------------{mia}------------------')
                        #for bandwidth in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                        attack_train_loader, attack_test_loader, num_exits, nb_classes, exit_acc = Attack_Dataloader(models_path, target_model_name, mia, bandwidth=0.15)
                        
                        if mia == 'black_box_top3':
                            attack_model = MLP_BLACKBOX(3)
                        elif mia == 'black_box_top3+exit_idx' and 'sdn' in target_model_name:
                            attack_model = MLP_BLACKBOX(3,  num_exits)
                        elif 'black_box_top3+predict_idx' in mia and 'sdn' in target_model_name:
                            attack_model = MLP_BLACKBOX(3,  num_exits)
                
                        elif mia == 'black_box_sorted':
                            attack_model = MLP_BLACKBOX(nb_classes, 1) 
                        elif mia == 'black_box_sorted+exit_idx' and 'sdn' in target_model_name:
                            attack_model = MLP_BLACKBOX(nb_classes, num_exits)
                        elif 'black_box_sorted+predict_idx' in mia and 'sdn' in target_model_name:
                            attack_model = MLP_BLACKBOX(nb_classes, num_exits)
                        else:
                            continue

                        epoch = 100
                        attack_optimizer = torch.optim.SGD(attack_model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
                        attack_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attack_optimizer, T_max=epoch)
                        attack_model = attack_model.to(device)
                        loss_fn = nn.CrossEntropyLoss()
                        best_prec1 = 0.0
                        best_auc = 0.0
                        for epoch in range(epoch):
                            #os.system('nvidia-smi')
                            train_loss, train_prec1 = train_black_mia_attack_model(mia, attack_model, attack_train_loader, attack_optimizer, loss_fn, device, nb_classes)
                            val_loss, val_prec1, val_auc = test_black_mia_attack_model(mia, attack_model, attack_test_loader, loss_fn, device)
                            attack_scheduler.step()
                            is_best_prec1 = val_prec1 > best_prec1
                            is_best_auc = val_auc > best_auc
                            if is_best_prec1:
                                best_prec1 = val_prec1
                            if is_best_auc:
                                best_auc = val_auc
                            if epoch > 0:
                                print(('epoch:{} \t train_loss:{:.4f} \t test_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t best_auc:{:.4f} \t best_prec1:{:.4f}')
                                        .format(epoch, train_loss, val_loss,
                                                train_prec1, val_prec1, best_prec1, best_auc))
                        predict_acc_results.append({'targetdataset':target_task,'targetmodel':target_model, 'shadowdataset':shadow_task, 'shadowmodel':shadow_model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'exit_acc':exit_acc, 'best_prec1':best_prec1, 'mia_type':mia, 'bandwidth':0.15})
                        #torch.save(attack_model.state_dict(), save_path + '/' + mia + '.pkl')
    df = pd.DataFrame()
    Final_Results = df.append(predict_acc_results, ignore_index=True)
    Final_Results.to_csv(f'xxx.csv')

