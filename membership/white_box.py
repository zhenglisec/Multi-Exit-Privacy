import itertools, collections
import deeplearning.aux_funcs  as af
import deeplearning.model_funcs as mf
from deeplearning import network_architectures as arcs
from deeplearning.profiler import profile_sdn, profile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from membership import check_and_transform_label_format, WhiteBoxAttackModel, WhiteBoxAttackModel_ok_feature_dataset, WhiteBoxAttackModel_new, model_feature_len, train_mia_attack_model, test_mia_attack_model
def white_box_attack(args, models_path, device='cpu'):
    sdn_training_type = args.training_type
    csv_save_path = f'/home/c01zhli/Workspace/multi-exit-privacy/plots/{args.seed}/attack_performance/white_box/{args.training_type}'
    os.makedirs(csv_save_path, exist_ok=True)
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
    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))

    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        #cnns_sdns.append('None')
        cnns_sdns.append(model_name + '_cnn')

    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

        
    Final_Results = []
    for model_idx, model_name in  enumerate(cnns_sdns):
        print(f'------------------model: {model_name}, num_braches:{model_idx+1}-------------------')
        # if model_idx in [0,1,2,4,5,6]: continue
        orgin_model_name = model_name
        if 'cnn' in model_name:
            model_name = model_name
            save_path = models_path + '/attack/' + model_name
        elif 'sdn' in model_name:
            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
            save_path = models_path + '/attack/' + model_name
        else:
            continue
        af.create_path(save_path)

        #AttackModelTrainSet = np.load(models_path + f'/shadow/{model_name}/trainset.npy', allow_pickle=True).item()
        #AttackModelTestSet = np.load(models_path + f'/target/{model_name}/testset.npy', allow_pickle=True).item()
        AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
        AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
        if type(AttackModelTrainSet) == np.ndarray:
            AttackModelTrainSet = AttackModelTrainSet.item()
            AttackModelTestSet = AttackModelTestSet.item()
        num_exits = AttackModelTrainSet['num_exits']
        nb_classes = AttackModelTestSet['nb_classes']
        #AttackModelTrainSet['model_features'] = AttackModelTrainSet['model_features'][:, :512]
        feature_length = AttackModelTrainSet['model_features'].shape[1]
        # dict_keys(['model_scores', 'model_loss', 'orginal_labels', 'predicted_labels', 'predicted_status', 'infer_time', 'infer_time_norm', 'branch_idx', 
        #           'branch_norm', 'member_status', 'early_status', 'num_exits', 'nb_classes', 'model_features', 'model_gradient'])
        # print(AttackModelTrainSet['member_status'].shape)
        train_set = torch.utils.data.TensorDataset(
                torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
                torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True))).type(torch.float),
                torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
                #torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
                #torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
                torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),
                )
        test_set = torch.utils.data.TensorDataset(
                torch.from_numpy(np.array(AttackModelTrainSet['data_seed'])).type(torch.long),
                torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'], nb_classes=2, return_one_hot=True))).type(torch.float),
                torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
                #torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
                torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['exit_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
                #torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
                torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
                torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),
                )
        attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
        attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=1)
        # for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, infer_time_norm, branch_idx, branch_norm, member_status, early_status, model_features, model_gradient) in enumerate(attack_test_loader):
        #     print(model_features.shape)
        #     print(model_gradient.shape)
        #     exit()
        MIA_list = ['white_box', 'white_box+exit_idx'] #'white_box', 
 
        for mia in MIA_list:
            print(f'-------------------{mia}------------------')
            if mia == 'white_box':
                attack_model = WhiteBoxAttackModel(nb_classes, feature_length)
            elif mia == 'white_box+exit_idx':# and 'cnn' not in orgin_model_name:
                attack_model = WhiteBoxAttackModel(nb_classes, feature_length, num_exits)
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
                train_loss, train_prec1 = train_mia_attack_model(mia, attack_model, attack_train_loader, attack_optimizer, loss_fn, device)
                val_loss, val_prec1, val_auc = test_mia_attack_model(mia, attack_model, attack_test_loader, loss_fn, device)
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
            torch.save(attack_model.state_dict(), save_path + '/' + mia + '.pkl')
            Final_Results.append({'dataset':args.task,'model':args.model, 'num_exits':num_exits, 'training_type':sdn_training_type, 'accuracy':best_prec1, 'auc':best_auc, 'mia_type':mia})
    df = pd.DataFrame()
    Final_Results = df.append(Final_Results, ignore_index=True)
    Final_Results.to_csv(csv_save_path + f'/{args.task}_{args.model}.csv')   

def white_box_Statistic_Data_cpu(args, models_path, device='cpu'):
    datasets_list = ['cifar10', 'cifar100', 'tinyimagenet']
    models_list = ['vgg', 'resnet', 'mobilenet', 'wideresnet']
    train_type = ['ic_only']
    for sdn_training_type in train_type:
        for task in datasets_list:
            args.task = task
            for model in models_list:
                args.model = model
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

                cnns_sdns = []
                num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))

                if sdn_training_type == 'ic_only':
                    cnns_sdns.append(model_name + '_cnn')
                elif sdn_training_type == 'sdn_training':
                    cnns_sdns.append('None')

                for i in range(num_exits):
                    cnns_sdns.append(model_name + '_sdn')
                Statistic_Data = []
                for model_idx, model_name in  enumerate(cnns_sdns):
                    #if model_idx in [0,1,2,3,4]: continue
                    print(f'------------------model: {model_name}, num_braches:{model_idx+1}-------------------')
                    orgin_model_name = model_name
                    if 'cnn' in model_name:
                        model_name = model_name
                        #save_path = models_path + '/attack/' + model_name
                    elif 'sdn' in model_name:
                        model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
                        #save_path = models_path + '/attack/' + model_name
                    else:
                        continue
                    #

                    #AttackModelTrainSet = np.load(models_path + f'/shadow/{model_name}/trainset.npy', allow_pickle=True).item()
                    #AttackModelTestSet = np.load(models_path + f'/target/{model_name}/testset.npy', allow_pickle=True).item()
                    #AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
                    AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
                    if type(AttackModelTestSet) == np.ndarray:
                        #AttackModelTrainSet = AttackModelTrainSet.item()
                        AttackModelTestSet = AttackModelTestSet.item()

                    num_exits = AttackModelTestSet['num_exits']
                    nb_classes = AttackModelTestSet['nb_classes']
                    #AttackModelTrainSet['model_features'] = AttackModelTrainSet['model_features'][:, :512]
                    #feature_length = AttackModelTestSet['model_features'].shape[1]
                    # dict_keys(['model_scores', 'model_loss', 'orginal_labels', 'predicted_labels', 'predicted_status', 'infer_time', 'infer_time_norm', 'branch_idx', 
                    #           'branch_norm', 'member_status', 'early_status', 'num_exits', 'nb_classes', 'model_features', 'model_gradient'])

                    # train_set = torch.utils.data.TensorDataset(
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['branch_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),)
                    test_set = torch.utils.data.TensorDataset(
                            torch.from_numpy(np.array(AttackModelTestSet['data_seed'])).type(torch.long),
                            torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['orginal_labels'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['predicted_labels'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['predicted_status'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['branch_idx'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
                            torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),)
                    #attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
                    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
                    
                    for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, infer_time_norm, branch_idx, branch_norm, member_status, early_status, model_features, model_gradient) in enumerate(attack_test_loader):
                        results = {'model_scores':model_scores.numpy()[0], 'orginal_labels':int(orginal_labels.item()), 'model_loss':model_loss.item(), 'predicted_status':int(predicted_status.item()), 'branch_idx':int(branch_idx.item()), 'member_status':int(member_status.item()), 'num_exits':int(model_idx+1)}
                        Statistic_Data.append(results)

                save_path = '/p/project/hai_unganable/projects/multi-exit/multiexit-CCS/plots/statistic_data/' + sdn_training_type
                with open(save_path + '/' + orgin_model_name + '.pkl', "wb") as wf:
                    import pickle as pkl
                    pkl.dump(Statistic_Data, wf)
                    #print("Finish")

                # af.create_path(save_path)
                # df = pd.DataFrame()
                # Statistic_Data = df.append(Statistic_Data, ignore_index=True)
                # Statistic_Data.to_csv(save_path + '/' + orgin_model_name + '.csv')

    

def white_box_ratio_wrt_exit(args, models_path, device='cpu'):
    # args.add_ic = [[1, 3, 5, 7, 10],                     #3000, 2000, 1500, 1000, 500, 100
    # [[3, 7], [2, 6], [2]],                      #9000, 8000, 7000, 6000, 5000, 4000  # 7000, 6000, 5000, 4000, 3000, 2000
    # [[2, 4], [1, 3], [1]],  #600, 500, 400, 300, 200, 100            
    # [1, 3, 5, 7, 10],  #350, 300, 250, 200, 150, 100                
    # ]
    datasets_list = ['cifar10', 'cifar100', 'tinyimagenet']
    models_list = ['vgg', 'resnet', 'mobilenet', 'wideresnet']
    train_type = ['ic_only', 'sdn_training']
    for sdn_training_type in train_type:
        for task in datasets_list:
            args.task = task
            for model in models_list:
                args.model = model
                #sdn_training_type = args.training_type

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

                cnns_sdns = []
                num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))

                if sdn_training_type == 'ic_only':
                    cnns_sdns.append(model_name + '_cnn')
                elif sdn_training_type == 'sdn_training':
                    cnns_sdns.append('None')

                for i in range(num_exits):
                    cnns_sdns.append(model_name + '_sdn')
                Statistic_Data = []
                for model_idx, model_name in  enumerate(cnns_sdns):
                    if model_idx in [0]: continue
                    print(f'------------------model: {model_name}, num_braches:{model_idx+1}-------------------')
                    orgin_model_name = model_name
                    if 'cnn' in model_name:
                        model_name = model_name
                        #save_path = models_path + '/attack/' + model_name
                    elif 'sdn' in model_name:
                        model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type
                        #save_path = models_path + '/attack/' + model_name
                    else:
                        continue
                    #

                    #AttackModelTrainSet = np.load(models_path + f'/shadow/{model_name}/trainset.npy', allow_pickle=True).item()
                    #AttackModelTestSet = np.load(models_path + f'/target/{model_name}/testset.npy', allow_pickle=True).item()
                    #AttackModelTrainSet = pickle.load(open(models_path + f'/shadow/{model_name}/trainset.pkl', 'rb'))#.item()
                    AttackModelTestSet = pickle.load(open(models_path + f'/target/{model_name}/testset.pkl', 'rb'))#.item()
                    if type(AttackModelTestSet) == np.ndarray:
                        #AttackModelTrainSet = AttackModelTrainSet.item()
                        AttackModelTestSet = AttackModelTestSet.item()

                    num_exits = AttackModelTestSet['num_exits']
                    nb_classes = AttackModelTestSet['nb_classes']
                    #AttackModelTrainSet['model_features'] = AttackModelTrainSet['model_features'][:, :512]
                    #feature_length = AttackModelTestSet['model_features'].shape[1]
                    # dict_keys(['model_scores', 'model_loss', 'orginal_labels', 'predicted_labels', 'predicted_status', 'infer_time', 'infer_time_norm', 'branch_idx', 
                    #           'branch_norm', 'member_status', 'early_status', 'num_exits', 'nb_classes', 'model_features', 'model_gradient'])

                    # train_set = torch.utils.data.TensorDataset(
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_scores'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_loss'], dtype='f')),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['orginal_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=nb_classes, return_one_hot=True))).type(torch.float),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.float),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['infer_time'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['infer_time_norm'], dtype='f')),
                    #         torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['branch_idx'], nb_classes=num_exits, return_one_hot=True))).type(torch.float),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['branch_norm'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['early_status'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_features'], dtype='f')),
                    #         torch.from_numpy(np.array(AttackModelTrainSet['model_gradient'], dtype='f')),)
                    test_set = torch.utils.data.TensorDataset(
                            torch.from_numpy(np.array(AttackModelTestSet['data_seed'])).type(torch.long),
                            torch.from_numpy(np.array(AttackModelTestSet['model_scores'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['model_loss'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['orginal_labels'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['predicted_labels'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['predicted_status'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['infer_time'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['infer_time_norm'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['branch_idx'])).type(torch.float),
                            torch.from_numpy(np.array(AttackModelTestSet['branch_norm'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),
                            torch.from_numpy(np.array(AttackModelTestSet['early_status'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['model_features'], dtype='f')),
                            torch.from_numpy(np.array(AttackModelTestSet['model_gradient'], dtype='f')),)
                    #attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
                    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
                    member_count = [0 for _ in range(model_idx+1)]
                    non_member_count = [0 for _ in range(model_idx+1)]
                    for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, infer_time_norm, branch_idx, branch_norm, member_status, early_status, model_features, model_gradient) in enumerate(attack_test_loader):
                        #results = {'model_loss':model_loss.item(), 'predicted_status':predicted_status.item(), 'branch_idx':branch_idx.item(), 'member_status':member_status.item(), 'num_exits':model_idx+1}
                        if member_status.item():
                            member_count[int(branch_idx.item())] += 1
                        else:
                            non_member_count[int(branch_idx.item())] += 1

                    #print(member_count)
                    #print(non_member_count)

                    for exit_idx, (member, non_member) in enumerate(zip(member_count, non_member_count)):
                        if member+non_member != 0:
                            ratio = round(non_member/(member+non_member),2)
                        else:
                            ratio = 1
                        results = {'mem':member, "nonmem":non_member, 'ratio':ratio, 'exit_idx':exit_idx, 'num_exits':model_idx+1}
                        #print(results)
                    ##exit()
                        Statistic_Data.append(results)

                df = pd.DataFrame()
                Statistic_Data = df.append(Statistic_Data, ignore_index=True)
                save_path = '/p/project/hai_unganable/projects/multi-exit/multiexit-CCS/plots/statistic_data/' + sdn_training_type
                af.create_path(save_path)
                Statistic_Data.to_csv(save_path + '/' + orgin_model_name + '_ratio.csv')
