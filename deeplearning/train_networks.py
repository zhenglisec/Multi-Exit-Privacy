
import os
import copy
import torch
import time
import os
import random
import numpy as np

from deeplearning import aux_funcs  as af
from deeplearning import network_architectures as arcs

from architectures.CNNs.VGG import VGG

def train(args, models_path, untrained_models, sdn=False, ic_only_sdn=False, device='cpu'):
    print('Training models...')
    for idx, base_model in enumerate(untrained_models):
        trained_model, model_params = arcs.load_model(models_path, base_model, 0, idx+1)
        #print(model_params)
        dataset = af.get_dataset(model_params['task'], batch_size=128)

        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = model_params['epochs']

        model_params['optimizer'] = 'SGD'
        
        if ic_only_sdn:  # IC-only training, freeze the original weights
            learning_rate = model_params['ic_only']['learning_rate']
            num_epochs = model_params['ic_only']['epochs']
            milestones = model_params['ic_only']['milestones']
            gammas = model_params['ic_only']['gammas']
            model_params['optimizer'] = 'Adam'
            trained_model.ic_only = True

        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        if sdn:
            if ic_only_sdn:
                optimizer, scheduler = af.get_sdn_ic_only_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'/ic_only'

            else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'/sdn_training'
                trained_model.ic_only = False
        else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model

        print('Training: {}...'.format(trained_model_name))
        trained_model.to(device)
        metrics = trained_model.train_func(args, trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def train_sdns(args, models_path, networks, ic_only=False, device='cpu'):
    if ic_only: # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else: # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    for idx, sdn_name in enumerate(networks):
        cnn_to_tune = sdn_name.replace('sdn', 'cnn')
        sdn_params = arcs.load_params(models_path, sdn_name, 0, idx=idx+1)
        sdn_params = arcs.get_net_params(sdn_params['network_type'], sdn_params['task'], sdn_params['add_ic'])
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, load_epoch) # load the CNN and convert it to a SDN
        arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0) # save the resulting SDN
    train(args, models_path, networks, sdn=True, ic_only_sdn=ic_only, device=device)

def train_models(args, models_path, device='cpu'):
    #tasks = ['cifar10', 'cifar100', 'tinyimagenet']    
    cnns = []
    sdns = []
    #for task in tasks:
    if args.model == 'vgg':
        [cnns, sdns] = arcs.create_vgg16bn(models_path, args.task, args.add_ic[0], )
    elif args.model == 'resnet':
        [cnns, sdns] = arcs.create_resnet56(models_path, args.task, args.add_ic[1], )
    elif args.model == 'wideresnet':
        [cnns, sdns] = arcs.create_wideresnet32_4(models_path, args.task, args.add_ic[2], )
    elif args.model == 'mobilenet':
        [cnns, sdns] = arcs.create_mobilenet(models_path, args.task, args.add_ic[3], )
    elif args.model == 'fcn_1':
        [cnns, sdns] = arcs.create_fcn_1(models_path, args.task, args.add_ic[4])
    elif args.model == 'fcn_2':
        [cnns, sdns] = arcs.create_fcn_2(models_path, args.task, args.add_ic[4])
    elif args.model == 'fcn_3':
        [cnns, sdns] = arcs.create_fcn_3(models_path, args.task, args.add_ic[4])
    elif args.model == 'fcn_4':
        [cnns, sdns] = arcs.create_fcn_4(models_path, args.task, args.add_ic[4])
    
    train(args, models_path, cnns, sdn=False, device=device)
    train_sdns(args, models_path, sdns, ic_only=False, device=device) # train SDNs with SDN-training strategy


    