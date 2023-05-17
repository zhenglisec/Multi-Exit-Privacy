# network_architectures.py
# contains the functions to create and save CNNs and SDNs
# VGG, ResNet, Wide ResNet and MobileNet
# also contains the hyper-parameters for model training

import torch

import pickle
import os

import os.path
from deeplearning import aux_funcs as af
import numpy as np

import itertools
import collections
from deeplearning.profiler import profile_sdn

from architectures.SDNs.VGG_SDN import VGG_SDN
from architectures.CNNs.VGG import VGG

from architectures.SDNs.ResNet_SDN import ResNet_SDN
from architectures.CNNs.ResNet import ResNet

from architectures.SDNs.MobileNet_SDN import MobileNet_SDN
from architectures.CNNs.MobileNet import MobileNet

from architectures.SDNs.WideResNet_SDN import WideResNet_SDN
from architectures.CNNs.WideResNet import WideResNet

from architectures.SDNs.FCN_SDN import FCN_SDN_1, FCN_SDN_2, FCN_SDN_3, FCN_SDN_4
from architectures.CNNs.FCN import FCN_1, FCN_2, FCN_3, FCN_4
# from architectures.CNNs.MLP_CNN import fcn_1, fcn_2, fcn_3, fcn_4
# from architectures.SDNs.MLP_SDN_CNN import MLP_SDN_0, MLP_SDN_1, MLP_SDN_2, MLP_SDN_3
from deeplearning.membership_attack_experiments import FeatureExtractor
# xx = 0
def save_networks(model_name, model_params, models_path, save_type):
    # global xx
    cnn_name = model_name+'_cnn'
    sdn_name = model_name+'_sdn'

    if 'c' in save_type:
        print('Saving CNN...')
        model_params['architecture'] = 'cnn'
        model_params['base_model'] = cnn_name
        network_type = model_params['network_type']

        if 'wideresnet' in network_type:
          model = WideResNet(model_params)
        elif 'resnet' in network_type:
            model = ResNet(model_params)
        elif 'vgg' in network_type: 
            model = VGG(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_4(model_params)
        
        save_model(model, model_params, models_path, cnn_name, epoch=0)

    if 'd' in save_type:
        
        print('Saving SDN...')
        model_params['architecture'] = 'sdn'
        model_params['base_model'] = sdn_name
        network_type = model_params['network_type']
        
        if 'wideresnet' in network_type:
            model = WideResNet_SDN(model_params)
        elif 'resnet' in network_type:
            model = ResNet_SDN(model_params)
        elif 'vgg' in network_type: 
            model = VGG_SDN(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_SDN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_SDN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_SDN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_SDN_4(model_params)
        # xx += 1
        # if xx == 5:
        #     print(model)
        #     # x = torch.randn(1,3,32,32)
        #     # CNNFeatureExtractor = FeatureExtractor(model, ['layers.3.output.avg_pool', 'layers.7.output.avg_pool', 'layers.11.output.avg_pool', 'layers.15.output.avg_pool',  'end_layers.1'])
        #     # features = CNNFeatureExtractor(x)
        #     # e = features['layers.3.output.avg_pool']
        #     # a = features['layers.7.output.avg_pool']
        #     # b = features['layers.11.output.avg_pool']
        #     # c = features['layers.15.output.avg_pool']
        #     # print(e.shape)
        #     # print(a.shape)
        #     # print(b.shape)
        #     # print(c.shape)
        #     exit()
        # print(model)
        save_model(model, model_params, models_path, sdn_name, epoch=0)
        
    return cnn_name, sdn_name

def create_vgg16bn(models_path, task, add_ic, get_params=False):
    print('Creating VGG16BN untrained {} models...'.format(task))

    model_params = get_task_params(task)
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]

    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_name = '{}_vgg16bn'.format(task)

    # architecture params
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = False
    #model_params['add_ic'] = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs

    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #for ic_idx in range(len(add_ic)):
    for ic_idx, ic_loc in enumerate(add_ic): #[1, 3, 5, 6, 8, 9]
        IC[ic_loc] = 1
        model_params['add_ic'] = IC
        cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if ic_idx == 0 else 'd')
        sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    
    return cnns, sdns

def create_resnet56(models_path, task, add_ic, get_params=False):
    print('Creating resnet56 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    #model_params['add_ic'] = [[0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0]] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    
    model_name = '{}_resnet56'.format(task)
    model_params['network_type'] = 'resnet56'
    model_params['augment_training'] = False
    model_params['init_weights'] = True

    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for block_idx, block_ic in enumerate(add_ic): #[[3, 7], [2, 6], [1, 5]]
        for ic_idx, ic_loc in enumerate(block_ic):
            IC[block_idx][ic_loc] = 1
            
            model_params['add_ic'] = IC
            
            cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if (block_idx + ic_idx) == 0 else 'd')
            sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    
    return cnns, sdns


def create_wideresnet32_4(models_path, task, add_ic, get_params=False):
    print('Creating wrn32_4 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['num_blocks'] = [5,5,5]
    model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3

    model_name = '{}_wideresnet32_4'.format(task)

    #model_params['add_ic'] = [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    
    model_params['network_type'] = 'wideresnet32_4'
    model_params['augment_training'] = False
    model_params['init_weights'] = True

    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for block_idx, block_ic in enumerate(add_ic): #[[2, 4], [1, 3], [0, 2]]
        for ic_idx, ic_loc in enumerate(block_ic):
            IC[block_idx][ic_loc] = 1
            
            model_params['add_ic'] = IC
            
            cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if (block_idx + ic_idx) == 0 else 'd')
            sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    
    return cnns, sdns


def create_mobilenet(models_path, task, add_ic, get_params=False):
    print('Creating MobileNet untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_mobilenet'.format(task)
    
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    
    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #for ic_idx in range(len(add_ic)):
    for ic_idx, ic_loc in enumerate(add_ic): #[2, 4, 6, 8, 11]
        IC[ic_loc] = 1
        model_params['add_ic'] = IC
        cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if ic_idx == 0 else 'd')
        sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    
    return cnns, sdns

def create_fcn_1(models_path, task, add_ic, get_params=False):
    print('Creating FCN_1 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_1'.format(task)
    
    model_params['network_type'] = 'fcn_1'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    
    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [0, 0, 0, 0, 0]
    for ic_idx, ic_loc in enumerate(add_ic): 
        IC[ic_loc] = 1
        model_params['add_ic'] = IC

        cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if ic_idx == 0 else 'd')
        sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    return cnns, sdns
def create_fcn_2(models_path, task, add_ic, get_params=False):
    print('Creating FCN_2 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_2'.format(task)
    
    model_params['network_type'] = 'fcn_2'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    
    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [0, 0, 0, 0, 0]
    for ic_idx, ic_loc in enumerate(add_ic): 
        IC[ic_loc] = 1
        model_params['add_ic'] = IC

        cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if ic_idx == 0 else 'd')
        sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    return cnns, sdns
def create_fcn_3(models_path, task, add_ic, get_params=False):
    print('Creating FCN_3 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_3'.format(task)
    
    model_params['network_type'] = 'fcn_3'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    
    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [0, 0, 0, 0, 0]
    for ic_idx, ic_loc in enumerate(add_ic): 
        IC[ic_loc] = 1
        model_params['add_ic'] = IC

        cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if ic_idx == 0 else 'd')
        sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    return cnns, sdns

def create_fcn_4(models_path, task, add_ic, get_params=False):
    print('Creating FCN_4 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_4'.format(task)
    
    model_params['network_type'] = 'fcn_4'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    
    get_lr_params(model_params)
    if get_params:
        model_params['add_ic'] = add_ic
        return model_params

    cnns = []
    sdns = []
    IC = [0, 0, 0, 0, 0]
    for ic_idx, ic_loc in enumerate(add_ic): 
        IC[ic_loc] = 1
        model_params['add_ic'] = IC

        cnn, sdn = save_networks(model_name, model_params, models_path, save_type='cd' if ic_idx == 0 else 'd')
        sdns.append(sdn) 

    cnns.append(model_name+'_cnn')
    return cnns, sdns

def get_task_params(task):
    if task == 'cifar10':
        return cifar10_params()
    elif task == 'cifar100':
        return cifar100_params()
    elif task == 'tinyimagenet':
        return tiny_imagenet_params()
    elif task == 'purchase':
        return purchase_params()
    elif task == 'location':
        return location_params()
    elif task == 'texas':
        return texas_params()
    elif task == 'adult':
        return adult_params()
def cifar10_params():
    model_params = {}
    model_params['task'] = 'cifar10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    return model_params

def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    return model_params

def tiny_imagenet_params():
    model_params = {}
    model_params['task'] = 'tinyimagenet'
    model_params['input_size'] = 64
    model_params['num_classes'] = 200
    return model_params

def purchase_params():
    model_params = {}
    model_params['task'] = 'purchase'
    model_params['input_size'] = 600
    model_params['num_classes'] = 100
    # model_params['in_channels'] = [128, 256, 384, 512] seed 123
    model_params['in_channels'] = [1024, 2048, 3702, 4096] 
    return model_params

def texas_params():
    model_params = {}
    model_params['task'] = 'texas'
    model_params['input_size'] = 6168
    model_params['num_classes'] = 100
    # model_params['in_channels'] = [128, 192, 256, 370] seed 123
    model_params['in_channels'] = [1024, 2048, 3702, 4096] 
    return model_params

def location_params():
    model_params = {}
    model_params['task'] = 'location'
    model_params['input_size'] = 446
    model_params['num_classes'] = 30
    # model_params['in_channels'] = [96, 160, 288, 352] seed 123
    model_params['in_channels'] = [1024, 2048, 3702, 4096] 
    return model_params

def adult_params():
    model_params = {}
    model_params['task'] = 'adult'
    model_params['input_size'] = 105
    model_params['num_classes'] = 2
    model_params['in_channels'] = [32, 64, 128, 160]
    return model_params

def get_lr_params(model_params):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type or 'wideresnet' in network_type:
        model_params['weight_decay'] = 0.0005
    elif 'mlp' in network_type:
        model_params['weight_decay'] = 0.0005
    else:
        model_params['weight_decay'] = 0.0001
    
    # only texas mlp2/3 is 0.01, others are 0.02
    if ('fcn_3' in network_type and model_params['task'] == 'texas') or ('fcn_4' in network_type and model_params['task'] == 'texas'):
        model_params['learning_rate'] = 0.01
    elif 'mlp' in network_type:
        model_params['learning_rate'] = 0.02 
    else:
        model_params['learning_rate'] = 0.1
    model_params['epochs'] = 100
    # if model_params['task'] == 'location':
    #     model_params['milestones'] = [30, 60, 80]
    # else:
    #     model_params['milestones'] = [35, 60, 85]
    model_params['milestones'] = [35, 60, 85]
    model_params['gammas'] = [0.1, 0.1, 0.1]

    # SDN ic_only training params
    model_params['ic_only'] = {}
    model_params['ic_only']['learning_rate'] = 0.001 # lr for full network training after sdn modification
    model_params['ic_only']['epochs'] = 25
    model_params['ic_only']['milestones'] = [15]
    model_params['ic_only']['gammas'] = [0.1]
    


def save_model(model, model_params, models_path, model_name, epoch=-1):
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    if 'sdn' in model_name:
        ic_num = sum(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in model_params['add_ic'])))
        if model_name.find('/') > 0:
            t = model_name.find('/')
            model_name_ = model_name[:t]
            sdn_train_type = model_name[t+1:]           
            network_path = models_path + '/' + model_name_ + '/' + str(ic_num) + '/' + sdn_train_type
        else:
            network_path = models_path + '/' + model_name + '/' + str(ic_num)
    else:
        network_path = models_path + '/' + model_name

    if not os.path.exists(network_path):
        os.makedirs(network_path)

    # epoch == 0 is the untrained network, epoch == -1 is the last
    if epoch == 0:
        path =  network_path + '/untrained'
        params_path = network_path + '/parameters_untrained'
    elif epoch == -1:
        path =  network_path + '/last'
        params_path = network_path + '/parameters_last'
    else:
        path = network_path + '/' + str(epoch)
        params_path = network_path + '/parameters_'+str(epoch)

    torch.save(model.state_dict(), path)

    if model_params is not None:
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)

def load_params(models_path, model_name, epoch=0, idx = 0, sdn_training_type=None):
    if ('sdn' in model_name) and (idx!=-1):
        if sdn_training_type == None:
            params_path = models_path + '/' + model_name + '/' + str(idx)
        else:
            params_path = models_path + '/' + model_name + '/' + str(idx) + '/' + sdn_training_type
    else:
        params_path = models_path + '/' + model_name

    if epoch == 0:
        params_path = params_path + '/parameters_untrained'
    else:
        params_path = params_path + '/parameters_last'

    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params

def load_model(models_path, model_name, epoch=0, idx=-1, sdn_training_type=None):
    model_params = load_params(models_path, model_name, epoch, idx=idx, sdn_training_type=sdn_training_type)

    architecture = 'empty' if 'architecture' not in model_params else model_params['architecture'] 
    network_type = model_params['network_type']

    if architecture == 'sdn' or 'sdn' in model_name:
            
        if 'wideresnet' in network_type:
            model = WideResNet_SDN(model_params)
        elif 'resnet' in network_type:
            model = ResNet_SDN(model_params)
        elif 'vgg' in network_type:
            model = VGG_SDN(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_SDN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_SDN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_SDN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_SDN_4(model_params)
        
        if sdn_training_type == None:
            network_path = models_path + '/' + model_name + '/' + str(idx)
        else:
            network_path = models_path + '/' + model_name + '/' + str(idx) + '/' + sdn_training_type

    elif architecture == 'cnn' or 'cnn' in model_name:
        if 'wideresnet' in network_type:
            model = WideResNet(model_params)
        elif 'resnet' in network_type:
            model = ResNet(model_params)
        elif 'vgg' in network_type:
            model = VGG(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_4(model_params)
        network_path = models_path + '/' + model_name

    if epoch == 0: # untrained model
        load_path = network_path + '/untrained'
    elif epoch == -1: # last model
        load_path = network_path + '/last'
    else:
        load_path = network_path + '/' + str(epoch)
        
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(load_path), strict=False)
    else:
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)

    return model, model_params

def get_sdn(cnn):
    if (isinstance(cnn, VGG)):
        return VGG_SDN
    elif (isinstance(cnn, ResNet)):
        return ResNet_SDN
    elif (isinstance(cnn, WideResNet)):
        return WideResNet_SDN
    elif (isinstance(cnn, MobileNet)):
        return MobileNet_SDN
    elif (isinstance(cnn, FCN_1)):
        return FCN_SDN_1
    elif (isinstance(cnn, FCN_2)):
        return FCN_SDN_2
    elif (isinstance(cnn, FCN_3)):
        return FCN_SDN_3
    elif (isinstance(cnn, FCN_4)):
        return FCN_SDN_4
def get_cnn(sdn):
    if (isinstance(sdn, VGG_SDN)):
        return VGG
    elif (isinstance(sdn, ResNet_SDN)):
        return ResNet
    elif (isinstance(sdn, WideResNet_SDN)):
        return WideResNet
    elif (isinstance(sdn, MobileNet_SDN)):
        return MobileNet
    elif (isinstance(sdn, FCN_SDN_1)):
        return FCN_1
    elif (isinstance(sdn, FCN_SDN_2)):
        return FCN_2
    elif (isinstance(sdn, FCN_SDN_3)):
        return FCN_3
    elif (isinstance(sdn, FCN_SDN_4)):
        return FCN_4
def get_net_params(net_type, task, add_ic):
    if net_type == 'vgg16':
        return create_vgg16bn(None, task,  add_ic, True)
    elif net_type == 'resnet56':
        return create_resnet56(None, task,  add_ic, True)
    elif net_type == 'wideresnet32_4':
        return create_wideresnet32_4(None, task,  add_ic, True)
    elif net_type == 'mobilenet':
        return create_mobilenet(None, task,  add_ic, True)
    elif net_type == 'fcn_1':
        return create_fcn_1(None,  task,  add_ic, True)
    elif net_type == 'fcn_2':
        return create_fcn_2(None,  task,  add_ic, True)
    elif net_type == 'fcn_3':
        return create_fcn_3(None,  task,  add_ic, True)
    elif net_type == 'fcn_4':
        return create_fcn_4(None,  task,  add_ic, True)