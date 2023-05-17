from enum import unique
import imp
import torch
import numpy as np
import os
import itertools, collections
import deeplearning.aux_funcs  as af
import deeplearning.model_funcs as mf
from deeplearning import network_architectures as arcs
from deeplearning.profiler import profile_sdn, profile
from deeplearning import data as DATA
from time import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
# import hkdf
import hashlib, sys
from binascii import hexlify, unhexlify
from PIL import Image
from deeplearning import imagehash, hkdf
# import imagehash
model_feature_len = {
    'cifar10_vgg16bn': 512,
    'cifar10_resnet56': 512,
    'cifar10_mobilenet': 1024,
    'cifar10_wideresnet32': 512,
    'cifar100_vgg16bn': 512,
    'cifar100_resnet56': 512,
    'cifar100_mobilenet': 1024,
    'cifar100_wideresnet32': 512,
    'tinyimagenet_vgg16bn': 1024,
    'tinyimagenet_resnet56': 512,
    'tinyimagenet_mobilenet': 1024,
    'tinyimagenet_wideresnet32': 512,

    # add new
    'purchase_mlp_0': 256,
    'purchase_mlp_1': 256,
    'purchase_mlp_2': 256,
    'purchase_mlp_3': 256,
    'texas_mlp_0': 256,
    'texas_mlp_1': 256,
    'texas_mlp_2': 256,
    'texas_mlp_3': 256,
    'location_mlp_0': 256,
    'location_mlp_1': 256,
    'location_mlp_2': 256,
    'location_mlp_3': 256,
    'adult_mlp_0': 256,
    'adult_mlp_1': 256,
    'adult_mlp_2': 256,
    'adult_mlp_3': 256,
}   

model_layer_index = {
    #ok
    'vgg16bn_cnn':   ['end_layers.0'], 
    'vgg16bn_sdn_1': ['layers.1.output.layer_0', 'end_layers.0'],
    'vgg16bn_sdn_2': ['layers.1.output.layer_0', 'layers.3.output.layer_0', 'end_layers.0'],
    'vgg16bn_sdn_3': ['layers.1.output.layer_0', 'layers.3.output.layer_0',  'layers.5.output.layer_0', 'end_layers.0'],
    'vgg16bn_sdn_4': ['layers.1.output.layer_0', 'layers.3.output.layer_0',  'layers.5.output.layer_0', 'layers.7.output.layer_0', 'end_layers.0'],
    'vgg16bn_sdn_5': ['layers.1.output.layer_0', 'layers.3.output.layer_0',  'layers.5.output.layer_0', 'layers.7.output.layer_0', 'layers.10.output.layer_0', 'end_layers.0'],
    
    #ok
    'resnet56_cnn':   ['end_layers.2'], 
    'resnet56_sdn_1': ['layers.3.output.layer_0', 'end_layers.2'],
    'resnet56_sdn_2': ['layers.3.output.layer_0', 'layers.7.output.layer_0', 'end_layers.2'],
    'resnet56_sdn_3': ['layers.3.output.layer_0', 'layers.7.output.layer_0', 'layers.11.output.layer_0', 'end_layers.2'],
    'resnet56_sdn_4': ['layers.3.output.layer_0', 'layers.7.output.layer_0', 'layers.11.output.layer_0', 'layers.15.output.layer_0', 'end_layers.2'],
    'resnet56_sdn_5': ['layers.3.output.layer_0', 'layers.7.output.layer_0', 'layers.11.output.layer_0', 'layers.15.output.layer_0', 'layers.20.output.layer_0', 'end_layers.2'],
    
    #ok
    'mobilenet_cnn':   ['end_layers.2'],
    'mobilenet_sdn_1': ['layers.1.output.layer_0', 'end_layers.2'],
    'mobilenet_sdn_2': ['layers.1.output.layer_0', 'layers.3.output.layer_0', 'end_layers.2'],
    'mobilenet_sdn_3': ['layers.1.output.layer_0', 'layers.3.output.layer_0', 'layers.5.output.layer_0', 'end_layers.2'],
    'mobilenet_sdn_4': ['layers.1.output.layer_0', 'layers.3.output.layer_0', 'layers.5.output.layer_0', 'layers.7.output.layer_0', 'end_layers.2'],
    'mobilenet_sdn_5': ['layers.1.output.layer_0', 'layers.3.output.layer_0', 'layers.5.output.layer_0', 'layers.7.output.layer_0', 'layers.10.output.layer_0', 'end_layers.2'],

    # ok
    'wideresnet32_4_cnn':   ['end_layers.4'], 
    'wideresnet32_4_sdn_1': ['layers.2.output.layer_0', 'end_layers.4'],
    'wideresnet32_4_sdn_2': ['layers.2.output.layer_0', 'layers.4.output.layer_0', 'end_layers.4'],
    'wideresnet32_4_sdn_3': ['layers.2.output.layer_0', 'layers.4.output.layer_0', 'layers.6.output.layer_0', 'end_layers.4'],
    'wideresnet32_4_sdn_4': ['layers.2.output.layer_0', 'layers.4.output.layer_0', 'layers.6.output.layer_0', 'layers.8.output.layer_0', 'end_layers.4'],
    'wideresnet32_4_sdn_5': ['layers.2.output.layer_0', 'layers.4.output.layer_0', 'layers.6.output.layer_0', 'layers.8.output.layer_0', 'layers.11.output.layer_0', 'end_layers.4'],
    
    #ok
    'mlp_0_cnn':   ['end_layers.2'], 
    'mlp_0_sdn_1': ['layers.0.output.0', 'end_layers.2'],
    'mlp_0_sdn_2': ['layers.0.output.0', 'layers.1.output.0', 'end_layers.2'],
    'mlp_0_sdn_3': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'end_layers.2'],
    'mlp_0_sdn_4': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'end_layers.2'],
    'mlp_0_sdn_5': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'layers.4.output.0', 'end_layers.2'],

    #ok
    'mlp_1_cnn':   ['end_layers.2'], 
    'mlp_1_sdn_1': ['layers.0.output.0', 'end_layers.2'],
    'mlp_1_sdn_2': ['layers.0.output.0', 'layers.1.output.0', 'end_layers.2'],
    'mlp_1_sdn_3': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'end_layers.2'],
    'mlp_1_sdn_4': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'end_layers.2'],
    'mlp_1_sdn_5': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'layers.4.output.0', 'end_layers.2'],

    #ok
    'mlp_2_cnn':   ['end_layers.2'], 
    'mlp_2_sdn_1': ['layers.0.output.0', 'end_layers.2'],
    'mlp_2_sdn_2': ['layers.0.output.0', 'layers.1.output.0', 'end_layers.2'],
    'mlp_2_sdn_3': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'end_layers.2'],
    'mlp_2_sdn_4': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'end_layers.2'],
    'mlp_2_sdn_5': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'layers.4.output.0', 'end_layers.2'],

    #ok
    'mlp_3_cnn':   ['end_layers.2'], 
    'mlp_3_sdn_1': ['layers.0.output.0', 'end_layers.2'],
    'mlp_3_sdn_2': ['layers.0.output.0', 'layers.1.output.0', 'end_layers.2'],
    'mlp_3_sdn_3': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'end_layers.2'],
    'mlp_3_sdn_4': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'end_layers.2'],
    'mlp_3_sdn_5': ['layers.0.output.0', 'layers.1.output.0',  'layers.2.output.0', 'layers.3.output.0', 'layers.4.output.0', 'end_layers.2'],
}

gradient_layer_index = {
    #ok
    'vgg16bn_cnn':   ['end_layers.2.weight'], 
    'vgg16bn_sdn_1': ['layers.1.output.layer_2.weight', 'end_layers.2.weight'],
    'vgg16bn_sdn_2': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight', 'end_layers.2.weight'],
    'vgg16bn_sdn_3': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight',  'layers.5.output.layer_2.weight', 'end_layers.2.weight'],
    'vgg16bn_sdn_4': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight',  'layers.5.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'end_layers.2.weight'],
    'vgg16bn_sdn_5': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight',  'layers.5.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'layers.10.output.layer_2.weight', 'end_layers.2.weight'],
    
    #ok
    'resnet56_cnn':   ['end_layers.4.weight'], 
    'resnet56_sdn_1': ['layers.3.output.layer_2.weight', 'end_layers.4.weight'],
    'resnet56_sdn_2': ['layers.3.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'end_layers.4.weight'],
    'resnet56_sdn_3': ['layers.3.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'layers.11.output.layer_2.weight', 'end_layers.4.weight'],
    'resnet56_sdn_4': ['layers.3.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'layers.11.output.layer_2.weight', 'layers.15.output.layer_2.weight', 'end_layers.4.weight'],
    'resnet56_sdn_5': ['layers.3.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'layers.11.output.layer_2.weight', 'layers.15.output.layer_2.weight', 'layers.20.output.layer_2.weight', 'end_layers.4.weight'],
    
    #ok
    'mobilenet_cnn':   ['end_layers.4.weight'],
    'mobilenet_sdn_1': ['layers.1.output.layer_2.weight', 'end_layers.4.weight'],
    'mobilenet_sdn_2': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight', 'end_layers.4.weight'],
    'mobilenet_sdn_3': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight', 'layers.5.output.layer_2.weight', 'end_layers.4.weight'],
    'mobilenet_sdn_4': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight', 'layers.5.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'end_layers.4.weight'],
    'mobilenet_sdn_5': ['layers.1.output.layer_2.weight', 'layers.3.output.layer_2.weight', 'layers.5.output.layer_2.weight', 'layers.7.output.layer_2.weight', 'layers.10.output.layer_2.weight', 'end_layers.4.weight'],

    # ok
    'wideresnet32_4_cnn':   ['end_layers.6.weight'], 
    'wideresnet32_4_sdn_1': ['layers.2.output.layer_2.weight', 'end_layers.6.weight'],
    'wideresnet32_4_sdn_2': ['layers.2.output.layer_2.weight', 'layers.4.output.layer_2.weight', 'end_layers.6.weight'],
    'wideresnet32_4_sdn_3': ['layers.2.output.layer_2.weight', 'layers.4.output.layer_2.weight', 'layers.6.output.layer_2.weight', 'end_layers.6.weight'],
    'wideresnet32_4_sdn_4': ['layers.2.output.layer_2.weight', 'layers.4.output.layer_2.weight', 'layers.6.output.layer_2.weight', 'layers.8.output.layer_2.weight', 'end_layers.6.weight'],
    'wideresnet32_4_sdn_5': ['layers.2.output.layer_2.weight', 'layers.4.output.layer_2.weight', 'layers.6.output.layer_2.weight', 'layers.8.output.layer_2.weight', 'layers.11.output.layer_2.weight', 'end_layers.6.weight'],

    #ok
    'mlp_0_cnn':   ['end_layers.4.weight'], 
    'mlp_0_sdn_1': ['layers.0.output.2.weight', 'end_layers.4.weight'],
    'mlp_0_sdn_2': ['layers.0.output.2.weight', 'layers.1.output.2.weight', 'end_layers.4.weight'],
    'mlp_0_sdn_3': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'end_layers.4.weight'],
    'mlp_0_sdn_4': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'end_layers.4.weight'],
    'mlp_0_sdn_5': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'layers.4.output.2.weight', 'end_layers.4.weight'],

    #ok
    'mlp_1_cnn':   ['end_layers.4.weight'], 
    'mlp_1_sdn_1': ['layers.0.output.2.weight', 'end_layers.4.weight'],
    'mlp_1_sdn_2': ['layers.0.output.2.weight', 'layers.1.output.2.weight', 'end_layers.4.weight'],
    'mlp_1_sdn_3': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'end_layers.4.weight'],
    'mlp_1_sdn_4': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'end_layers.4.weight'],
    'mlp_1_sdn_5': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'layers.4.output.2.weight', 'end_layers.4.weight'],

    #ok
    'mlp_2_cnn':   ['end_layers.4.weight'], 
    'mlp_2_sdn_1': ['layers.0.output.2.weight', 'end_layers.4.weight'],
    'mlp_2_sdn_2': ['layers.0.output.2.weight', 'layers.1.output.2.weight', 'end_layers.4.weight'],
    'mlp_2_sdn_3': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'end_layers.4.weight'],
    'mlp_2_sdn_4': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'end_layers.4.weight'],
    'mlp_2_sdn_5': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'layers.4.output.2.weight', 'end_layers.4.weight'],

    #ok
    'mlp_3_cnn':   ['end_layers.4.weight'], 
    'mlp_3_sdn_1': ['layers.0.output.2.weight', 'end_layers.4.weight'],
    'mlp_3_sdn_2': ['layers.0.output.2.weight', 'layers.1.output.2.weight', 'end_layers.4.weight'],
    'mlp_3_sdn_3': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'end_layers.4.weight'],
    'mlp_3_sdn_4': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'end_layers.4.weight'],
    'mlp_3_sdn_5': ['layers.0.output.2.weight', 'layers.1.output.2.weight',  'layers.2.output.2.weight', 'layers.3.output.2.weight', 'layers.4.output.2.weight', 'end_layers.4.weight'],
}


def get_gradient(model, loss, exit_idx, gradient_layers):
        gradient_list = list(model.named_parameters())
        gradients = []
        #for loss in losses:
        loss.backward()
        for name, parameter in gradient_list:
            if gradient_layers[exit_idx] in name:
                gradient = parameter.grad.clone()
                gradients.append(gradient.unsqueeze_(0).unsqueeze_(0))
                break
        gradients = torch.cat(gradients, 0)
        return gradients

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        
        self._features = {layer: torch.empty(0) for layer in layers}
        a = dict([*self.model.named_modules()])
       
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features

def build_membership_dataset(args, models_path, device='cpu'):
    global_salt = 111
    #sdn_training_type = 'ic_only' # IC-only training
    sdn_training_type = args.training_type # SDN training
    #device ='cpu'
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
    num_exits= len(list(itertools.chain.from_iterable(item if isinstance(item, collections.Iterable) else [item] for item in add_ic)))
    if sdn_training_type == 'ic_only':
        cnns_sdns.append(model_name + '_cnn')
    elif sdn_training_type == 'sdn_training':
        #cnns_sdns.append('None')
        cnns_sdns.append(model_name + '_cnn')
        
    for i in range(num_exits):
        cnns_sdns.append(model_name + '_sdn')

    thds_path = f'networks/{args.seed}/target/{model_name}_sdn/threshold/{sdn_training_type}/thds.txt'
    with open(thds_path, "r") as f:  # 打开文件
        confidence_thresholds = f.read()  # 读取文件
        confidence_thresholds = str(confidence_thresholds).split()
        confidence_thresholds = [float(x) for x in confidence_thresholds]
    print(confidence_thresholds)

    for model_idx, model_name in  enumerate(cnns_sdns):
        # if model_idx in [1,2,4,5,6]:
        #     continue
        # else:
        #     pass
        # if model_idx < (args.finish_exits - 1):
        #     continue
        print(f'------------------model: {model_name}, num_exits:{model_idx+1}-------------------')
        if 'cnn' in model_name:
            cnn_model, cnn_params = arcs.load_model(models_path, model_name, epoch=-1)
            MODEL = cnn_model.to(device)

            dataset = af.get_dataset(cnn_params['task'], batch_size=1)
            if args.mode == 'target':
                print('load target_dataset ... ')
                train_loader = dataset.target_train_loader
                test_loader = dataset.target_test_loader
            elif args.mode == 'shadow':
                print('load shadow_dataset ...')
                train_loader = dataset.shadow_train_loader
                test_loader = dataset.shadow_test_loader

            feature_layers = model_layer_index[model_name[model_name.find('_')+1:]]
            gradient_layers = gradient_layer_index[model_name[model_name.find('_')+1:]]
            
        elif 'sdn' in model_name:
            sdn_model, sdn_params = arcs.load_model(models_path, model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            MODEL = sdn_model.to(device)    

            # to test early-exits with the SDN
            dataset = af.get_dataset(sdn_params['task'], batch_size=1)
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

            feature_layers = model_layer_index[model_name[model_name.find('_')+1:]+f'_{model_idx}']
            gradient_layers = gradient_layer_index[model_name[model_name.find('_')+1:]+f'_{model_idx}']
      
            #confidence_thresholds = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] # search for the confidence threshold for early exits
            MODEL.forward = MODEL.early_exit_seq
            MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds

            print(f'--------------theshold:{MODEL.confidence_threshold}----------------')
        else:
            continue

        MODEL.eval()
        feature_extractor = FeatureExtractor(MODEL, feature_layers)

        #with torch.no_grad():
        
        for loader_idx, data_loader in enumerate([train_loader, test_loader]):
            # GPUtil.showUtilization()
            
            top1 = DATA.AverageMeter()
            for data_idx, (data, target) in enumerate(data_loader):
                
                
                data_seed = data.numpy()
                if args.task in ['cifar10', 'cifar100', 'tinyimagenet']:
                    data_seed = (data_seed) * 255
                    data_seed = data_seed.reshape(data_seed.shape[0], data_seed.shape[2], data_seed.shape[3], data_seed.shape[1])
                    data_seed = Image.fromarray(data_seed[0].astype('uint8')).convert('L').resize((8, 8), Image.ANTIALIAS)
                    data_seed = np.asarray(data_seed)
                    avg = np.mean(data_seed)
                    diff = data_seed > avg
                    unique_hash = imagehash.ImageHash(diff).__str__()
                else:
                    print('------------------------------')
                    #print(data_seed)
                    # data_seed = np.tile((data_seed+0.5)*255,(3,data_seed.shape[1],1))
                    # data_seed = data_seed.reshape(data_seed.shape[1], data_seed.shape[2], data_seed.shape[0])
                    # data_seed = Image.fromarray(data_seed.astype('uint8')).convert('L').resize((8, 8), Image.ANTIALIAS)
                    # data_seed = np.asarray(data_seed)
                    # avg = np.mean(data_seed)
                    # diff = data_seed > avg
                    # unique_imagehash = imagehash.ImageHash(diff).__str__()
                    unique_hash = ""
                    for id in data_seed[0]:
                        unique_hash += str(int(id))
                    sha512 = hashlib.sha512()
                    sha512.update(unique_hash.encode('utf-8'))
                    unique_hash = sha512.hexdigest()
     
                kdf = hkdf.Hkdf(bytearray((global_salt,) * 512), b'asecretseed')
                key = kdf.expand(unique_hash.encode('utf-8'), 1)
                unique_seed = ord(key)
                
                #os.system('nvidia-smi')
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                repetitions = 1
                batch_time = np.zeros((repetitions,1))
                
                data, target = data.to(device), target.to(device)
                if 'cnn' in model_name:
                    #for rep in range(repetitions):
                    starter.record()
                    batch_logit = MODEL(data)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)

                    for rep in range(repetitions):
                        batch_time[rep] = curr_time

                    exit_idx = 0
                    is_early = False
                    num_exits = 1
                    features = list(feature_extractor(data).values())[exit_idx].view(1, -1).detach().cpu().numpy()
                    
                    if 'mlp' in model_name:
                        model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+5)]]
                    else:
                        model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+1)]]
                        
                    if 'mlp' in model_name:
                        pass
                    elif features.shape[1] < model_feature_len_:
                        feature_len = int(model_feature_len_/features.shape[1])
                        feature_list = [features for _ in range(feature_len)]
                        features = np.concatenate(feature_list, axis=1)
                elif 'sdn' in model_name:
                    if args.task in ['purchase', 'texas', 'location']:
                        dummy_input = torch.randn(1, dataset.input_size, dtype=torch.float).to(device)
                    else:
                        dummy_input = torch.randn(1, 3, dataset.img_size, dataset.img_size,dtype=torch.float).to(device)
                    #GPU-WARM-UP
                    for _ in range(repetitions):
                        _, _, _ = MODEL(dummy_input)
                    for rep in range(repetitions):
                        starter.record()
                        batch_logit, exit_idx, is_early = MODEL(data)
                        ender.record()
                        # WAIT FOR GPU SYNC
                        torch.cuda.synchronize()
                        curr_time = starter.elapsed_time(ender)
                        batch_time[rep] = curr_time

                    num_exits = model_idx+1
                    # print(feature_extractor(data))
                    # print(exit_idx)
                    # print(MODEL)
                    # exit()
                    features = list(feature_extractor(data).values())[exit_idx].view(1, -1).detach().cpu().numpy()
                    if 'mlp' in model_name:
                        model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+5)]]
                    else:
                        model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+1)]]

                    if 'mlp' in model_name:
                        pass
                    else:
                        if features.shape[1] < model_feature_len_:
                            feature_len = int(model_feature_len_/features.shape[1])
                            feature_list = [features for _ in range(feature_len)]
                            features = np.concatenate(feature_list, axis=1)
                        if exit_idx == 1:
                            features = np.concatenate((features, features), axis=1)
                batch_time = np.array(batch_time)
                batch_time = batch_time.reshape(-1)
                _, batch_predict_label = batch_logit.max(1)
                batch_predict_label = batch_predict_label.long().cpu().detach().numpy()
                batch_orginal_label = target.long().cpu().detach().numpy()
                batch_score = F.softmax(batch_logit, dim=1)
                batch_loss = F.cross_entropy(batch_logit, target, size_average=False)
                batch_gradient = get_gradient(MODEL, batch_loss, exit_idx, gradient_layers)
                
                batch_score = batch_score.cpu().detach().numpy()
                batch_loss = [batch_loss.cpu().detach().numpy()]
                batch_gradient = batch_gradient.cpu().detach().numpy()
                batch_predicted_status = (torch.argmax(batch_logit, dim=1) == target).float().cpu().detach().numpy()
                batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)
                member = int(1 - loader_idx)   

                data_seed           = [unique_seed]             if loader_idx == 0 and data_idx == 0     else np.concatenate((data_seed, unique_seed), axis=0)
                model_score         = batch_score               if loader_idx == 0 and data_idx == 0     else np.concatenate((model_score, batch_score), axis=0)
                model_loss          = batch_loss                if loader_idx == 0 and data_idx == 0     else np.concatenate((model_loss, batch_loss), axis=0)
                orginal_labels      = batch_orginal_label       if loader_idx == 0 and data_idx == 0     else np.concatenate((orginal_labels, batch_orginal_label), axis=0)
                predicted_labels    = batch_predict_label       if loader_idx == 0 and data_idx == 0     else np.concatenate((predicted_labels, batch_predict_label), axis=0)
                predicted_status    = batch_predicted_status    if loader_idx == 0 and data_idx == 0     else np.concatenate((predicted_status, batch_predicted_status), axis=0)
                infer_time          = [batch_time]              if loader_idx == 0 and data_idx == 0     else np.concatenate((infer_time, [batch_time]), axis=0)
                branch_idx          = [exit_idx]                if loader_idx == 0 and data_idx == 0     else np.concatenate((branch_idx, [exit_idx]), axis=0)
                #branch_norm         = [exit_idx/num_exits]      if loader_idx == 0 and data_idx == 0     else np.concatenate((branch_norm, [exit_idx/num_exits]), axis=0)
                member_status       = [member]                  if loader_idx == 0 and data_idx == 0     else np.concatenate((member_status, [member]), axis=0)
                early_status        = [is_early]                if loader_idx == 0 and data_idx == 0     else np.concatenate((early_status, [is_early]), axis=0)
                model_features      = features                  if loader_idx == 0 and data_idx == 0     else np.concatenate((model_features, features), axis=0)
                model_gradient      = batch_gradient            if loader_idx == 0 and data_idx == 0     else np.concatenate((model_gradient, batch_gradient), axis=0)
                # print(infer_time)
                prec1, _ = DATA.accuracy(batch_logit, target, topk=(1,5))
                top1.update(prec1[0], batch_logit.size(0))
                # print(data_idx, exit_idx)
                # if exit_idx == 5:
                #     exit()
                # print(infer_time)
                # print(loader_idx, data_idx)
                if args.task in ['location']:
                    if data_idx == 1999:
                        break
                elif data_idx == 4999:
                    break
            top1_acc = top1.avg.data.cpu().numpy()[()]
            print(f'------------member_status: {1-loader_idx}, top1 acc: {top1_acc}---------')
           
        data = {'data_seed':data_seed,
                'model_scores':model_score, 
                'model_loss':model_loss,
                'orginal_labels':orginal_labels,
                'predicted_labels':predicted_labels,
                'predicted_status':predicted_status,   
                'infer_time':infer_time,         
                #'infer_time_norm':normalization(infer_time),
                'exit_idx':branch_idx,
                #'branch_norm':branch_norm,
                'member_status':member_status,
                'early_status':early_status,
                'num_exits':num_exits,
                'nb_classes':dataset.num_classes,
                'model_features':model_features,
                'model_gradient':model_gradient
                }
 
        dataset_type = 'testset' if args.mode=='target' else 'trainset'
        if 'cnn' in model_name:
            model_name = model_name
        elif 'sdn' in model_name:
            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type

        pickle.dump(data, open(models_path + f'/{model_name}/{dataset_type}.pkl', 'wb'), protocol=4)
        print('save dataset ...', models_path + f'/{model_name}/{dataset_type}.pkl')
        print()

def build_membership_dataset_cpu(args, models_path, device='cpu'):
    global_salt = 111
    #sdn_training_type = 'ic_only' # IC-only training
    sdn_training_type = args.training_type # SDN training
    device ='cpu'
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

    thds_path = 'networks/'+str(args.seed)+'/target/' + model_name + '_sdn/threshold/' + sdn_training_type + '/thds.txt'
    with open(thds_path, "r") as f:  # 打开文件
        confidence_thresholds = f.read()  # 读取文件
        confidence_thresholds = str(confidence_thresholds).split()
        confidence_thresholds = [float(x) for x in confidence_thresholds]
    print(confidence_thresholds)

    for model_idx, model_name in  enumerate(cnns_sdns):
        if model_idx in [4,5]:
            continue
        # else:
        #     pass
        # if model_idx < (args.finish_exits - 1):
        #     continue
        print(f'------------------model: {model_name}, num_exits:{model_idx+1}-------------------')
        if 'cnn' in model_name:
            cnn_model, cnn_params = arcs.load_model(models_path, model_name, epoch=-1)
            MODEL = cnn_model.to(device)
            #print(cnn_params)
            
            dataset = af.get_dataset(cnn_params['task'], batch_size=1)
            if args.mode == 'target':
                print('load target_dataset ... ')
                train_loader = dataset.target_train_loader
                test_loader = dataset.target_test_loader
            elif args.mode == 'shadow':
                print('load shadow_dataset ...')
                train_loader = dataset.shadow_train_loader
                test_loader = dataset.shadow_test_loader

            feature_layers = model_layer_index[model_name[model_name.find('_')+1:]]
            gradient_layers = gradient_layer_index[model_name[model_name.find('_')+1:]]
        elif 'sdn' in model_name:
            sdn_model, sdn_params = arcs.load_model(models_path, model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            MODEL = sdn_model.to(device)    
            # to test early-exits with the SDN
            dataset = af.get_dataset(sdn_params['task'], batch_size=1)
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

            #feature_layers = model_layer_index[model_name[model_name.find('_')+1:]+f'_{model_idx}']
            #gradient_layers = gradient_layer_index[model_name[model_name.find('_')+1:]+f'_{model_idx}']
            
            #confidence_thresholds = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] # search for the confidence threshold for early exits
            MODEL.forward = MODEL.early_exit_seq
            MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds

            print(f'--------------theshold:{MODEL.confidence_threshold}----------------')
        else:
            continue
        #print(MODEL)
        #feature_extractor = FeatureExtractor(MODEL, feature_layers)

        # model_score = None
        # model_loss = None
        # orginal_labels = None
        # predicted_labels = None
        # predicted_status = None
        # infer_time = None
        # exit_idx = None
        # #branch_norm = None
        # member_status = None
        # early_status = None
        # #model_features = None
        # #model_gradient = None

        #with torch.no_grad():
        MODEL.eval()
        for loader_idx, data_loader in enumerate([train_loader, test_loader]):
            # GPUtil.showUtilization()
            
            top1 = DATA.AverageMeter()
            for data_idx, (data, target) in enumerate(data_loader):
                data_seed = data.numpy()
                if args.task in ['cifar10', 'cifar100', 'tinyimagenet']:
                    data_seed = (data_seed) * 255
                    data_seed = data_seed.reshape(data_seed.shape[0], data_seed.shape[2], data_seed.shape[3], data_seed.shape[1])
                    data_seed = Image.fromarray(data_seed[0].astype('uint8')).convert('L').resize((8, 8), Image.ANTIALIAS)
                    data_seed = np.asarray(data_seed)
                    avg = np.mean(data_seed)
                    diff = data_seed > avg
                    unique_hash = imagehash.ImageHash(diff).__str__()
                else:
                    print('------------------------------')
                    #print(data_seed)
                    # data_seed = np.tile((data_seed+0.5)*255,(3,data_seed.shape[1],1))
                    # data_seed = data_seed.reshape(data_seed.shape[1], data_seed.shape[2], data_seed.shape[0])
                    # data_seed = Image.fromarray(data_seed.astype('uint8')).convert('L').resize((8, 8), Image.ANTIALIAS)
                    # data_seed = np.asarray(data_seed)
                    # avg = np.mean(data_seed)
                    # diff = data_seed > avg
                    # unique_imagehash = imagehash.ImageHash(diff).__str__()
                    unique_hash = ""
                    for id in data_seed[0]:
                        unique_hash += str(int(id))
                    sha512 = hashlib.sha512()
                    sha512.update(unique_hash.encode('utf-8'))
                    unique_hash = sha512.hexdigest()
     
                kdf = hkdf.Hkdf(bytearray((global_salt,) * 512), b'asecretseed')
                key = kdf.expand(unique_hash.encode('utf-8'), 1)
                unique_seed = ord(key)

                #os.system('nvidia-smi')
                #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                repetitions = 20
                batch_time = np.zeros((repetitions,1))
                
                data, target = data.to(device), target.to(device)
                if 'cnn' in model_name:
                    #for rep in range(repetitions):
                    starter = time()
                    batch_logit = MODEL(data)
                    ender = time()
                    #torch.cuda.synchronize()
                    curr_time = ender - starter

                    for rep in range(repetitions):
                        batch_time[rep] = curr_time

                    exit_idx = 0
                    is_early = False
                    num_exits = 1
                    #features = list(feature_extractor(data).values())[exit_idx].view(1, -1).detach().cpu().numpy()
                    
                    # model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+1)]]

                    # # print(features.shape)
                    # # print(model_feature_len_)
                    # # exit()
                    # if features.shape[1] < model_feature_len_:
                    #     feature_len = int(model_feature_len_/features.shape[1])
                    #     feature_list = [features for _ in range(feature_len)]
                    #     features = np.concatenate(feature_list, axis=1)
                elif 'sdn' in model_name:
                    # dummy_input = torch.randn(1, 3, dataset.img_size, dataset.img_size,dtype=torch.float).to(device)
                    # #GPU-WARM-UP
                    # for _ in range(repetitions):
                    #     _, _, _ = MODEL(dummy_input)
                    for rep in range(repetitions):
                        starter  = time()
                        batch_logit, exit_idx, is_early = MODEL(data)
                        ender = time()
                        # WAIT FOR GPU SYNC
                        #torch.cuda.synchronize()
                        curr_time = ender - starter
                        batch_time[rep] = curr_time




                    num_exits = model_idx+1
                    #features = list(feature_extractor(data).values())[exit_idx].view(1, -1).detach().cpu().numpy()

                    #model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+1)]]
                    # if features.shape[1] < model_feature_len_:
                    #     feature_len = int(model_feature_len_/features.shape[1])
                    #     feature_list = [features for _ in range(feature_len)]
                    #     features = np.concatenate(feature_list, axis=1)
                    # if exit_idx == 1:
                    #     features = np.concatenate((features, features), axis=1)
                batch_time = np.array(batch_time)
                batch_time = batch_time.reshape(-1)
                _, batch_predict_label = batch_logit.max(1)
                batch_predict_label = batch_predict_label.long().detach().numpy()
                batch_orginal_label = target.long().detach().numpy()
                batch_score = F.softmax(batch_logit, dim=1)
                batch_loss = F.cross_entropy(batch_logit, target, size_average=False)
                #batch_gradient = get_gradient(MODEL, batch_loss, exit_idx, gradient_layers)
                
                batch_score = batch_score.detach().numpy()
                batch_loss = [batch_loss.detach().numpy()]
                #batch_gradient = batch_gradient.detach().numpy()
                batch_predicted_status = (torch.argmax(batch_logit, dim=1) == target).float().detach().numpy()
                batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)
                member = int(1 - loader_idx)   

                model_score         = batch_score               if loader_idx == 0 and data_idx == 0     else np.concatenate((model_score, batch_score), axis=0)
                model_loss          = batch_loss                if loader_idx == 0 and data_idx == 0     else np.concatenate((model_loss, batch_loss), axis=0)
                orginal_labels      = batch_orginal_label       if loader_idx == 0 and data_idx == 0     else np.concatenate((orginal_labels, batch_orginal_label), axis=0)
                predicted_labels    = batch_predict_label       if loader_idx == 0 and data_idx == 0     else np.concatenate((predicted_labels, batch_predict_label), axis=0)
                predicted_status    = batch_predicted_status    if loader_idx == 0 and data_idx == 0     else np.concatenate((predicted_status, batch_predicted_status), axis=0)
                infer_time          = [batch_time]              if loader_idx == 0 and data_idx == 0     else np.concatenate((infer_time, [batch_time]), axis=0)
                branch_idx          = [exit_idx]                if loader_idx == 0 and data_idx == 0     else np.concatenate((branch_idx, [exit_idx]), axis=0)
                #branch_norm         = [exit_idx/num_exits]      if loader_idx == 0 and data_idx == 0     else np.concatenate((branch_norm, [exit_idx/num_exits]), axis=0)
                member_status       = [member]                  if loader_idx == 0 and data_idx == 0     else np.concatenate((member_status, [member]), axis=0)
                early_status        = [is_early]                if loader_idx == 0 and data_idx == 0     else np.concatenate((early_status, [is_early]), axis=0)
                #model_features      = features                  if loader_idx == 0 and data_idx == 0     else np.concatenate((model_features, features), axis=0)
                #model_gradient      = batch_gradient            if loader_idx == 0 and data_idx == 0     else np.concatenate((model_gradient, batch_gradient), axis=0)
                # print(infer_time)
                prec1, _ = DATA.accuracy(batch_logit, target, topk=(1,5))
                top1.update(prec1[0], batch_logit.size(0))
                print(data_idx)
                if data_idx == 20:
                    break
            top1_acc = top1.avg.data.numpy()[()]
            print(f'------------member_status: {1-loader_idx}, top1 acc: {top1_acc}---------')
           
        data = {'data_seed':data_seed,
                'model_scores':model_score, 
                'model_loss':model_loss,
                'orginal_labels':orginal_labels,
                'predicted_labels':predicted_labels,
                'predicted_status':predicted_status,   
                'infer_time':infer_time,         
                #'infer_time_norm':normalization(infer_time),
                'exit_idx':branch_idx,
                #'branch_norm':branch_norm,
                'member_status':member_status,
                'early_status':early_status,
                'num_exits':num_exits,
                'nb_classes':dataset.num_classes,
                #'model_features':model_features,
                #'model_gradient':model_gradient
                }
        dataset_type = 'testset' if args.mode=='target' else 'trainset'
        if 'cnn' in model_name:
            model_name = model_name
        elif 'sdn' in model_name:
            model_name = model_name + '/' + str(model_idx) + '/' + sdn_training_type

        pickle.dump(data, open(models_path + f'/{model_name}/{dataset_type}_4_process_1_cpu_device.pkl', 'wb'), protocol=4)
      
        # np.save(models_path + f'/{model_name}/{dataset_type}', data)


def eval_acc(args, models_path, device='cpu'):
    #sdn_training_type = 'ic_only' # IC-only training
    sdn_training_type = args.training_type # SDN training
    device ='cpu'
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

    thds_path = 'networks/'+str(args.seed)+'/target/' + model_name + '_sdn/threshold/' + sdn_training_type + '/thds.txt'
    with open(thds_path, "r") as f:  # 打开文件
        confidence_thresholds = f.read()  # 读取文件
        confidence_thresholds = str(confidence_thresholds).split()
        confidence_thresholds = [float(x) for x in confidence_thresholds]
    print(confidence_thresholds)

    for model_idx, model_name in  enumerate(cnns_sdns):
        if model_idx in [4,5]:
            continue
        # else:
        #     pass
        print(f'------------------model: {model_name}, num_exits:{model_idx+1}-------------------')
        if 'cnn' in model_name:
            cnn_model, cnn_params = arcs.load_model(models_path, model_name, epoch=-1)
            MODEL = cnn_model.to(device)
            #print(cnn_params)
            
            dataset = af.get_dataset(cnn_params['task'], batch_size=1)
            if args.mode == 'target':
                print('load target_dataset ... ')
                train_loader = dataset.target_train_loader
                test_loader = dataset.target_test_loader
            elif args.mode == 'shadow':
                print('load shadow_dataset ...')
                train_loader = dataset.shadow_train_loader
                test_loader = dataset.shadow_test_loader

            feature_layers = model_layer_index[model_name[model_name.find('_')+1:]]
            gradient_layers = gradient_layer_index[model_name[model_name.find('_')+1:]]
        elif 'sdn' in model_name:
            sdn_model, sdn_params = arcs.load_model(models_path, model_name, epoch=-1, idx=model_idx, sdn_training_type=sdn_training_type)
            MODEL = sdn_model.to(device)    
            # to test early-exits with the SDN
            dataset = af.get_dataset(sdn_params['task'], batch_size=1)
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

            feature_layers = model_layer_index[model_name[model_name.find('_')+1:]+f'_{model_idx}']
            gradient_layers = gradient_layer_index[model_name[model_name.find('_')+1:]+f'_{model_idx}']
            #confidence_thresholds = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] # search for the confidence threshold for early exits
            MODEL.forward = MODEL.early_exit_seq
            MODEL.confidence_threshold = confidence_thresholds[model_idx-1] # confidence_thresholds

            print(f'--------------theshold:{MODEL.confidence_threshold}----------------')
        else:
            continue
        #print(MODEL)
        #feature_extractor = FeatureExtractor(MODEL, feature_layers)

 
      
        #with torch.no_grad():
        MODEL.eval()
    
        for loader_idx, data_loader in enumerate([train_loader, test_loader]):
            # GPUtil.showUtilization()
            
            top1 = DATA.AverageMeter()
            for data_idx, (data, target) in enumerate(data_loader):
     
                data, target = data.to(device), target.to(device)
                if 'cnn' in model_name:
                    #for rep in range(repetitions):
                    #starter.record()
                    batch_logit = MODEL(data)
                    #ender.record()
                    #torch.cuda.synchronize()
                    #curr_time = starter.elapsed_time(ender)
                    
                    exit_idx = 0
                    is_early = False
                    num_exits = 1
                    #features = list(feature_extractor(data).values())[exit_idx].view(1, -1).detach().cpu().numpy()
                    
                    #model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+1)]]

                    # print(features.shape)
                    # print(model_feature_len_)
                    # exit()
                    # if features.shape[1] < model_feature_len_:
                    #     feature_len = int(model_feature_len_/features.shape[1])
                    #     feature_list = [features for _ in range(feature_len)]
                    #     features = np.concatenate(feature_list, axis=1)
                elif 'sdn' in model_name:
                    
                    
                    batch_logit, exit_idx, is_early = MODEL(data)
                        
                    
                    num_exits = model_idx+1
                    #features = list(feature_extractor(data).values())[exit_idx].view(1, -1).detach().cpu().numpy()

                    #model_feature_len_ = model_feature_len[model_name[:model_name.find('_', model_name.find('_')+1)]]
                    # if features.shape[1] < model_feature_len_:
                    #     feature_len = int(model_feature_len_/features.shape[1])
                    #     feature_list = [features for _ in range(feature_len)]
                    #     features = np.concatenate(feature_list, axis=1)
                    # if exit_idx == 1:
             
                
                prec1, _ = DATA.accuracy(batch_logit, target, topk=(1,5))
                top1.update(prec1[0], batch_logit.size(0))
                
                if data_idx == 9999:
                    break
            top1_acc = top1.avg.data.cpu().numpy()[()]
            print(f'------------member_status: {1-loader_idx}, top1 acc: {top1_acc}---------')
     
