import torch
import math

import torch.nn as nn
import numpy as np

from deeplearning import aux_funcs as af
from deeplearning import model_funcs as mf

class FullyBlockWOutput(nn.Module):
    def __init__(self, linear_params, output_params):
        super(FullyBlockWOutput, self).__init__()
        layer_id = linear_params[0]
        input_size = linear_params[1]
        linear_channels = linear_params[2]
     
        
        add_output = output_params[0]
        num_classes = output_params[1]
        #input_size = output_params[2]
        self.output_id = output_params[2]

        # self.depth = 1

        layers = []
        if layer_id == 0:
            layers.append(nn.Linear(input_size, linear_channels))
            layers.append(nn.BatchNorm1d(linear_channels))
            layers.append(nn.ReLU())

        else:
            layers.append(nn.Linear(linear_channels, linear_channels))
            layers.append(nn.BatchNorm1d(linear_channels))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        if add_output:
            self.output = nn.Sequential(
                nn.Linear(linear_channels, 512),
                nn.Dropout(),
                nn.Linear(512, num_classes))
            self.no_output = False

        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True
        
    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        fwd = self.layers(x)
        return fwd, 0, None

def EndBlockWOutput(end_layers, linear_params):
    in_channels = linear_params[0]
    num_classes = linear_params[1]
    end_layers.append(nn.Linear(in_channels, in_channels))         
    end_layers.append(nn.Dropout())         
    end_layers.append(nn.Linear(in_channels, 512))         
    end_layers.append(nn.Dropout())         
    end_layers.append(nn.Linear(512, num_classes))
    return end_layers
    
class FCN_SDN_1(nn.Module):
    def __init__(self, params):
        super(FCN_SDN_1, self).__init__()
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.init_weights = params['init_weights']
        self.add_output = params['add_ic']
        self.in_channels = params['in_channels'][0]
        self.augment_training = params['augment_training']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1

        # self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        output_id = 0

        for layer_id, add_output in enumerate(self.add_output):
            linear_params =  (layer_id, self.input_size, self.in_channels)
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FullyBlockWOutput(linear_params, output_params))
            output_id += add_output
        
        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        fwd = x
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit_seq(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)      
                #print(softmax)         
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        # outputs.append(output)

        is_early = False
        return output, output_id, is_early

    # takes a single input
    def early_exit_only_score(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output#, output_id, is_early
                
                output_id += is_output
        output = self.end_layers(fwd)
        is_early = False
        return output #output_id, is_early

    def early_exit_only_index(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)                
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    #print(output)
                    output = output * 0 + (1-self.confidence_threshold)
                    output = output[:, 0:self.num_output]
                    output[:,output_id] = 1
                    
                    return output#, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        output = output * 0 + (1-self.confidence_threshold)
        output = output[:, 0:self.num_output]
        output[:,output_id] = 1
        is_early = False
        return output#, output_id, is_early

class FCN_SDN_2(nn.Module):
    def __init__(self, params):
        super(FCN_SDN_2, self).__init__()
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.init_weights = params['init_weights']
        self.add_output = params['add_ic']
        self.in_channels = params['in_channels'][1]
        self.augment_training = params['augment_training']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1

        # self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        output_id = 0
        for layer_id, add_output in enumerate(self.add_output):
            linear_params =  (layer_id, self.input_size, self.in_channels)
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FullyBlockWOutput(linear_params, output_params))
            output_id += add_output
        
        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        fwd = x
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit_seq(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)      
                #print(softmax)         
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        # outputs.append(output)

        is_early = False
        return output, output_id, is_early

    # takes a single input
    def early_exit_only_score(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output#, output_id, is_early
                
                output_id += is_output
        output = self.end_layers(fwd)
        # outputs.append(output)

        # softmax = nn.functional.softmax(output[0], dim=0)
        # confidence = torch.max(softmax)
        # confidences.append(confidence)
        # max_confidence_output = np.argmax(confidences)
        # output_id += 1
        is_early = False
        return output #output_id, is_early

    def early_exit_only_index(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)                
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    #print(output)
                    output = output * 0 + (1-self.confidence_threshold)
                    output = output[:, 0:self.num_output]
                    output[:,output_id] = 1
                    
                    return output#, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        output = output * 0 + (1-self.confidence_threshold)
        output = output[:, 0:self.num_output]
        output[:,output_id] = 1
        is_early = False
        return output#, output_id, is_early

class FCN_SDN_3(nn.Module):
    def __init__(self, params):
        super(FCN_SDN_3, self).__init__()
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.init_weights = params['init_weights']
        self.add_output = params['add_ic']
        self.in_channels = params['in_channels'][2]
        self.augment_training = params['augment_training']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1

        # self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        output_id = 0

        for layer_id, add_output in enumerate(self.add_output):
            linear_params =  (layer_id, self.input_size, self.in_channels)
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FullyBlockWOutput(linear_params, output_params))
            output_id += add_output
        
        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        fwd = x
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit_seq(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)      
                #print(softmax)         
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        # outputs.append(output)

        is_early = False
        return output, output_id, is_early

    # takes a single input
    def early_exit_only_score(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output#, output_id, is_early
                
                output_id += is_output
        output = self.end_layers(fwd)
        is_early = False
        return output #output_id, is_early

    def early_exit_only_index(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)                
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    #print(output)
                    output = output * 0 + (1-self.confidence_threshold)
                    output = output[:, 0:self.num_output]
                    output[:,output_id] = 1
                    
                    return output#, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        output = output * 0 + (1-self.confidence_threshold)
        output = output[:, 0:self.num_output]
        output[:,output_id] = 1
        is_early = False
        return output#, output_id, is_early

class FCN_SDN_4(nn.Module):
    def __init__(self, params):
        super(FCN_SDN_4, self).__init__()
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.add_output = params['add_ic']
        self.in_channels = params['in_channels'][3]
        self.augment_training = params['augment_training']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1

        self.layers = nn.ModuleList()
        output_id = 0

        for layer_id, add_output in enumerate(self.add_output):
            linear_params =  (layer_id, self.input_size, self.in_channels)
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FullyBlockWOutput(linear_params, output_params))
            output_id += add_output
        
        
        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        fwd = x
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit_seq(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)      
                #print(softmax)         
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        # outputs.append(output)

        is_early = False
        return output, output_id, is_early

    # takes a single input
    def early_exit_only_score(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output#, output_id, is_early
                
                output_id += is_output
        output = self.end_layers(fwd)
        # outputs.append(output)

        # softmax = nn.functional.softmax(output[0], dim=0)
        # confidence = torch.max(softmax)
        # confidences.append(confidence)
        # max_confidence_output = np.argmax(confidences)
        # output_id += 1
        is_early = False
        return output #output_id, is_early

    def early_exit_only_index(self, x):
        confidences = []
        outputs = []

        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)                
                confidence = torch.max(softmax)
                confidences.append(confidence)
                if confidence >= self.confidence_threshold:
                    is_early = True
                    #print(output)
                    output = output * 0 + (1-self.confidence_threshold)
                    output = output[:, 0:self.num_output]
                    output[:,output_id] = 1
                    
                    return output#, output_id, is_early
                output_id += is_output

        output = self.end_layers(fwd)
        output = output * 0 + (1-self.confidence_threshold)
        output = output[:, 0:self.num_output]
        output[:,output_id] = 1
        is_early = False
        return output#, output_id, is_early

