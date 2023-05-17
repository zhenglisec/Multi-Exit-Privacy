from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
model_feature_len = {
    'cifar10_vgg16bn': 8192,
    'cifar10_resnet56': 1024,
    'cifar10_mobilenet': 8192,
    'cifar10_wideresnet32': 4096,
    'cifar100_vgg16bn': 8192,
    'cifar100_resnet56': 1024,
    'cifar100_mobilenet': 4096,
    'cifar100_wideresnet32': 4096,
    'tinyimagenet_vgg16bn': 8192,
    'tinyimagenet_resnet56': 1024,
    'tinyimagenet_mobilenet': 16384,
    'tinyimagenet_wideresnet32': 4096,
}   

model_layer_index = {
    'vgg16bn_cnn':   ['layers.12.layers'],
    'vgg16bn_sdn_1': ['layers.1.output.avg_pool', 'layers.12.layers'],
    'vgg16bn_sdn_2': ['layers.1.output.avg_pool', 'layers.3.output.avg_pool', 'layers.12.layers'],
    'vgg16bn_sdn_3': ['layers.1.output.avg_pool', 'layers.3.output.avg_pool',  'layers.5.output.avg_pool', 'layers.12.layers'],
    'vgg16bn_sdn_4': ['layers.1.output.avg_pool', 'layers.3.output.avg_pool',  'layers.5.output.avg_pool', 'layers.6.output.avg_pool', 'layers.12.layers'],
    'vgg16bn_sdn_5': ['layers.1.output.avg_pool', 'layers.3.output.avg_pool',  'layers.5.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'layers.12.layers'],
    'vgg16bn_sdn_6': ['layers.1.output.avg_pool', 'layers.3.output.avg_pool',  'layers.5.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'layers.9.layers', 'layers.12.layers'],

    'resnet56_cnn':   ['end_layers.0'],
    'resnet56_sdn_1': ['layers.3.output.avg_pool', 'end_layers.0'],
    'resnet56_sdn_2': ['layers.3.output.avg_pool', 'layers.7.output.avg_pool', 'end_layers.0'],
    'resnet56_sdn_3': ['layers.3.output.avg_pool', 'layers.7.output.avg_pool', 'layers.11.output.avg_pool', 'end_layers.0'],
    'resnet56_sdn_4': ['layers.3.output.avg_pool', 'layers.7.output.avg_pool', 'layers.11.output.avg_pool', 'layers.15.output.avg_pool', 'end_layers.0'],
    'resnet56_sdn_5': ['layers.3.output.avg_pool', 'layers.7.output.avg_pool', 'layers.11.output.avg_pool', 'layers.15.output.avg_pool', 'layers.19.output.avg_pool', 'end_layers.0'],
    'resnet56_sdn_6': ['layers.3.output.avg_pool', 'layers.7.output.avg_pool', 'layers.11.output.avg_pool', 'layers.15.output.avg_pool', 'layers.19.output.avg_pool', 'layers.23.output.avg_pool', 'end_layers.0'],
    
    'mobilenet_cnn':   ['end_layers.0'],
    'mobilenet_sdn_1': ['layers.2.output.avg_pool', 'end_layers.0'],
    'mobilenet_sdn_2': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'end_layers.0'],
    'mobilenet_sdn_3': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'end_layers.0'],
    'mobilenet_sdn_4': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'end_layers.0'],
    'mobilenet_sdn_5': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'layers.11.layers', 'end_layers.0'],

    'wideresnet32_4_cnn':   ['end_layers.2'],
    'wideresnet32_4_sdn_1': ['layers.2.output.avg_pool', 'end_layers.2'],
    'wideresnet32_4_sdn_2': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'end_layers.2'],
    'wideresnet32_4_sdn_3': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'end_layers.2'],
    'wideresnet32_4_sdn_4': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'end_layers.2'],
    'wideresnet32_4_sdn_5': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'layers.10.output.avg_pool', 'end_layers.2'],
    'wideresnet32_4_sdn_6': ['layers.2.output.avg_pool', 'layers.4.output.avg_pool', 'layers.6.output.avg_pool', 'layers.8.output.avg_pool', 'layers.10.output.avg_pool', 'layers.12.output.avg_pool', 'end_layers.2'],

}

def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True):
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
        elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2:
            labels = np.squeeze(labels)
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2:
            # pass
            labels = np.squeeze(labels)
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 1:
            if return_one_hot:
                if nb_classes == 2:
                    labels = np.expand_dims(labels, axis=1)
                    labels = to_categorical(labels, nb_classes)
                else:
                    labels = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels

def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None):
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=np.int32)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)

    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical

def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]


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

class MLP_BLACKBOX__(nn.Module):
    def __init__(self, dim_in):
        super(MLP_BLACKBOX, self).__init__()
        self.dim_in = dim_in
        # self.bn1 = nn.BatchNorm1d(self.dim_in)
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        # x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class posteriorAttackModel(nn.Module):
            def __init__(self, dim_in, dim_out):
                super(posteriorAttackModel, self).__init__()
                self.dim_in = dim_in
                # self.bn1 = nn.BatchNorm1d(self.dim_in)
                self.fc1 = nn.Linear(self.dim_in, 64)
                self.fc2 = nn.Linear(64, 32)
                # self.fc3 = nn.Linear(64, 32)
                # self.fc3 = nn.Linear(10, 10)
                self.fc4 = nn.Linear(32, dim_out)

            def forward(self, x):
                #x = x.view(-1, self.dim_in)
                # x = self.bn1(x)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                # x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x 
class MLP_BLACKBOX(nn.Module):
    def __init__(self, score_len, num_exits=-1):
        super(MLP_BLACKBOX, self).__init__()
        self.output_component = nn.Sequential(
            # nn.Dropout(p=0.2),
            # self.dropout,
            nn.Linear(score_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.exit_component = nn.Sequential(
            # nn.Dropout(p=0.2),
            # self.dropout,
            nn.Linear(1 if num_exits == -1 else num_exits, 128),
            nn.ReLU(),
            #nn.Linear(128, 128),
        )
        self.encoder_component = nn.Sequential(
            #nn.Dropout(p=0.2),
            # self.dropout,
            nn.Linear(128 if num_exits == -1 else 128 * 2, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            # self.dropout,
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

       
    def forward(self, output, exit_idx=None):
    
        output_component_result = self.output_component(output)
        # gradient_component_result = self.gradient_component(gradient)

        if exit_idx != None:
            exit_component_result = self.exit_component(exit_idx)
            final_input = torch.cat((output_component_result, exit_component_result), 1)

        else:
            final_input = output_component_result

        final_result = self.encoder_component(final_input)
       
        return final_result
class WhiteBoxAttackModel(nn.Module):
    def __init__(self, class_num, embedding_dim, num_exits=-1):
        super(WhiteBoxAttackModel, self).__init__()

        # self.dropout = nn.Dropout(p=0.2)
        self.output_component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )

        self.loss_component = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )

        self.gradient_component = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=0),
            nn.AdaptiveAvgPool2d(5),  # [batch_size, channel, 5, 5]
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(125, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )

        self.label_component = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )
        self.embedding_component = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )

        if num_exits != -1:
            self.exit_component = nn.Sequential(
                nn.Linear(num_exits, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 64),
            )
        self.encoder_component = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 5 if num_exits == -1 else 64 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

        

    def forward(self, original_label, output, embedding, loss, gradient, exit_idx=None):
        label_component_result = self.label_component(original_label)
        
        output_component_result = self.output_component(output)
        gradient_component_result = self.gradient_component(gradient)
        embedding_component_result = self.embedding_component(embedding)

        loss_component_result = self.loss_component(loss)

        if exit_idx != None:
            exit_component_result = self.exit_component(exit_idx)
            final_input = torch.cat((label_component_result, output_component_result, gradient_component_result,
                                  embedding_component_result, loss_component_result, exit_component_result), 1)
        else:
            final_input = torch.cat((label_component_result, output_component_result, gradient_component_result,
                                  embedding_component_result, loss_component_result), 1)
        # print(final_input.shape)

        final_result = self.encoder_component(final_input)
        return final_result
        
class WhiteBoxAttackModel_ok_feature_dataset(nn.Module):
    def __init__(self, class_num, embedding_dim, num_exits=-1):
        super(WhiteBoxAttackModel_ok_feature_dataset, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        self.output_component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 128),
        )

        self.loss_component = nn.Sequential(
            nn.Linear(1, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 128),
        )

        self.gradient_component = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=0),
            nn.BatchNorm2d(5),
            nn.AdaptiveAvgPool2d(5),  # [batch_size, channel, 5, 5]
            nn.Flatten(),
            # nn.Dropout(p=0.5),
            nn.Linear(125, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 128),
        )

        self.label_component = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 128),
        )
        self.embedding_component = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 128),
        )
        if num_exits != -1:
            self.exit_component = nn.Sequential(
                nn.Linear(num_exits, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                # nn.Dropout(p=0.5),
                # nn.Linear(128, 64),
            )
            
        self.encoder_component = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(128 * 5 if num_exits == -1 else 128 * 6, 128),
            # nn.ReLU(),
            nn.Dropout(p=0.5),
            # self.dropout,
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

        

    def forward(self, original_label,  output, embedding, loss, gradient, exit_idx=None):
        label_component_result = self.label_component(original_label)
        
        output_component_result = self.output_component(output)
        gradient_component_result = self.gradient_component(gradient)
        embedding_component_result = self.embedding_component(embedding)

        loss_component_result = self.loss_component(loss)

        if exit_idx == None:
            final_input = torch.cat((label_component_result, output_component_result, loss_component_result,
                                  embedding_component_result, gradient_component_result), 1)
            
        else:
            exit_component_result = self.exit_component(exit_idx)
            final_input = torch.cat((label_component_result, output_component_result, loss_component_result,
                                  embedding_component_result, gradient_component_result, exit_component_result), 1)
        # print(final_input.shape)
        final_result = self.encoder_component(final_input)
        return final_result

class WhiteBoxAttackModel_new(nn.Module):
    def __init__(self, class_num, embedding_dim, num_exits=0):
        super(WhiteBoxAttackModel_new, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        self.output_component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.ReLU(),
           #nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )

        self.loss_component = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )

        # self.gradient_component = nn.Sequential(
        #     nn.Conv2d(1, 5, kernel_size=3, padding=0),
        #     nn.AdaptiveAvgPool2d(5),  # [batch_size, channel, 5, 5]
        #     nn.Flatten(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(125, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128, 64),
        # )

        self.label_component = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )
        self.embedding_component = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )
        self.exit_component = nn.Sequential(
            nn.Linear(num_exits, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(128, 64),
        )
        self.encoder_component = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(64 * 4, 256),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(64, 2),
        )

        self.combine_component = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(64+num_exits, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

    def forward_old(self, predicted_status, output, embedding, loss, exit_idx=None):
        label_component_result = self.label_component(predicted_status)
        
        output_component_result = self.output_component(output)
        # gradient_component_result = self.gradient_component(gradient)
        embedding_component_result = self.embedding_component(embedding)

        loss_component_result = self.loss_component(loss)

        if exit_idx == None:
            final_input = torch.cat((label_component_result, output_component_result,
                                  embedding_component_result, loss_component_result), 1)
        else:
            exit_component_result = self.exit_component(exit_idx)
            final_input = torch.cat((label_component_result, output_component_result,
                                  embedding_component_result, loss_component_result), 1)

        final_result = self.encoder_component(final_input)
        return final_result

    def forward(self, predicted_status, output, embedding, loss, exit_idx=None):
        label_component_result = self.label_component(predicted_status)
        
        output_component_result = self.output_component(output)
        # gradient_component_result = self.gradient_component(gradient)
        embedding_component_result = self.embedding_component(embedding)

        loss_component_result = self.loss_component(loss)

        out = torch.cat((label_component_result, output_component_result,
                                  embedding_component_result, loss_component_result), 1)

        out = self.encoder_component(out)  
        if exit_idx == None:
            out = self.combine_component(out)
            return out
        else:
            #print(exit_idx)
            # exit_component_result = self.exit_component(exit_idx)
            out = torch.cat((out, exit_idx), 1)
            out = self.combine_component(out)
            return out

###########
# if mia == 'black_box_top3':
#                 attack_model = MLP_BLACKBOX(3)
#             elif mia == 'black_box_top3+exit_idx' and 'sdn' in model_name:
#                 attack_model = MLP_BLACKBOX(3,  num_exits)
#             elif mia == 'black_box_top3+predict_idx+1_process_8_gpu_core' and 'sdn' in model_name:
#                 attack_model = MLP_BLACKBOX(3,  num_exits)
#             elif mia == 'black_box_top3+predict_idx+8_process_8_gpu_core' and 'sdn' in model_name:
#                 attack_model = MLP_BLACKBOX(3,  num_exits)
#             elif mia == 'black_box_top3+predict_idx+4_process_per_gpu_core' and 'sdn' in model_name:
#                 attack_model = MLP_BLACKBOX(3,  num_exits)
#             elif mia == 'black_box_top3+predict_idx+12_process_per_cpu_device' and 'sdn' in model_name:
#                 attack_model = MLP_BLACKBOX(3,  num_exits)
#########
def train_mia_attack_model(mia, model, attack_train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    correct = 0
    # for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, infer_time_norm, exit_idx, branch_norm, member_status, early_status, model_features, train_predict_idx) in enumerate(attack_train_loader):
    for batch_idx, (data_seed, model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, exit_idx, member_status, early_status, model_features, model_gradient) in enumerate(attack_train_loader):
        
        if mia == 'white_box':
            input_1 = predicted_status.to(device)
            input_2, _ = torch.sort(model_scores, dim=1, descending=True)
            input_2 = input_2.to(device)
            input_3 = model_features.to(device)
            model_loss = model_loss.view(-1, 1)
            input_4 = model_loss.to(device)
            input_5 = model_gradient.to(device)
        elif mia == 'white_box+exit_idx':
            input_1 = predicted_status.to(device)
            input_2, _ = torch.sort(model_scores, dim=1, descending=True)
            input_2 = input_2.to(device)
            input_3 = model_features.to(device)
            model_loss = model_loss.view(-1, 1)
            input_4 = model_loss.to(device)
            input_5 = model_gradient.to(device)
            input_6 = exit_idx.to(device)



        elif mia == 'black_box_top3':
            input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
            input_1 = input_1.to(device)
        elif mia == 'black_box_top3+exit_idx':
            input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
            #input = torch.cat((input_1, input_2), 1)
        elif 'black_box_top3+predict_idx' in mia:
            input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
            #input = torch.cat((input_1, input_2), 1)
        # elif mia == 'black_box_top3+infer_time_5':
        #     pass
        # elif mia == 'black_box_top3+infer_time_10':
        #     pass

        elif mia == 'black_box_sorted':
            input_1, _ = torch.sort(model_scores, dim=1, descending=True)
            input_1 = input_1.to(device)
        elif mia == 'black_box_sorted+exit_idx':
            input_1, _ = torch.sort(model_scores, dim=1, descending=True)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
        elif 'black_box_sorted+predict_idx' in mia:
            input_1, _ = torch.sort(model_scores, dim=1, descending=True)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
        # elif mia == 'black_box_sorted+infer_time_5':
        #     pass
        # elif mia == 'black_box_sorted+infer_time_10':
        #     pass

        if mia == 'white_box':
            output = model(input_1, input_2, input_3, input_4, input_5)
        elif mia == 'white_box+exit_idx':
            output = model(input_1, input_2, input_3, input_4, input_5, input_6)
        elif 'black_box' in mia and 'idx' not in mia: 
            output = model(input_1)
        elif 'black_box' in mia and 'idx' in mia:
            output = model(input_1, input_2)
        #data, target = data.to(args.device), target.to(args.device)
        
        member_status = member_status.to(device)
        loss = loss_fn(output, member_status)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % 49 == 0:
        #     print('AttackModel Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( 
        #         epoch,
        #         batch_idx * len(input_1),
        #         len(attack_train_loader.dataset),
        #         100. * batch_idx / len(attack_train_loader),
        #         loss.item()))
                
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(member_status.view_as(pred)).sum().item()
    train_loss /= len(attack_train_loader.dataset)
    accuracy = 100. * correct / len(attack_train_loader.dataset)
    return train_loss, accuracy/100.

def test_mia_attack_model(mia, model, attack_test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    auc_ground_truth = None
    auc_pred = None
    with torch.no_grad():
        # for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, infer_time_norm, exit_idx, branch_norm, member_status, early_status, model_features, test_predict_idx) in enumerate(attack_test_loader):
        for batch_idx, (data_seed, model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, exit_idx, member_status, early_status, model_features, model_gradient) in enumerate(attack_test_loader):
            if mia == 'white_box':
                input_1 = predicted_status.to(device)
                input_2, _ = torch.sort(model_scores, dim=1, descending=True)
                input_2 = input_2.to(device)
                input_3 = model_features.to(device)
                model_loss = model_loss.view(-1, 1)
                input_4 = model_loss.to(device)
                input_5 = model_gradient.to(device)
            elif mia == 'white_box+exit_idx':
                input_1 = predicted_status.to(device)
                input_2, _ = torch.sort(model_scores, dim=1, descending=True)
                input_2 = input_2.to(device)
                input_3 = model_features.to(device)
                model_loss = model_loss.view(-1, 1)
                input_4 = model_loss.to(device)
                input_5 = model_gradient.to(device)
                input_6 = exit_idx.to(device)



            elif mia == 'black_box_top3':
                input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
                input_1 = input_1.to(device)
            elif mia == 'black_box_top3+exit_idx':
                input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
                #input = torch.cat((input_1, input_2), 1)
            elif 'black_box_top3+predict_idx' in mia:
                input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
                #input = torch.cat((input_1, input_2), 1)
            # elif mia == 'black_box_top3+infer_time_5':
            #     pass
            # elif mia == 'black_box_top3+infer_time_10':
            #     pass

            elif mia == 'black_box_sorted':
                input_1, _ = torch.sort(model_scores, dim=1, descending=True)
                input_1 = input_1.to(device)
            elif mia == 'black_box_sorted+exit_idx':
                input_1, _ = torch.sort(model_scores, dim=1, descending=True)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
            elif 'black_box_sorted+predict_idx' in mia:
                input_1, _ = torch.sort(model_scores, dim=1, descending=True)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
            # elif mia == 'black_box_sorted+infer_time_5':
            #     pass
            # elif mia == 'black_box_sorted+infer_time_10':
            #     pass
            
            if mia == 'white_box':
                output = model(input_1, input_2, input_3, input_4, input_5)
            elif mia == 'white_box+exit_idx':
                output = model(input_1, input_2, input_3, input_4, input_5, input_6)
            elif 'black_box' in mia and 'idx' not in mia: 
                output = model(input_1)
            elif 'black_box' in mia and 'idx' in mia:
                output = model(input_1, input_2)
            #input_2 = input_2.to(args.device)
            member_status = member_status.to(device)
            
            #output = model(input)
            test_loss += loss_fn(output, member_status).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(member_status.view_as(pred)).sum().item()

            # auc_ground_truth += member_status.cpu().numpy()
            # auc_pred += pred.cpu().numpy()

            auc_ground_truth = member_status.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_ground_truth, member_status.cpu().numpy()), axis=0)
            auc_pred = pred.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_pred, pred.cpu().numpy()), axis=0)

    test_loss /= len(attack_test_loader.dataset)

    accuracy = 100. * correct / len(attack_test_loader.dataset)
    # print(auc_ground_truth)
    # print(auc_pred)
    fpr, tpr, thresholds = metrics.roc_curve(auc_ground_truth, auc_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # print('\nTargetModel Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.0f}\n'.format(
    #     test_loss, correct, len(attack_test_loader.dataset), accuracy, auc))
    return test_loss, accuracy/100., auc

def train_black_mia_attack_model(mia, model, attack_train_loader, optimizer, loss_fn, device, nb_classes=None):  
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data_seed, model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, exit_idx, member_status, early_status, model_features) in enumerate(attack_train_loader):
    # for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, exit_idx, member_status, early_status) in enumerate(attack_train_loader):
        if mia == 'black_box_top3':
            input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
            input_1 = input_1.to(device)
            # input_2 = predicted_status.to(device)
        elif mia == 'black_box_top3+exit_idx':
            input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
        elif 'black_box_top3+predict_idx' in mia:
            input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
            #input = torch.cat((input_1, input_2), 1)
        # elif mia == 'black_box_top3+infer_time_5':
        #     pass
        # elif mia == 'black_box_top3+infer_time_10':
        #     pass

        elif mia == 'black_box_sorted':
            input_1, _ = torch.sort(model_scores, dim=1, descending=True)
            input_1 = input_1.to(device)
        elif mia == 'black_box_sorted+exit_idx':
            input_1, _ = torch.sort(model_scores, dim=1, descending=True)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
        elif 'black_box_sorted+predict_idx' in mia:
            input_1, _ = torch.sort(model_scores, dim=1, descending=True)
            input_1 = input_1.to(device)
            input_2 = exit_idx.to(device)
        # elif mia == 'black_box_sorted+infer_time_5':
        #     pass
        # elif mia == 'black_box_sorted+infer_time_10':
        #     pass

        if nb_classes is not None:
            input_1 = input_1[:, 0:nb_classes]
        
        if 'black_box' in mia and 'idx' not in mia: 
            output = model(input_1)
        elif 'black_box' in mia and 'idx' in mia:
            #input_2 = input_2.view(-1, 1)
            output = model(input_1, input_2)
        #data, target = data.to(args.device), target.to(args.device)
        
        member_status = member_status.to(device)
        loss = loss_fn(output, member_status)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % 49 == 0:
        #     print('AttackModel Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( 
        #         epoch,
        #         batch_idx * len(input_1),
        #         len(attack_train_loader.dataset),
        #         100. * batch_idx / len(attack_train_loader),
        #         loss.item()))
                
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(member_status.view_as(pred)).sum().item()
    train_loss /= len(attack_train_loader.dataset)
    accuracy = 100. * correct / len(attack_train_loader.dataset)
    return train_loss, accuracy/100.

def test_black_mia_attack_model(mia, model, attack_test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    correct = 0
    auc_ground_truth = None
    auc_pred = None
    with torch.no_grad():
        for batch_idx, (data_seed, model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, exit_idx, member_status, early_status, model_features) in enumerate(attack_test_loader):
        # for batch_idx, (model_scores, model_loss, orginal_labels, predicted_labels, predicted_status, infer_time, exit_idx, member_status, early_status) in enumerate(attack_test_loader):

            if mia == 'black_box_top3':
                input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
                input_1 = input_1.to(device)
                # input_2 = predicted_status.to(device)
            elif mia == 'black_box_top3+exit_idx':
                input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
                input_1 = input_1.to(device)
                # input_2 = predicted_status.to(device)
                input_2 = exit_idx.to(device)
                #input = torch.cat((input_1, input_2), 1)
            elif 'black_box_top3+predict_idx' in mia:
                input_1, _ = torch.topk(model_scores, k=3, dim=1, largest=True, sorted=False)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
                #input = torch.cat((input_1, input_2), 1)
            # elif mia == 'black_box_top3+infer_time_5':
            #     pass
            # elif mia == 'black_box_top3+infer_time_10':
            #     pass

            elif mia == 'black_box_sorted':
                input_1, _ = torch.sort(model_scores, dim=1, descending=True)
                input_1 = input_1.to(device)
            elif mia == 'black_box_sorted+exit_idx':
                input_1, _ = torch.sort(model_scores, dim=1, descending=True)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
            elif 'black_box_sorted+predict_idx' in mia:
                input_1, _ = torch.sort(model_scores, dim=1, descending=True)
                input_1 = input_1.to(device)
                input_2 = exit_idx.to(device)
            # elif mia == 'black_box_sorted+infer_time_5':
            #     pass
            # elif mia == 'black_box_sorted+infer_time_10':
            #     pass
            
           
            if 'black_box' in mia and 'idx' not in mia: 
                output = model(input_1)
            elif 'black_box' in mia and 'idx' in mia:
                #input_2 = input_2.view(-1, 1)
                output = model(input_1, input_2)
            #input_2 = input_2.to(args.device)
            member_status = member_status.to(device)
            
            #output = model(input)
            test_loss += loss_fn(output, member_status).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(member_status.view_as(pred)).sum().item()

            # auc_ground_truth += member_status.cpu().numpy()
            # auc_pred += pred.cpu().numpy()

            auc_ground_truth = member_status.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_ground_truth, member_status.cpu().numpy()), axis=0)
            auc_pred = pred.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_pred, pred.cpu().numpy()), axis=0)

    test_loss /= len(attack_test_loader.dataset)

    accuracy = 100. * correct / len(attack_test_loader.dataset)
    # print(auc_ground_truth)
    # print(auc_pred)
    fpr, tpr, thresholds = metrics.roc_curve(auc_ground_truth, auc_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # print('\nTargetModel Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.0f}\n'.format(
    #     test_loss, correct, len(attack_test_loader.dataset), accuracy, auc))
    return test_loss, accuracy/100., auc

class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
                           'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,           
            "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)
