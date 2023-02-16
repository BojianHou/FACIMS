# the code is inspired by: https://github.com/katerakelly/pytorch-maml

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import list_mult
from .stochastic_layers import StochasticLinear, StochasticLayer
from .layer_inits import init_layers

# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------
def count_weights(model):
    # note: don't counts batch-norm parameters
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            count += list_mult(m.weight.shape)
            if hasattr(m, 'bias'):
                if m.bias == None:
                    continue
                count = count + list_mult(m.bias.shape)
        elif isinstance(m, StochasticLayer):
            count += m.weights_count
    return count


#  -------------------------------------------------------------------------------------------
#  Main function
#  -------------------------------------------------------------------------------------------
def get_model(prm, model_type='Stochastic'):

    model_name = prm.model_name

    # Define default layers functions
    def linear_layer(in_dim, out_dim, use_bias=True):
        if model_type == 'Standard':
            return nn.Linear(in_dim, out_dim, use_bias)
        elif model_type == 'Stochastic':
            return StochasticLinear(in_dim, out_dim, prm, use_bias)

    #  Return selected model:
    if model_name == 'FcNet3':
        model = FcNet3(model_type, model_name, linear_layer, prm)
    elif model_name == 'FcNet4':
        model = FcNet4(model_type, model_name, linear_layer, prm)
    elif model_name == 'FcNet6':
        model = FcNet6(model_type, model_name, linear_layer, prm)
    # elif model_name == 'ConvNet3':
    #     model = ConvNet3()
    else:
        raise ValueError('Invalid model_name')

    # Move model to device (GPU\CPU):
    model.to(prm.device)

    # init model:
    init_layers(model, prm.log_var_init)

    model.weights_count = count_weights(model)

    return model

#  -------------------------------------------------------------------------------------------
#   Base class for all stochastic models
# -------------------------------------------------------------------------------------------
class general_model(nn.Module):
    def __init__(self):
        super(general_model, self).__init__()

    def set_eps_std(self, eps_std):
        old_eps_std = None
        for m in self.modules():
            if isinstance(m, StochasticLayer):
                old_eps_std = m.set_eps_std(eps_std)
        return old_eps_std

    def _init_weights(self, log_var_init):
        init_layers(self, log_var_init)




# -------------------------------------------------------------------------------------------
# Models collection
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
#  3-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet3(general_model):
    def __init__(self, model_type, model_name, linear_layer, prm):
        super(FcNet3, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.layers_names = ('FC1', 'FC2', 'FC_out')
        input_size = prm.input_shape
        output_dim = prm.output_dim


        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 100
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc_out = linear_layer(n_hidden2, output_dim)

        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x.squeeze(dim=-1))
        return x

# -------------------------------------------------------------------------------------------
#  4-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet4(general_model):
    def __init__(self, model_type, model_name, linear_layer, prm):
        super(FcNet4, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.layers_names = ('FC1', 'FC2', 'FC3', 'FC_out')
        input_size = prm.input_shape
        output_dim = prm.output_dim


        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 200
        n_hidden3 = 100
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc3 = linear_layer(n_hidden2, n_hidden3)
        self.fc_out = linear_layer(n_hidden3, output_dim)

        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x.squeeze(dim=-1))
        return x


# -------------------------------------------------------------------------------------------
#  5-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet6(general_model):
    def __init__(self, model_type, model_name, linear_layer, prm):
        super(FcNet6, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.layers_names = ('FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC_out')
        input_size = prm.input_shape
        output_dim = prm.output_dim


        self.input_size = input_size
        n_hidden1 = 400
        n_hidden2 = 200
        n_hidden3 = 100
        n_hidden4 = 100
        n_hidden5 = 50
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc3 = linear_layer(n_hidden2, n_hidden3)
        self.fc4 = linear_layer(n_hidden3, n_hidden4)
        self.fc5 = linear_layer(n_hidden4, n_hidden5)
        self.fc_out = linear_layer(n_hidden5, output_dim)
        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc4(x)
        x = F.elu(x)
        x = self.fc5(x)
        x = F.elu(x)
        x = self.fc_out(x)
        x = torch.sigmoid(x.squeeze(dim=-1))
        return x


# class ConvNet3(nn.Module):
#     def __init__(self, in_channels=3):
#         super(ConvNet3, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, bias=True)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=True)
#         self.Pool_en = nn.AvgPool2d(4)
#
#         self.drop_fc = nn.Dropout(0.5)
#         self.activ = nn.Sigmoid()
#         self.activ_pred = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(32)
#
#         self.surv_pred1 = nn.Linear(128, 32)
#         self.surv_pred2 = nn.Linear(32, 32)
#         self.surv_pred3 = nn.Linear(32, 1, bias=False)
#
#         self.drop_conv = nn.Dropout(0.5)
#         self.drop_fc = nn.Dropout(0.5)
#         self.Pool = nn.MaxPool2d(3)
#         self.activ = nn.Sigmoid()
#
#         self.bn1_conv = nn.BatchNorm2d(32)
#         self.bn2_conv = nn.BatchNorm2d(64)
#         self.bn3_conv = nn.BatchNorm2d(128)
#
#     def forward(self, x):
#         x = self.bn1_conv(self.drop_conv(self.activ(self.conv1(x.unsqueeze(1)))))
#         x = self.bn2_conv(self.drop_conv(self.Pool(self.activ(self.conv2(x)))))
#         x = self.bn3_conv(self.drop_conv(self.Pool(self.activ(self.conv3(x)))))
#         x_en = self.Pool_en(x).squeeze()
#         x = x_en.view(x.shape[0], -1)
#         x = self.bn1(self.activ(self.surv_pred1(x)))
#         x = self.bn2(self.activ(self.surv_pred2(x)))
#         x = self.activ_pred(self.surv_pred3(x))
#         return x, x_en
