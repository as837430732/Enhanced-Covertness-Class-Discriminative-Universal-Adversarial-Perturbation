from __future__ import division

import torch
import torch.nn as nn
import torchvision.models as models
from networks.resnet import resnet20
from networks.vgg_cifar import VGG

def get_network(model_arch, input_size, num_classes=1000, finetune=False):
    #### CIFAR-10 & CIFAR-100 models ####
    if model_arch == "resnet20":
        net = resnet20(input_size=input_size, num_classes=num_classes)
    elif model_arch == "vgg16_cifar":
        net = VGG('VGG16', num_classes=num_classes)

    #### ImageNet models ####
    elif model_arch == "resnet50":
        net = models.resnet50(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
    elif model_arch == "vgg16":
        net = models.vgg16(pretrained=True)
        if finetune:
            set_parameter_requires_grad(net)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        if num_classes != 1000:
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Network {} not supported".format(model_arch))
    return net

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = False


