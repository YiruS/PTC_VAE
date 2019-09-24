#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def initialize_weights(*models):
    '''
    Module to initialize pytorch model weights
    :param list of models
    '''
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def summary(model, input_size, batch_size=-1, device="cuda"):
    '''
    Module to display pytorch model architecture and model parameters/size stats
    :param model: pytorch model
    :param input_size: 3-dimensional tuple of input image size (c,h,w)
    :param batch_size: default is -1
    :param device: device type: cpu or cuda
    '''
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            if hasattr(module, "kernel_size"):
                summary[m_key]["filter"] = module.kernel_size

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).to(device).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20} {:>20} {:>10} {:>10} {:>8} {:>5}".format(
        "Layer (type)", "Output Shape", "Filter", "MAC #", "Param #", "Seperable-Conv")
    # line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_mac = 0
    total_params = 0
    total_output = 0
    trainable_params = 0
    in_channel = input_size[0][0]
    bool_linear = False
    print(in_channel)
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if 'Conv' in layer:
            filter = summary[layer]["filter"]
            out_channel = summary[layer]["output_shape"][1]
            flat_out_channel = out_channel * np.prod(summary[layer]["output_shape"][2:])
            mac = (np.prod(summary[layer]['filter']) * out_channel) * in_channel * np.prod(summary[layer]["output_shape"][2:])
            in_channel = out_channel

        elif 'Linear' in layer:
            filter = None
            if not bool_linear:
                lin_in_channel = flat_out_channel
                bool_linear = True
            lin_out_channel = summary[layer]["output_shape"][1]
            mac = lin_in_channel * lin_out_channel
            lin_in_channel = lin_out_channel

        else:
            mac = 0
            filter = None

        line_new = "{:>20} {:>20} {:>10} {:>10} {:>8}".format(layer, str(
            summary[layer]["output_shape"]), str(filter), mac, "{0:,}".format(summary[layer]["nb_params"]))
        #line_new = "{:>20}  {:>25} {:>15}".format(layer,str(summary[layer]["output_shape"]),"{0:,}".format(summary[layer]["nb_params"]),)
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        total_mac += mac
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if summary[layer]["nb_params"] > 0:
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_mac_size = abs(total_mac / (1000**2))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("Total MACs: {0:,}".format(total_mac))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("Estimated MACs (M): %.2f" % total_mac_size)
    print("----------------------------------------------------------------")
