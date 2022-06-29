import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            output_shape = output[0].shape
            flops['bias'] = 0

            if isinstance(module, nn.Linear):
                if module.bias != None:
                    flops['bias'] = module.bias.size()[0] * batch_size
                flops['weight'] = 2 * np.prod(module.weight.size()) * batch_size

            if isinstance(module, nn.Conv2d):
                if module.bias != None:
                    flops['bias'] = np.prod(output_shape) * batch_size
                layer_shape = list(module.parameters())[0].size()
                flops['weight'] = np.prod(output_shape) * 4 * layer_shape[-1] * layer_shape[-2] * batch_size

            if isinstance(module, nn.BatchNorm1d):
                flops['bias'] = 0
                flops['weight'] = 0

            if isinstance(module, nn.BatchNorm2d):
                flops['bias'] = 0
                flops['weight'] = 0
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    return sum([np.prod(layer.size()) for layer in list(model.parameters())]) 



def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    input_size = np.prod(input_shape)
    dummy = torch.ones(input_shape)
    output_size = np.prod(model(dummy).size())
    return (input_size + output_size) * 4

