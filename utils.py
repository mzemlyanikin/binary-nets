import collections
from itertools import repeat
from urllib.parse import urlparse

import torch
import torch.utils.model_zoo as model_zoo


def load_weights(network, save_path, partial=False):
    if urlparse(save_path).scheme != '':
        pretrained_dict = model_zoo.load_url(save_path)
    else:
        pretrained_dict = torch.load(save_path)

    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    try:
        network.load_state_dict(pretrained_dict)
        print('Pretrained network has absolutely the same layers!')
    except:
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        try:
            network.load_state_dict(pretrained_dict)
            print('Pretrained network has excessive layers; Only loading layers that are used')
        except:
            print('Pretrained network has fewer layers; The following are not initialized:')
            not_initialized = set()
            partially_initialized = set()
            for k, v in pretrained_dict.items():
                if v.size() == model_dict[k].size():
                    model_dict[k] = v
                elif partial:
                    sizes_div = [model_dict[k].size()[i] % v.size()[i] for i in range(len(min(v.size(), model_dict[k].size())))]
                    if not any(sizes_div):
                        model_dict[k] = v.repeat(*[model_dict[k].size()[i] // v.size()[i] for i in range(len(min(v.size(), model_dict[k].size())))])
                    else:
                        min_shape = [min(v.size()[i], model_dict[k].size()[i]) for i in range(len(min(v.size(), model_dict[k].size())))]
                        if len(model_dict[k].size()) in [2, 4]: # fc and conv layers
                            model_dict[k][:min_shape[0], :min_shape[1], ...] = \
                                v[:min_shape[0], :min_shape[1], ...]
                        elif len(model_dict[k].size()) == 1:
                            model_dict[k][:min_shape[0]] = v[:min_shape[0]]
                        else:
                            print('{} has size: '.format(k, model_dict[k].size()))

            for k, v in model_dict.items():
                if k not in pretrained_dict or (not partial and v.size() != pretrained_dict[k].size()):
                    not_initialized.add(k)
                elif partial and v.size() != pretrained_dict[k].size():
                    partially_initialized.add(k)
            print(sorted(not_initialized))
            if partial:
                print('Partially initialized:')
                print(sorted(partially_initialized))
            network.load_state_dict(model_dict)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
