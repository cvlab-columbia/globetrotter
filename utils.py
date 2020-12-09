import PIL
import cv2
import numpy as np
import os
import random
import shutil
import torch
from functools import partial
from torch._six import int_classes, string_classes, container_abcs
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format, default_collate


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


def save_checkpoint(model, optim, is_best, epoch, path, global_step=-1, fn='checkpoint.pth',
                    fn_best='checkpoint_best.pth', fn_cfg='config', args=None, save_always=False, list_lang=None):
    model = model.module if hasattr(model, 'module') else model
    checkpoint_fn = os.path.join(path, fn)
    args = vars(args)
    torch.save({'model': model.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch,
                'global_step': global_step,
                'args': {k: v for k, v in args.items() if k != 'writer'},
                'list_lang': dict(list_lang) if list_lang else None},
               checkpoint_fn)
    if is_best:
        shutil.copyfile(checkpoint_fn, os.path.join(path, fn_best))
    if save_always:
        shutil.copyfile(checkpoint_fn, os.path.join(path, fn.replace('.pth', f'_{epoch}.pth')))
    model.config.to_json_file(os.path.join(path, f'{fn_cfg}.json'))
    print(f'Checkpoint saved at: {os.path.join(path, fn)}')


def load_checkpoint(model, optim, path, fn='checkpoint.pth', fn_best='checkpoint_best.pth', load_best=False,
                    strict=True, list_lang=None):
    model = model.module if hasattr(model, 'module') else model
    checkpoint_fn = os.path.join(path, fn_best if load_best else fn)
    checkpoint = torch.load(checkpoint_fn, map_location=(torch.device("cuda", torch.cuda.current_device()) if
                                                         model.get_type_device() == 'cuda' else 'cpu'))
    model_to_load = checkpoint['model']
    list_lang_load = checkpoint['list_lang'] if 'list_lang' in checkpoint else None
    list_lang_load_reverse = {v: k for k, v in list_lang_load.items()} \
        if 'list_lang' in checkpoint and checkpoint['list_lang'] is not None else None
    list_lang = dict(list_lang) if list_lang else None

    if list_lang is not None:
        # language layer numbers can correspond to different languages depending on the list of languages
        provisional_dict = {}
        for layer in model_to_load.keys():
            if 'lang' in layer:
                num = int([part for part in layer.split('.') if 'lang' in part][0].replace('lang_', ''))
                lang = list_lang_load_reverse[num]
                if lang in list_lang:
                    new_num = list_lang[lang]
                    provisional_dict[layer.replace(f'lang_{num}', f'lang_{new_num}')] = model_to_load[layer].clone()
        for k, v in provisional_dict.items():
            # Overwrite the necessary layers.
            model_to_load[k] = v
        if set(list_lang.keys()) != set(list_lang_load.keys()):
            # If there are missing or extra languages, just load those that correspond
            assert not strict
    model.load_state_dict(model_to_load, strict=strict)

    try:
        optim.load_state_dict(checkpoint['optim'])
    except:
        print('Warning! Not loading optimizer')
    return checkpoint['epoch'], checkpoint.get('global_step', -1)


def return_all_leaf_children(model: torch.nn.Module, filter_layers_class: str = '') -> list:
    list_children = []
    for layer in model.children():
        if not list(layer.children()):  # if leaf node, add it to list
            if filter_layers_class in str(type(layer)):
                list_children.append(layer)
        else:
            list_children.extend(return_all_leaf_children(layer, filter_layers_class))
    return list_children


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def gather_score(x, n):
    if torch.distributed.is_initialized():
        to_tensorize = [xx * n for xx in x] + [n]
        xn = torch.Tensor(to_tensorize).cuda()
        torch.distributed.all_reduce(xn)
        n = xn[-1]
        x = [xx / n for xx in xn[:-1]]
        return x
    else:
        return x


# as data gets more complicated, this may be too general and require a custom method in the dataset class
# we are lucky that the index of the [PAD] token in text is 0 so we can naively 0 pad
def collate_fn(batch, ignore_lists=True, pad_before_stack=True, cat_tensors=False):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # taken from pytorch source code - pytorch data collater, but does not do anything with lists (avoids zip behavior)
    f = partial(collate_fn, ignore_lists=ignore_lists, pad_before_stack=pad_before_stack, cat_tensors=cat_tensors)
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if cat_tensors:
            return torch.cat(batch, 0, out=out)
        else:
            if pad_before_stack:
                return pad_sequence(batch, batch_first=True)
            else:
                return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return f([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: f([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(f(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        if ignore_lists:
            return batch
        else:
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def _get_mask_for_indexed_mean_max(x, indices, dim):
    """
    The indices are given always for every element across dimension 0, and the mean is performed for dimension dim
    """
    indices = indices.to(x.device)
    if dim == 0:
        raise Exception('dimension cannot be 0')
    elif dim < 0:
        dim = len(x.shape) + dim
    # assert indices.min() > 0
    mask = torch.arange(x.shape[dim])[None, :].to(x.device) < indices[:, None]
    # Match dimensions
    for i in range(1, dim):  # for every dimension in between
        mask.unsqueeze_(1)
    for i in range(dim, len(x.shape) - 1):
        mask.unsqueeze_(-1)
    mask[:, 0] = True  # Make sure at least the first position is True, in case indices is 0 (that would create error)
    return mask


def indexed_mean(x, indices, dim, keepdim=False):
    mask = _get_mask_for_indexed_mean_max(x, indices, dim)
    x = x.masked_fill(~mask, 0).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim).float()
    return x


def indexed_max(x, indices, dim, keepdim=False):
    mask = _get_mask_for_indexed_mean_max(x, indices, dim)
    x, _ = x.masked_fill(~mask, x.min()).max(dim=dim, keepdim=keepdim)
    return x
