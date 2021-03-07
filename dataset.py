import os
import random
from collections import defaultdict
import functools
from multiprocessing import Manager

import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
import PIL
from PIL import Image, ImageFilter
import xmltodict
import pickle as pkl

import warnings

import utils
import time

warnings.simplefilter("ignore", UserWarning)


def get_color_distortion(s=1.0):  # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def round_up_to_odd(num):
    return int(np.ceil(num) // 2 * 2 + 1)


class GaussianSmoothing(object):
    def __init__(self, sigma_min=0.1, sigma_max=2.0, size_img=224, ksize=0.1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.size_img = size_img
        self.ksize = ksize  # percentage wrt the image size (assume square image)

    def __call__(self, image):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img_blur = image.filter(ImageFilter.GaussianBlur(sigma))
        return img_blur


class MultipleDatasets(data.Dataset):
    """
    Dataset class that combines the MSCOCO, Fickr30k and Conceptual Captions datasets
    """
    def __init__(self, dataset_path, dataset_info_path, img_size, subsplit, max_txt_seq_len, config_data,
                 prob_predict_token=1/6, augment_image=True, language_split='training', not_use_images=False,
                 randomize=True):

        list_datasets = ['coco', 'flickr', 'conceptual']
        self.dataset_path = dataset_path
        self.dataset_info_path = dataset_info_path
        self.img_size = img_size
        self.subsplit = subsplit
        self.max_txt_seq_len = max_txt_seq_len
        self.prob_predict_token = prob_predict_token  # see explanation in get_positions_to_predict
        self.config_data = config_data
        self.augment_image = augment_image and randomize
        self.not_use_images = not_use_images
        self.language_split = language_split
        # Not randomize implies not at all. Not even possible to compute image loss from augmentations. So a regular
        # validation that uses augmentations can be randomized (then the sampling is also random, which is not ideal,
        # but we are randomizing the augmentations anyway, so it is already random).
        self.randomize = randomize

        # Create a dictionary that given a sample_index (unique across languages) returns (0) the caption,
        # (1) the language, (2) the original captioning dataset, (3) the image path, (4) the original text path,
        # and (5) the index of the caption in that text file.

        # Load sample_index, sample dict, language_dict and list_txt_files if already computed and saved before
        info_file = '' if dataset_info_path is None else \
            os.path.join(dataset_info_path,
                         f'dataset_info_{config_data}_{language_split}lang_{subsplit}subsplit.pth')
        if os.path.isfile(info_file):
            self.sample_index, self.sample_dict, self.same_sample_dict, self.list_txt_files = torch.load(info_file)
        else:
            self.sample_index = 0
            self.sample_dict = {}
            self.same_sample_dict = defaultdict(list)
            self.list_txt_files = []  # Useful for the huggingface tokenizer
            for dataset_name in list_datasets:
                self.create_sample_dict(dataset_name)
            if dataset_info_path is not None:
                torch.save([self.sample_index, self.sample_dict, self.same_sample_dict, self.list_txt_files], info_file)

        with open(f'config/data/{self.config_data}.json') as json_file:
            lang_split = json.load(json_file)
        self.language_to_id = {lang: i for i, lang in enumerate(lang_split[language_split])}

        # If the input to Resize is just (size), then it will scale it proportionally so that the shortest side is size.
        # And then it will crop it. However, if we input (size, size), it will rescale everything to (size, size), and
        # the crop is completely useless.
        self.transform_image_base = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size) if randomize else transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # Same transformations as "A Simple Framework for Contrastive Learning of Visual Representations"
        self.transform_image_augment = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            get_color_distortion(),
            GaussianSmoothing(0, 2.0, self.img_size),  # We start from 0 so that natural images are in distribution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.tokenizer = None  # easier to create it later and make sure all splits have the same

        # Native python lists create memory leaks in multiprocessing (https://github.com/pytorch/pytorch/issues/13246)
        manager = Manager()
        self.list_txt_files = manager.list(self.list_txt_files)  # overhead of pickling the object
        self.same_sample_dict = manager.dict(self.same_sample_dict)  # overhead of pickling the object
        self.language_to_id = manager.dict(self.language_to_id)  # overhead of pickling the object

        # For large datasets, this gives problems when pickling, so we decompose it
        # self.sample_dict = manager.dict(self.sample_dict)
        self.sample_dict_0 = manager.dict({k: info[0] for k, info in self.sample_dict.items()})
        self.sample_dict_1 = manager.dict({k: info[1] for k, info in self.sample_dict.items()})
        self.sample_dict_2 = manager.dict({k: info[2] for k, info in self.sample_dict.items()})
        self.sample_dict_3 = manager.dict({k: info[3] for k, info in self.sample_dict.items()})
        self.sample_dict_4 = manager.dict({k: info[4] for k, info in self.sample_dict.items()})
        self.sample_dict_5 = manager.dict({k: info[5] for k, info in self.sample_dict.items()})

    def create_sample_dict(self, dataset_name):
        with open(f'config/data/{self.config_data}.json') as json_file:
            lang_split = json.load(json_file)
        if dataset_name not in lang_split['datasets']:
            return

        if dataset_name == 'coco':
            annotations_path = 'COCO/annotations'
        elif dataset_name == 'flickr':
            annotations_path = 'flickr30k/entities'
        elif dataset_name == 'conceptual':
            annotations_path = 'ConceptualCaptions'
        else:
            raise Exception(f'The dataset {dataset_name} is not a valid one.')

        if lang_split['other'] == 'same_split':
            annotations_path += '/translated_alllangs'
        else:
            annotations_path += '/translated_independent'

        set_imgs_conceptual = None
        if dataset_name == 'conceptual':
            set_imgs_conceptual = set(line.replace('\n', '') for line in
                                      open(os.path.join(self.dataset_path, 'ConceptualCaptions/listing_images.txt')))

        listing_file = f'listing_{self.subsplit}.txt'

        with open(os.path.join(self.dataset_path, annotations_path, listing_file), 'r') as listing:
            for index_to_remove, line in enumerate(tqdm(listing, desc=f'Sampling {dataset_name} dataset')):
                line = line.replace('\n', '')
                try:
                    lang, filename = line.split('/')[1:]
                    number = int(filename.split('.')[-2].split('_')[-1])
                except ValueError:  # Probably some other file like 'listing.txt' itself
                    continue
                if lang not in lang_split[self.language_split]:
                    continue
                if dataset_name == 'coco':
                    split = 'val' if 'val' in filename else 'train'
                    image_path_short = f'COCO/{split}2014/COCO_{split}2014_{number:012d}.jpg'
                elif dataset_name == 'flickr':
                    image_path_short = f'flickr30k/images/{number}.jpg'
                else:  # conceptual
                    if len(str(number)) > 8:
                        number = int(str(number)[1:])
                    exists_image = False
                    for extension in ['.jpeg', '.jpg', '.png']:
                        if f'{number:08d}{extension}' in set_imgs_conceptual:
                            image_path_short = f'ConceptualCaptions/train/{number:08d}{extension}'
                            exists_image = True
                            break
                    if not exists_image:
                        continue  # ignore this sample
                image_path = os.path.join(self.dataset_path, image_path_short)

                text_path = os.path.join(self.dataset_path, annotations_path, line)
                text_path_short = os.path.join(annotations_path, line)
                self.list_txt_files.append(text_path)
                with open(text_path) as f:
                    for i, caption in enumerate(f):
                        # Used to create pairs of idxs for supervised learning.
                        if dataset_name == 'coco':
                            number = image_path.split('/')[-1].split('.')[0].split('_')[-1]
                        elif dataset_name == 'flickr' or dataset_name == 'conceptual':
                            number = image_path.split('/')[-1].split('.')[0]
                        idx = int({'flickr': '1', 'coco': '2', 'conceptual': '3'}[dataset_name] + str(i) + str(number))

                        self.same_sample_dict[idx].append(self.sample_index)
                        self.sample_dict[self.sample_index] = [caption, lang, dataset_name, image_path_short,
                                                               text_path_short, i]
                        self.sample_index += 1

    def load_image(self, img_path):
        img_path = os.path.join(self.dataset_path, img_path)
        if self.not_use_images:
            return -1, -1, -1, -1
        with warnings.catch_warnings(record=True) as w:
            len_images = 2 if self.augment_image else 1
            try:
                img = default_loader(img_path)
                if self.augment_image:
                    img_1 = self.transform_image_augment(img)
                    img_2 = self.transform_image_augment(img)
                    img = torch.stack([img_1, img_2])
                else:
                    img = self.transform_image_base(img).unsqueeze(0)
            except FileNotFoundError:
                print(f'Image in path {img_path} does not exist')
                img = torch.zeros((len_images, 3, self.img_size, self.img_size))
            except (PIL.UnidentifiedImageError, OSError):
                print(f'Image in path {img_path} is empty or corrupted')
                img = torch.zeros((len_images, 3, self.img_size, self.img_size))
        return img

    def load_text(self, caption: str, pad: bool):
        text_token_ids, text_tokens = self.tokenizer.encode(caption)
        text_len = len(text_tokens)
        if pad:
            text_token_ids, text_tokens = self.pad_text(text_token_ids, text_tokens)
        else:
            text_tokens = text_tokens[:self.max_txt_seq_len]
            text_token_ids = text_token_ids[:self.max_txt_seq_len]

        return text_token_ids, text_tokens, text_len

    def pad_text(self, txt_ids, txt, text_len=None):
        """
        :param text_len: length we want to pad to
        :param txt: list of text tokens
        """
        text_len = text_len or self.max_txt_seq_len
        txt = txt[:text_len]
        txt_ids = txt_ids[:text_len]
        txt = txt + ['<pad>' for i in range(text_len - len(txt))]
        pad_tensor = torch.tensor([self.tokenizer.index_special_tokens['<pad>']] * (text_len - len(txt_ids)))
        try:
            txt_ids = torch.cat((txt_ids, pad_tensor), dim=0) if len(pad_tensor) > 0 else txt_ids
        except:
            print('hey')

        return txt_ids, txt

    def get_positions_to_predict(self, text_token_ids, text_len, seed=None):
        """
        Returns the positions that will have to be predicted by the language model. Each token is marked as "to be
        predicted" with a probability of self.prob_predict_token. If no token has been selected, then one token is
        selected randomly (for each sentence, it returns always at least one token to be predicted)
        """
        if not self.randomize:
            previous_state = np.random.get_state()
            np.random.seed(seed=seed)

        len_limit = np.minimum(text_len, text_token_ids.shape[0])
        indices_to_predict = np.random.rand(len_limit) < self.prob_predict_token
        positions_to_predict = np.where(indices_to_predict)[0]
        if len(positions_to_predict) == 0 and self.prob_predict_token > 0:  # always return at least one position
            positions_to_predict = np.array([np.random.randint(0, len_limit)])
        gt_tokens = torch.LongTensor(self.max_txt_seq_len).fill_(-1)
        gt_tokens[positions_to_predict] = text_token_ids[positions_to_predict]

        if not self.randomize:
            np.random.set_state(previous_state)

        return positions_to_predict, gt_tokens

    def get_info_sample(self, sample):

        img = None
        if not self.not_use_images:
            # info_sample = self.sample_dict[sample]
            image_path = self.sample_dict_3[sample]
            img = self.load_image(image_path)
        caption = self.sample_dict_0[sample]
        caption = caption.replace('\n', '').replace('.', '')
        if caption.replace('\n', '').replace('.', '') == '':  # original caption was eg '\n'
            caption = ' '
        text_token_ids, text_tokens, text_len = self.load_text(caption, pad=True)
        text_tokens = '/'.join(text_tokens)
        positions_to_predict, gt_tokens = self.get_positions_to_predict(text_token_ids, text_len, seed=sample)
        language = self.language_to_id[self.sample_dict_1[sample]]

        return gt_tokens, img, text_tokens, text_len, positions_to_predict, text_token_ids, language

    def __len__(self):
        return len(self.sample_dict)

    def __getitem__(self, sample_idx):
        gt_tokens, img, text_tokens, text_len, positions_to_predict, text_token_ids, language = \
            self.get_info_sample(sample_idx)

        img = img if not self.not_use_images else -1

        text_token_ids = utils.collate_fn([text_token_ids], cat_tensors=False)

        return {'imgs': img, 'text': text_token_ids, 'text_tokens': text_tokens, 'text_len': text_len,
                'language': language, 'pos_to_predict': positions_to_predict, 'gt_tokens': gt_tokens,
                'idxs': sample_idx}
