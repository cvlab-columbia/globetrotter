"""
Create a dictionary that relates an image path + name of the dataset to the path of the English caption
This is just because the clean captions in English are organized in subfolders that do not relate to the name of the
caption.
"""
import os
from tqdm import tqdm
import torch

dataset_info_path = '/path/to/dataset/info/dir'  # to store the file
dataset_path = '/path/to/datasets/dir'
annotations_paths = {'coco': 'COCO/annotations/', 'flickr': 'flickr30k/entities/', 'conceptual': 'ConceptualCaptions/'}

dict_paths = {}
for dataset, path in annotations_paths.items():
    for root, dirs, files in tqdm(os.walk(os.path.join(dataset_path, path, 'clean_sentences')), desc=path):
        for name in files:
            if name.endswith(".txt"):
                root_short = root.replace(os.path.join(dataset_path, path), '')
                dict_paths[(name[:-4], dataset)] = os.path.join(root_short, name)

torch.save(dict_paths, os.path.join(dataset_info_path, 'img_to_en.pth'))
