# """
# This code creates the listdir.txt
# """
#
# import os
# from tqdm import tqdm
#
# data_path = '/proj/vondrick/shared/globetrotter/make_public/dataset'  # '/proj/vondrick/datasets'
#
# annotations_paths = ['COCO/annotations/', 'flickr30k/entities/', 'ConceptualCaptions/']
#
# annotations_paths = [a_path + 'translated_independent/' for a_path in annotations_paths]
#
# for path in annotations_paths:
#     with open(os.path.join(os.path.join(data_path, path, 'listing.txt')), 'w') as f:
#         for root, dirs, files in tqdm(os.walk(os.path.join(data_path, path), followlinks=True), desc=path):
#             for name in files:
#                 if name.endswith(".txt") and "listing" not in name:
#                     root_short = root.replace(os.path.join(data_path, path), '')
#                     f.writelines('./' + os.path.join(root_short, name) + '\n')


import os.path
import random

data_path = '/proj/vondrick/shared/globetrotter/make_public/dataset'  # '/proj/vondrick/datasets'  #

annotations_paths = ['COCO/annotations/', 'flickr30k/entities/', 'ConceptualCaptions/']

# This is for the multiple languages (translation) case
# prob_train = 0.9
# prob_val = 0.1
# prob_test = 0  # The test is separated from train/val at a language level
# annotations_paths = [a_path + 'translated_independent/' for a_path in annotations_paths]

# This is for only English
# prob_train = 0.8
# prob_val = 0.1
# prob_test = 0.1

# This is for the split that is translated to all languages (only for testing)
prob_train = 0.
prob_val = 0.
prob_test = 1.
annotations_paths = [a_path + 'translated_alllangs/' for a_path in annotations_paths]


assert prob_train + prob_val + prob_test == 1

for annotations_path in annotations_paths:
    listing_path = os.path.join(data_path, annotations_path, 'listing{}.txt')
    with open(listing_path.format(''), 'r') as in_file, open(listing_path.format('_train'), 'w') as out_train, \
            open(listing_path.format('_val'), 'w') as out_val, open(listing_path.format('_test'), 'w') as out_test:
        for line in in_file:
            rand = random.random()
            if rand < prob_train:
                out_train.write(line)
            elif rand < prob_train + prob_val:
                out_val.write(line)
            else:
                out_test.write(line)
