import tokenization
import torch
import os
import collections
import json
import word2vec
from tqdm import tqdm
import numpy as np
import fasttext

data_path = '/path/to/datasets/dir'
dataset_info_path = '/path/to/dataset/info/dir'

config_data = 'all-lang_test-zh-en'  # 'all-lang_test-zh-en_cocoflickr'
tokenizer_type = 'huggingface'
language_split = 'training'
train_fasttext = True  # True implies fasttext, False implies word2vec

# The tokenizer has to exist beforehand, we just load it
tokenizer = tokenization.create_tokenizer(tokenizer_type, dataset_info_path=dataset_info_path,
                                          config_data=config_data, list_txt_files=None)

with open(f'config/data/{config_data}.json') as json_file:
    lang_split = json.load(json_file)
language_to_id = {lang: i for i, lang in enumerate(lang_split[language_split])}

dict_sentences = collections.defaultdict(list)
# The data info has to exist beforehand
for subsplit in ['train', 'var']:
    info_file = os.path.join(dataset_info_path,
                             f'dataset_info_{config_data}_{language_split}lang_{subsplit}subsplit.pth')
    if os.path.isfile(info_file):
        sample_index, sample_dict, language_dict, list_txt_files = torch.load(info_file)
        for i, (k, info_sample) in enumerate(tqdm(sample_dict.items())):
            caption = info_sample[0]
            caption = caption.replace('\n', '').replace('.', '')
            if caption.replace('\n', '').replace('.', '') == '':  # original caption was eg '\n'
                caption = ' '
            text_token_ids, text_tokens = tokenizer.encode(caption)
            language = info_sample[1]
            # Easier to work with numbers directly, word2vec works the same
            dict_sentences[language].append([str(element) for element in text_token_ids.numpy()])

# Word2vec (actually Token2vec)
pretrained_weights = {}
for language, sentences in dict_sentences.items():
    print(language)
    # The word2vec contains embeddings for every token. A lot of tokens do not appear in several languages. In that
    # case the embedding is zero, but it is not a problem because it will not be used. Same tokens will have different
    # embeddings in different languages.
    input_file = f'/tmp/text_files_lang{language}.txt'
    with open(input_file, 'w') as outfile:
        for sentence in sentences:
            outfile.write(' '.join(sentence) + '\n')

    if train_fasttext:
        model = fasttext.train_unsupervised(input_file, model='cbow', dim=300)
        embeddings = model.get_output_matrix()
        labels = model.get_words()  # I think (at least in this case) it is the same as model.get_labels()
        vector = np.zeros((len(tokenizer), 300))
        for i, label in enumerate(labels):
            if label.isdigit():
                vector[int(label)] = embeddings[i]
    else:
        output_file = f'/tmp/output_word2vec_lang{language}.txt'
        # Dimensionality of 300 because this is what they use in the Sigurdsson paper
        # cbow = 0 implies using skip-gram
        word2vec.word2vec(input_file, output_file, size=300, window=5, negative=5, verbose=True, cbow=0)
        model = word2vec.load(output_file)
        vector = np.zeros((len(tokenizer), 300))
        for index, value in zip(model.vocab, model.vectors):
            if index.isdigit():
                vector[int(index)] = value
    pretrained_weights[language] = vector

name_experiment = 'fasttext' if train_fasttext else 'word2vec'
path_save = os.path.join(dataset_info_path, f'{name_experiment}_weights_{config_data}_{language_split}lang.pth')
torch.save(pretrained_weights, path_save)
