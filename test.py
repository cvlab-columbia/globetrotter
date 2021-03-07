import os.path
import pickle
import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm
import utils
import losses
import sys
import csv
import matplotlib.pyplot as plt
import scipy.linalg
import torch.nn.functional as F
from collections import Counter
import argparse
from pytorch_transformers import AdamW
import models
from multiprocessing import Pool as ThreadPool
import tokenization
from multiprocessing import Pool
from copy import deepcopy
from matplotlib import gridspec
from torch.cuda.amp import autocast
from termcolor import colored, cprint
from colorama import Fore, Back, Style, init


dict_langs = {'ab': 'Abkhazian', 'aa': 'Afar', 'af': 'Afrikaans', 'ak': 'Akan', 'sq': 'Albanian', 'am': 'Amharic',
              'ar': 'Arabic', 'an': 'Aragonese', 'hy': 'Armenian', 'as': 'Assamese', 'av': 'Avaric', 'ae': 'Avestan',
              'ay': 'Aymara', 'az': 'Azerbaijani', 'bm': 'Bambara', 'ba': 'Bashkir', 'eu': 'Basque', 'be': 'Belarusian',
              'bn': 'Bengali', 'bh': 'Bihari languages', 'bi': 'Bislama', 'bs': 'Bosnian', 'br': 'Breton',
              'bg': 'Bulgarian', 'my': 'Burmese', 'ca': 'Catalan, Valencian', 'ch': 'Chamorro', 'ce': 'Chechen',
              'ny': 'Chichewa, Chewa, Nyanja', 'zh': 'Chinese', 'cv': 'Chuvash', 'kw': 'Cornish', 'co': 'Corsican',
              'cr': 'Cree', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'dv': 'Divehi, Dhivehi, Maldivian',
              'nl': 'Dutch', 'dz': 'Dzongkha', 'en': 'English', 'eo': 'Esperanto', 'et': 'Estonian', 'ee': 'Ewe',
              'fo': 'Faroese', 'fj': 'Fijian', 'fi': 'Finnish', 'fr': 'French', 'ff': 'Fulah', 'gl': 'Galician',
              'ka': 'Georgian', 'de': 'German', 'el': 'Greek', 'gn': 'Guarani', 'gu': 'Gujarati',
              'ht': 'Haitian, Haitian Creole', 'ha': 'Hausa', 'he': 'Hebrew', 'hz': 'Herero', 'hi': 'Hindi',
              'ho': 'Hiri Motu', 'hu': 'Hungarian', 'ia': 'Interlingua (International Auxiliary Language Association)',
              'id': 'Indonesian', 'ie': 'Interlingue, Occidental', 'ga': 'Irish', 'ig': 'Igbo', 'ik': 'Inupiaq',
              'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese', 'jv': 'Javanese',
              'kl': 'Kalaallisut, Greenlandic', 'kn': 'Kannada', 'kr': 'Kanuri', 'ks': 'Kashmiri', 'kk': 'Kazakh',
              'km': 'Central Khmer', 'ki': 'Kikuyu, Gikuyu', 'rw': 'Kinyarwanda', 'ky': 'Kirghiz, Kyrgyz', 'kv': 'Komi',
              'kg': 'Kongo', 'ko': 'Korean', 'ku': 'Kurdish', 'kj': 'Kuanyama, Kwanyama', 'la': 'Latin',
              'lb': 'Luxembourgish, Letzeburgesch', 'lg': 'Ganda', 'li': 'Limburgan, Limburger, Limburgish',
              'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga', 'lv': 'Latvian', 'gv': 'Manx',
              'mk': 'Macedonian', 'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam', 'mt': 'Maltese', 'mi': 'Maori',
              'mr': 'Marathi', 'mh': 'Marshallese', 'mn': 'Mongolian', 'na': 'Nauru', 'nv': 'Navajo, Navaho',
              'nd': 'North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'nb': 'Norwegian Bokmål',
              'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'ii': 'Sichuan Yi, Nuosu', 'nr': 'South Ndebele',
              'oc': 'Occitan', 'oj': 'Ojibwa',
              'cu': 'Church Slavic, Old Slavonic, Church Slavonic, Old Bulgarian, Old Church Slavonic', 'om': 'Oromo',
              'or': 'Oriya', 'os': 'Ossetian, Ossetic', 'pa': 'Punjabi, Panjabi', 'pi': 'Pali', 'fa': 'Persian',
              'fa-AF': 'Dari', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh',
              'rn': 'Rundi', 'ro': 'Romanian', 'ru': 'Russian', 'sa': 'Sanskrit', 'sc': 'Sardinian', 'sd': 'Sindhi',
              'se': 'Northern Sami', 'sm': 'Samoan', 'sg': 'Sango', 'sr': 'Serbian', 'gd': 'Gaelic, Scottish Gaelic',
              'sn': 'Shona', 'si': 'Sinhala, Sinhalese', 'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali',
              'st': 'Southern Sotho', 'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili', 'ss': 'Swati',
              'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya',
              'bo': 'Tibetan', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Tswana', 'to': 'Tonga (Tonga Islands)',
              'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'ug': 'Uighur, Uyghur',
              'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük',
              'wa': 'Walloon', 'cy': 'Welsh', 'wo': 'Wolof', 'fy': 'Western Frisian', 'xh': 'Xhosa', 'yi': 'Yiddish',
              'yo': 'Yoruba', 'za': 'Zhuang, Chuang', 'zu': 'Zulu'}
list_family = ["hu", "fi", "et", "lv", "ar", "am", "he", "fa", "fa-AF", "ps", "es", "pt", "fr", "it", "ro", "nl", "af",
               "de", "da", "sv", "no", "pl", "cs", "sk", "sl", "bg", "sr", "bs", "hr", "uk", "ru", "sq", "el", "ka",
               "az", "tr", "ha", "so", "sw", "vi", "th", "hi", "ur", "bn", "ta", "ms", "id", "tl", "ko", "ja"]


def main():
    name_script = sys.argv[1]
    sys.argv = [''] + sys.argv[2:]
    if name_script == 'sentence_translation':
        sentence_translation()
    if name_script == 'align_words':
        align_words()
    else:
        print('That option does not exist')


def sentence_translation():
    """
    Previously run extract_features()
    """
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument('--name_model', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--results_path', type=str, default='', help='Results path')
    parser.add_argument('--extracted_features_name', type=str, default='', help='Name of the features pickle')
    parser.add_argument('--method', type=str, default='common', choices=['common', 'alpha', 'beta'],
                        help='How to choose anchors')
    parser.add_argument('--alpha_xm', action='store_true', help='In case method is alpha, whether or not to use '
                                                                'cross-modal information')
    parser.add_argument('--normalize', action='store_true', help='Normalize features across languages')

    args = parser.parse_args()

    txt_to_txt_embs, txt_to_img_embs, img_to_txt_embs, img_to_img_embs, info, text_lens, count_tokens, *_ = \
        torch.load(os.path.join(args.results_path, args.name_model, args.extracted_features_name))

    if len(txt_to_txt_embs) == 0:  # sigurdsson:
        txt_to_txt_embs = txt_to_img_embs

    np.random.seed(0)  # Same indices for all models to be compared

    random_indices = np.random.permutation(txt_to_txt_embs.shape[0])
    text_predictions = txt_to_txt_embs[random_indices]
    info = [info[i] for i in random_indices]
    text_predictions = torch.tensor(text_predictions).squeeze(1)

    # for simplicity, just use N x n_lang samples. All samples will have n_lang-1 positives
    N = 200
    used_indices = []
    set_numbers = set()
    list_numbers = []
    num = None
    for i, element in tqdm(enumerate(info)):
        if len(element) > 2:
            dataset = element[2]
            jpg = element[3]
            caption_id = element[5]
            if dataset == 'coco':
                number = jpg.split('/')[-1].split('.')[0].split('_')[-1]
            elif dataset == 'flickr' or dataset == 'conceptual':
                number = jpg.split('/')[-1].split('.')[0]
            else:
                raise Exception(f'Wrong dataset {dataset}')

            number = int({'flickr': '1', 'coco': '2', 'conceptual': '3'}[dataset] + str(caption_id) + str(number))
        else:
            number = element[0]
        if num is None:
            num = number
        if number not in set_numbers and len(set_numbers) >= N:
            continue
        elif number not in set_numbers:
            set_numbers.add(number)
        used_indices.append(i)
        list_numbers.append(number)

    list_numbers = np.array(list_numbers)

    text_predictions = text_predictions[used_indices]

    # Do not match same languages
    list_languages = set()
    lang_vector = []
    for index in used_indices:
        list_languages.add(info[index][1])
        lang_vector.append(info[index][1])
    list_languages = list(list_languages)
    lang_dict = {list_languages[i]: i for i in range(len(list_languages))}
    lang_vector = torch.tensor([lang_dict[lang] for lang in lang_vector])

    punctuation_per_sample, matrix_results = \
        _process_alignment_languages(list_numbers, None, lang_vector, text_predictions.cpu().numpy(), None,
                                     "without_procrustes", name_model=args.name_model, results_path=args.results_path)

    score_matrix = np.zeros([len(matrix_results), len(matrix_results)])
    for i, (lang, queries) in enumerate(matrix_results.items()):
        for j, (query, val) in enumerate(queries.items()):
            score_matrix[i, j] = val

    list_family_ = [element for element in list_family if element in lang_dict]
    list_family_ = [lang_dict[element] for element in list_family_]

    # Sort by family
    score_matrix_index = np.array(list_family_)

    score_matrix = score_matrix[score_matrix_index][:, score_matrix_index]

    score_matrix = (score_matrix - score_matrix.transpose())**3

    lang_dict2 = {k: np.where(score_matrix_index == v)[0][0] for k, v in lang_dict.items()}
    lang_dict3 = {v: k for k, v in lang_dict2.items()}
    lang_dict2 = {lang_dict3[v]: v for v in range(len(lang_dict3))}

    fig = plt.figure(figsize=(7, 7), dpi=200)
    spec = gridspec.GridSpec(ncols=1, nrows=1, width_ratios=[1])

    ax0 = fig.add_subplot(spec[0])
    ax0.matshow(score_matrix, cmap=plt.get_cmap('winter'))
    ax0.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax0.set_xticks(np.arange(len(lang_dict2)))
    ax0.set_yticks(np.arange(len(lang_dict2)))
    ax0.set_xticklabels([dict_langs[k] for k in list(lang_dict2.keys())], fontsize=9, rotation=90)
    ax0.set_yticklabels([dict_langs[k] for k in list(lang_dict2.keys())], fontsize=9)
    ax0.set_ylabel('Query language', fontsize=12)
    ax0.set_xlabel('Target language', fontsize=12)

    fig.tight_layout()
    fig.show()

    plt.savefig(os.path.join(args.results_path, args.name_model, 'matrix_sentence.pdf'), format='pdf')
    print(f'Saved in {os.path.join(args.results_path, args.name_model, "matrix_sentence.pdf")}')

    print(punctuation_per_sample.mean())


def _process_alignment_languages(list_numbers=None, list_numbers_query=None, lang_vector=None, text_predictions=None,
                                 text_predictions_query=None, name=None, csls=False,
                                 cosine=True, name_model=None, results_path=None):
    """ Previously run extract_features() """
    if csls:
        assert cosine
        assert text_predictions_query is None  # simply not implemented and because I don't really use it I won't implement it

    NUM_LANG = 49
    if text_predictions_query is None:
        text_predictions_query = text_predictions
        list_numbers_query = list_numbers
        NUM_LANG = 50

    N = 10000
    num_batches = int(np.ceil(text_predictions_query.shape[0] / N))

    mean_similarity = None
    if csls:
        # Implementing CSLS from https://arxiv.org/pdf/1710.04087.pdf
        # For each point, we use as possible neighbors the points of ALL the other languages
        # Not super efficient because we will compute similarities again after this, but it is the easiest way to add
        # this to the code without modifying anything else
        mean_similarity = np.zeros(text_predictions.shape[0])
        K = 10  # "The performance is very stable and therefore K does not need cross-validation"
        text_predictions = text_predictions / np.sqrt((text_predictions ** 2).sum(-1)[..., np.newaxis])
        text_predictions_query = text_predictions_query / np.sqrt((text_predictions_query ** 2).sum(-1)[..., np.newaxis])
        for i in tqdm(range(num_batches)):
            start = i * N
            end = (i + 1) * N
            similarities_i = np.matmul(text_predictions_query[start:end], text_predictions.transpose())
            lang_vector_i = lang_vector[start:end]

            lang_vector_expanded_1 = lang_vector.unsqueeze(0).expand((lang_vector_i.shape[0], lang_vector.shape[0]))
            lang_vector_expanded_2 = lang_vector_i.unsqueeze(1).expand((lang_vector_i.shape[0], lang_vector.shape[0]))
            # We do not want to select the same language
            similarities_i[lang_vector_expanded_1 == lang_vector_expanded_2] = -1000

            topk_indices = np.argpartition(-similarities_i, kth=K, axis=1)[:, :K]
            topk_values = similarities_i[np.tile(np.array(range(lang_vector_i.shape[0]))[..., np.newaxis], (1, K)), topk_indices]
            mean_similarity[start:end] = topk_values.mean(1)

    histogram = np.zeros(text_predictions.shape[0])
    if cosine:
        text_predictions = text_predictions / np.sqrt((text_predictions ** 2).sum(-1)[..., np.newaxis])
        text_predictions_query = text_predictions_query / np.sqrt((text_predictions_query ** 2).sum(-1)[..., np.newaxis])

    punctuation_per_sample = np.zeros(text_predictions_query.shape[0])

    # Create langxlang matrix (query, target)
    matrix_results = {lang_query: {lang_target: 0 for lang_target in range(NUM_LANG)} for lang_query in range(NUM_LANG)}

    for i in tqdm(range(num_batches)):
        start = i * N
        end = (i + 1) * N

        similarities_i = np.matmul(text_predictions_query[start:end], text_predictions.transpose())
        if csls:
            # Mean similarities of the two points towards their knn
            mean_similarity_expanded = np.tile(mean_similarity[start:end][..., np.newaxis], (1, text_predictions.shape[0])) + \
                                       np.tile(mean_similarity[np.newaxis, ...], (similarities_i.shape[0], 1))
            similarities_i = 2*similarities_i - mean_similarity_expanded

        if lang_vector is not None:
            lang_vector_i = lang_vector[start:end]

            lang_vector_expanded_1 = lang_vector.unsqueeze(0).expand((lang_vector_i.shape[0], lang_vector.shape[0]))
            lang_vector_expanded_2 = lang_vector_i.unsqueeze(1).expand((lang_vector_i.shape[0], lang_vector.shape[0]))
            # We do not want to select the same language
            similarities_i[lang_vector_expanded_1 == lang_vector_expanded_2] = -1000
            # similarities_i = similarities_i.astype(np.float16)

        # Besides computing the argmax, we also compute the position for all 49 positive examples, to create histogram
        sorted_idx = (-similarities_i).argsort()
        list_numbers_ = list_numbers[sorted_idx]
        for j in tqdm(range(similarities_i.shape[0])):
            sorted_langs = lang_vector[sorted_idx[j][:NUM_LANG]]
            # The (j, j) sample will still be included, but it will go to the end of the histogram. Take into account
            histogram += (list_numbers_[j] == list_numbers_query[start + j]).astype(int)
            indices_first = (list_numbers_[j] == list_numbers_query[start + j]).astype(int)[:NUM_LANG]
            punctuation_per_sample[start+j] = indices_first.sum()
            for lang_target in lang_vector[start:end][sorted_idx[j][:NUM_LANG]][np.where(indices_first)[0]]:
                matrix_results[lang_vector[start+j].item()][lang_target.item()] += 1

    # Group histogram into more general columns
    num_cols = 100
    oldcols_per_newcol = np.floor(len(histogram)/num_cols).astype(int)
    histogram_ = np.array([histogram[i*oldcols_per_newcol:(i+1)*oldcols_per_newcol].sum() for i in range(num_cols)])
    plt.clf()
    plt.bar([oldcols_per_newcol*i for i in range(len(histogram_))], histogram_, width=oldcols_per_newcol*0.8)
    path_save = os.path.join(results_path, name_model, f'hist_alignment_languages_{name}')
    plt.savefig(path_save)

    return punctuation_per_sample, matrix_results


def _word_procrustes(embeddings, list_tokens, counter_tok, sigurdsson=False):
    # Just for convenience
    lang_id_to_str = {i: lang for i, lang in enumerate(list_tokens.keys())}

    # First, select only the tokens that appear in the 10th percentile *for each language*
    indexes_tokens = {}
    for lang, tokens_lang in list_tokens.items():
        count_tokens_lang = np.array([counter_tok[tok] if tok in counter_tok else 0 for tok in tokens_lang])
        percent_sentences = 100  # Use 100% of the sentences with higher count tokens
        threshold = np.percentile(count_tokens_lang, 100-percent_sentences)
        indexes_tokens_lang = count_tokens_lang > threshold
        if sigurdsson:
            indexes_tokens[lang] = np.array(indexes_tokens_lang)
        else:
            # the indexes have to be wrt all the tokens
            indexes_tokens[lang] = np.array(tokens_lang)[indexes_tokens_lang]

    # Compute similarity matrix for every pair of languages (and all the tokens selected for every language)
    if sigurdsson:
        similarities = {}
        for lang_0 in range(len(list_tokens)):
            for lang_1 in range(len(list_tokens)):
                if lang_1 <= lang_0:
                    continue
                lang_0_str = lang_id_to_str[lang_0]
                lang_1_str = lang_id_to_str[lang_1]
                sim = torch.mm(embeddings[lang_0_str][indexes_tokens[lang_0_str]],
                               embeddings[lang_1_str][indexes_tokens[lang_1_str]].transpose(1, 0))
                similarities[lang_0_str, lang_1_str] = sim.cpu().numpy()

    else:
        similarities = torch.mm(embeddings, embeddings.transpose(1, 0)).cpu().numpy()

    # The idea is that the ones that match here (have high similarity) will correspond because this similarity was
    # computed only on those tokens that appear a lot
    # In the paper they select only the matches such that the two of them are nearest neighbors of the other, but
    # because it is too restrictive, we work with topk (k=1 would be equal to the paper)
    k = 5
    correlation_matrices = dict()
    for lang_0 in range(len(list_tokens)):
        for lang_1 in range(len(list_tokens)):
            if lang_1 <= lang_0:
                continue
            # For the language pair, find correspondences
            # Use the previous similarities information to select the ground truth pairs for Procrustes
            if sigurdsson:
                similarities_pair = similarities[lang_id_to_str[lang_0], lang_id_to_str[lang_1]]
            else:
                similarities_pair = similarities[indexes_tokens[lang_id_to_str[lang_0]][:, None],
                                                 indexes_tokens[lang_id_to_str[lang_1]]]
            topk_0to1 = np.argpartition(-similarities_pair, kth=k, axis=1)[:, :k]
            topk_1to0 = np.argpartition(-similarities_pair, kth=k, axis=0).transpose()[:, :k]
            topk_0to1_set = set()
            topk_1to0_set = set()
            for i in range(similarities_pair.shape[0]):
                for j in range(k):
                    topk_0to1_set.add((i, topk_0to1[i][j]))
            for i in range(similarities_pair.shape[1]):
                for j in range(k):
                    topk_1to0_set.add((topk_1to0[i][j], i))
            # mutual nearest neighbours
            matches_indexes = np.array(list(set(topk_0to1_set).intersection(set(topk_1to0_set))))

            # Compute correlation matrix for each pair of languages
            if sigurdsson:
                features_0 = embeddings[lang_id_to_str[lang_0]][indexes_tokens[lang_id_to_str[lang_0]]][matches_indexes[:, 0]]
                features_1 = embeddings[lang_id_to_str[lang_1]][indexes_tokens[lang_id_to_str[lang_1]]][matches_indexes[:, 1]]
            else:
                features_0 = embeddings[indexes_tokens[lang_id_to_str[lang_0]][matches_indexes[:, 0]]]
                features_1 = embeddings[indexes_tokens[lang_id_to_str[lang_1]][matches_indexes[:, 1]]]
            c = features_1.transpose(1, 0).mm(features_0)
            correlation_matrices[(lang_0, lang_1)] = c.cpu()

    def get_corr_mat(idx_1, idx_2):
        assert idx_1 != idx_2
        if (idx_1, idx_2) in correlation_matrices:
            return correlation_matrices[(idx_1, idx_2)]
        elif (idx_2, idx_1) in correlation_matrices:
            return correlation_matrices[(idx_2, idx_1)].transpose(1, 0)
        else:
            print('Probably one of the indices is larger than the number of languages')
            return None

    feature_dim = embeddings[list(embeddings.keys())[0]].shape[1] if sigurdsson else embeddings.shape[1]

    # run several runs and choose best, where every run starts with a random matrix (one run starts with identity)
    t_0 = torch.eye(feature_dim)
    T = [t_0]
    for i in range(1, len(list_tokens)):
        M = torch.zeros((feature_dim, feature_dim))
        for j in range(0, i):
            M += torch.matmul(T[j].float(), get_corr_mat(i, j).float())
        # U, S, V_t = scipy.linalg.svd(M.cpu().numpy(), full_matrices=True)
        U, S, V = torch.svd(M)
        V_t = V.transpose(0, 1)
        t_i = torch.matmul(U, V_t)
        T.append(t_i)

    n_iter = 20
    with tqdm(total=n_iter * len(list_tokens)) as pbar:
        for _ in range(n_iter):  # technically, while not converged.
            for i in range(0, len(list_tokens)):
                M = torch.zeros((feature_dim, feature_dim))
                for j in range(0, len(list_tokens)):
                    if j == i:
                        continue
                    M += torch.matmul(T[j], get_corr_mat(i, j))
                # U, S, V_t = scipy.linalg.svd(M.cpu().numpy(), full_matrices=True)
                U, S, V = torch.svd(M)
                V_t = V.transpose(0, 1)
                t_i = torch.matmul(U, V_t)
                T[i] = t_i

                pbar.update(1)

    T = {lang_id_to_str[i]: v for i, v in enumerate(T)}

    return T


def align_words():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument('--name_model', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--results_path', type=str, default='', help='Results path')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='Checkpoint path')
    parser.add_argument('--dataset_info_path', type=str,
                        help='Directory with computed info about dataset and tokenizer')
    parser.add_argument('--model_type', type=str, default='globetrotter', help='Model type')
    parser.add_argument('--procrustes', action='store_true', help='Use Procrustes')
    parser.add_argument('--not_sametoken_synonyms', action='store_true', help='Not use same token as synonym')
    parser.add_argument('--epoch', type=int, default=-1, help='Action to load')
    parser.add_argument('--tokenizer_type', type=str, default='huggingface', choices=['fairseq', 'huggingface'],
                        help='Which tokenizer to use (see tokenization.py)')
    parser.add_argument('--config_data', default='all_lang_test-zh-en', help='Languages to work with')

    args = parser.parse_args()

    extra_name = ""
    if args.procrustes:
        extra_name = "_procrustes"
    token_synonyms = ""
    if args.not_sametoken_synonyms:
        token_synonyms = "_notsametokensynonyms"
    os.makedirs(os.path.join(args.results_path, args.name_model), exist_ok=True)
    path_save = os.path.join(args.results_path, args.name_model, f'score_matrix{extra_name}{token_synonyms}.pth')

    counter_tok = None
    if args.procrustes:
        tokenizer = tokenization.create_tokenizer(tokenizer_type, dataset_info_path=dataset_info_path,
                                                  config_data=config_data)
        counter_tok = torch.load(os.path.join(dataset_info_path,
                                              f'common_tokens_{config_data}_hug30000.pth'), map_location="cpu")
        # keys are indexes of the token instead of the string token
        counter_tok = {tokenizer.token_to_id(k): v for k, v in counter_tok.items()}

    if not os.path.isfile(path_save):
        name_checkpoint = 'checkpoint_best.pth' if args.epoch == -1 else f'checkpoint_{args.epoch}.pth'
        model_load = torch.load(os.path.join(args.checkpoint_dir, args.name_model, name_checkpoint))
        path_load = os.path.join(dataset_info_path, 'gt_word_translations_alllangs.pth')
        dict_synonyms_good, dict_lens, list_tokens = torch.load(path_load)

        if args.not_sametoken_synonyms:
            dict_synonyms_good = {k: [pair for pair in v if pair[0] != pair[1]] for k, v in dict_synonyms_good.items()}

        print('Computing distances all vs all')
        if args.model_type == 'globetrotter':
            print(args.model_type)
            if args.model_type == 'globetrotter':
                embeddings = model_load['model']['text_embeddings.word_embeddings.weight'].cpu()
            else:
                embeddings = model_load['encoder']['module.embeddings.weight'].cpu().float()
            if args.procrustes:
                print('Compute procrustes')
                projection_matrices_dict = _word_procrustes(embeddings, list_tokens, counter_tok)
                print('Start Computing distances all vs all')
                my_array = [(
                    pair,
                    embeddings.shape[0],
                    torch.matmul(projection_matrices_dict[pair[0]],
                                 embeddings[list_tokens[pair[0]]].transpose(1, 0)).transpose(1, 0),
                    torch.matmul(projection_matrices_dict[pair[1]],
                                 embeddings[list_tokens[pair[1]]].transpose(1, 0)).transpose(1, 0),
                    dict_synonyms_good[pair],
                    list_tokens[pair[0]], list_tokens[pair[1]], False)
                    for pair, synonyms in tqdm(dict_synonyms_good.items())]
            else:
                my_array = [(
                    pair,
                    embeddings.shape[0],
                    embeddings[list_tokens[pair[0]]],
                    embeddings[list_tokens[pair[1]]],
                    dict_synonyms_good[pair],
                    list_tokens[pair[0]], list_tokens[pair[1]], False)
                    for pair, synonyms in tqdm(dict_synonyms_good.items())]
        else:  # sigurdsson
            # Note that the list of tokens used in the ground truth comes from the "same_lang_all" dataset, which is the
            # test set (we use it because it is the ground truth). However, the tokens to create the word2vec come from
            # the training set. So the list of tokens that are included in each language are different. Here we use
            # the tokens that are in the word2vec.
            path_word2vec = os.path.join(args.dataset_info_path,
                                         f'word2vec_weights_{args.config_data}_traininglang.pth')
            word2vec = torch.load(path_word2vec)

            emb_total_size = model_load['model']['word_embeddings.lang_0.embedding_translate.weight'].shape[0]

            embeddings_lang = {}
            for lang in model_load["list_lang"]:
                adapt_layer = model_load['model'][f'adapt_layers.lang_{model_load["list_lang"][lang]}.linear.weight']
                word_embeddings = model_load['model'][f'word_embeddings.lang_{model_load["list_lang"][lang]}.embedding.weight'][:-1].transpose(1, 0)
                prod = torch.mm(adapt_layer, word_embeddings)
                total = prod.transpose(1, 0) + model_load['model'][f'adapt_layers.lang_{model_load["list_lang"][lang]}.linear.bias']
                embeddings_lang[lang] = total.cpu()
            if args.procrustes:
                list_tokens_sigurdsson = {lang: np.where(word2vec[lang].sum(-1) != 0)[0] for lang in model_load["list_lang"]}
                projection_matrices_dict = word_procrustes(embeddings_lang, list_tokens_sigurdsson, counter_tok, sigurdsson=True)
                my_array = [(
                    pair,
                    emb_total_size,
                    torch.matmul(projection_matrices_dict[pair[0]],
                                 embeddings_lang[pair[0]].transpose(1, 0)).transpose(1, 0),
                    torch.matmul(projection_matrices_dict[pair[1]],
                                 embeddings_lang[pair[1]].transpose(1, 0)).transpose(1, 0),
                    dict_synonyms_good[pair],
                    np.where(word2vec[pair[0]].sum(-1) != 0)[0],
                    np.where(word2vec[pair[1]].sum(-1) != 0)[0],
                    True)
                    for pair, synonyms in dict_synonyms_good.items()
                if pair[0] in model_load["list_lang"] and pair[1] in model_load["list_lang"]]
            else:
                my_array = [(
                    pair,
                    emb_total_size,
                    embeddings_lang[pair[0]],
                    embeddings_lang[pair[1]],
                    dict_synonyms_good[pair],
                    np.where(word2vec[pair[0]].sum(-1) != 0)[0],
                    np.where(word2vec[pair[1]].sum(-1) != 0)[0],
                    True)
                    for pair, synonyms in dict_synonyms_good.items()
                    if pair[0] in model_load["list_lang"] and pair[1] in model_load["list_lang"]]

        results = []
        for info in tqdm(my_array):
            results.append(_align_words_subrutine(info))

        print('Results computed')
        score_matrix = np.zeros([len(dict_lens), len(dict_lens)])
        score_matrix_chance = np.zeros([len(dict_lens), len(dict_lens)])
        lang_to_id = {lang: i for i, lang in enumerate(dict_lens.keys())}

        for pair, recall_0_to_1, recall_1_to_0, chance in tqdm(results):
            score_matrix[lang_to_id[pair[0]], lang_to_id[pair[1]]] = recall_0_to_1
            score_matrix[lang_to_id[pair[1]], lang_to_id[pair[0]]] = recall_1_to_0
            score_matrix_chance[lang_to_id[pair[0]], lang_to_id[pair[1]]] = chance[0]
            score_matrix_chance[lang_to_id[pair[1]], lang_to_id[pair[0]]] = chance[1]

        torch.save([lang_to_id, score_matrix, score_matrix_chance], path_save)

    else:
        lang_to_id, score_matrix, score_matrix_chance = torch.load(path_save)

    print(score_matrix.mean())
    print(score_matrix_chance.mean())

    with open(f'config/data/{self.config_data}.json') as json_file:
        lang_split = json.load(json_file)
    lang_dict = {k: v for k, v in lang_to_id.items() if k in lang_split['training']}
    score_matrix = score_matrix[:50, :50]
    score_matrix_chance = score_matrix_chance[:50, :50]
    score_matrix_chance[range(score_matrix_chance.shape[0]), range(score_matrix_chance.shape[0])] = 1
    scmat = score_matrix
    x = scmat.mean(1)[:, None]
    y = scmat.mean(0)[None, :]
    scmat = scmat / y ** 1.5
    scmat = scmat / x ** 1.5
    scmat[range(scmat.shape[0]), range(scmat.shape[0])] = scmat[score_matrix != 0].min()

    list_family = [element for element in list_family if element in lang_dict]
    list_family = [lang_dict[element] for element in list_family]

     # Sort by family
    score_matrix_index = np.array(list_family)

    score_matrix = scmat[score_matrix_index][:, score_matrix_index]
    lang_dict2 = {k: np.where(score_matrix_index == v)[0][0] for k, v in lang_dict.items()}
    lang_dict3 = {v: k for k, v in lang_dict2.items()}
    lang_dict2 = {lang_dict3[v]: v for v in range(len(lang_dict3))}

    score_matrix[range(score_matrix.shape[0]), range(score_matrix.shape[0])] = score_matrix[
    score_matrix != 0].min()

    fig = plt.figure(figsize=(7, 7), dpi=400)
    spec = gridspec.GridSpec(ncols=1, nrows=1, width_ratios=[1])

    ax0 = fig.add_subplot(spec[0])
    ax0.matshow(score_matrix, cmap=plt.get_cmap('winter'))
    ax0.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax0.set_xticks(np.arange(len(lang_dict2)))
    ax0.set_yticks(np.arange(len(lang_dict2)))
    ax0.set_xticklabels([dict_langs[k] for k in list(lang_dict2.keys())], fontsize=9, rotation=90)
    ax0.set_yticklabels([dict_langs[k] for k in list(lang_dict2.keys())], fontsize=9)
    ax0.set_ylabel('Query language', fontsize=12)
    ax0.set_xlabel('Target language', fontsize=12)

    fig.tight_layout()
    fig.show()

    plt.savefig(os.path.join(args.results_path, args.name_model, 'matrix.pdf'), format='pdf')
    print(f'Saved in {os.path.join(args.results_path, args.name_model, "matrix.pdf")}')


def _align_words_presubrutine(subarray):
    results = []
    for info in tqdm(subarray):
        results.append(_align_words_subrutine(info))
    return results


def _align_words_subrutine(*args):
    pair, emb_len, embeddings_lang_0, embeddings_lang_1, dict_synonyms_good_pair, list_tokens_0, \
        list_tokens_1, sigurdsson = args[0]

    embeddings_lang_0 = embeddings_lang_0.cpu().numpy()
    embeddings_lang_1 = embeddings_lang_1.cpu().numpy()
    embeddings_lang_0 = embeddings_lang_0/np.sqrt((embeddings_lang_0**2).sum(-1))[:, None]
    embeddings_lang_1 = embeddings_lang_1/np.sqrt((embeddings_lang_1**2).sum(-1))[:, None]

    similarity = np.matmul(embeddings_lang_0, embeddings_lang_1.transpose(1, 0))

    ground_truth = np.zeros((emb_len, emb_len))
    for a, b in dict_synonyms_good_pair:
        ground_truth[a, b] = 1
    ground_truth = ground_truth[np.array(list_tokens_0)[:, None], np.array(list_tokens_1)]

    sort_from_0_to_1 = (-similarity).argsort(1)  # for each token in lang 0, sort the tokens in lang 1
    sort_from_1_to_0 = (-similarity).argsort(0).transpose(1, 0)

    ground_truth_0_to_1 = \
        ground_truth[np.tile(np.array(range(ground_truth.shape[0])), (ground_truth.shape[1], 1)).transpose(),
                     sort_from_0_to_1]

    ground_truth_1_to_0 = \
        ground_truth.transpose(1, 0)[np.tile(np.array(range(ground_truth.shape[1])), (ground_truth.shape[0], 1)).
                                         transpose(), sort_from_1_to_0]

    # An alternative is to only count one of the several possible translations (any that is in the top k), but then
    #  divide by the number of words that have translation, not the total number of synonyms.
    top = 10
    recall_0_to_1 = ground_truth_0_to_1[:, :top].sum()/ground_truth_0_to_1.sum()
    recall_1_to_0 = ground_truth_1_to_0[:, :top].sum()/ground_truth_0_to_1.sum()
    chance = (float(top)/ground_truth_0_to_1.shape[1], float(top)/ground_truth_1_to_0.shape[1])

    return pair, recall_0_to_1, recall_1_to_0, chance


class Tester:
    def __init__(self, trainer):
        self.trainer = trainer

    def extract_features(self):
        """
        Extract features for all samples, in order to match them later on, or to use them to finetune other languages
        """
        if self.trainer.args.device == "cuda":
            torch.cuda.synchronize()
        self.trainer.model.eval()
        txt_to_txt_embs = []
        txt_to_img_embs = []
        img_to_txt_embs = []
        img_to_img_embs = []
        average_count_tokens = []
        idxs = []
        text_lens = []
        info = []

        split = self.trainer.args.test_options
        assert split in ['train', 'val', 'test'], 'Remember to add the split in args.test_options'

        with tqdm(self.trainer.loaders[split], desc=f'Extracting features',
                  disable=self.trainer.args.local_rank > 0) as t:
            for batch_idx, data in enumerate(t):
                # If we only want to extract a few features
                # if batch_idx > 25000:
                #     break
                # -------------- Organize inputs ------------- #
                text = data['text'].to(self.trainer.args.device)
                text = text.squeeze(1)
                language = torch.tensor(data['language']).squeeze(1).to(self.trainer.args.device)

                images = None
                if not self.trainer.args.not_use_images:
                    images = data['imgs'].to(self.trainer.args.device).squeeze(1)

                with torch.no_grad():
                    outputs = self.trainer.model(text, images, language=language)
                if images is None:
                    _, txt_to_txt_emb = outputs
                    txt_to_img_emb, img_to_txt_emb, img_to_img_emb = None, None, None
                else:
                    _, txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb = outputs

                if txt_to_txt_emb is not None:  # sigurdsson case
                    txt_to_txt_embs.append(txt_to_txt_emb.cpu().numpy())
                    if img_to_img_emb is not None:
                        img_to_img_embs.append(img_to_img_emb.cpu().numpy())

                if img_to_txt_emb is not None:
                    txt_to_img_embs.append(txt_to_img_emb.cpu().numpy())
                if img_to_txt_emb is not None:
                    img_to_txt_embs.append(img_to_txt_emb.cpu().numpy())
                text_lens.append(np.array(data['text_len'])[:, 0])
                for i in range(text.shape[0]):
                    # Here we use the unpickled sample_dict. Should not be a problem, but watch out for RAM
                    # If it gives problems, create the info_i vector from the sample_dict_X vectors
                    info_i = self.trainer.loaders[split].dataset.sample_dict[data['idxs'][i]]
                    info.append(info_i)
                    tokens = data['text_tokens'][i][0].split('/')[:data['text_len'][i][0]]
                    count_tokens = [0 for token in tokens]  # [counter[token] for token in tokens]
                    average_count_tokens.append(np.array(count_tokens).mean())
                    idxs.append(data['idxs'][i][0])

        print('Concatenating features')
        txt_to_txt_embs = np.concatenate(txt_to_txt_embs) if len(txt_to_txt_embs) > 0 else np.array([])
        img_to_img_embs = np.concatenate(img_to_img_embs) if len(img_to_img_embs) > 0 else np.array([])
        txt_to_img_embs = np.concatenate(txt_to_img_embs) if len(txt_to_img_embs) > 0 else np.array([])
        img_to_txt_embs = np.concatenate(img_to_txt_embs) if len(img_to_txt_embs) > 0 else np.array([])
        text_lens = np.concatenate(text_lens)
        average_count_tokens = np.array(average_count_tokens)
        epoch = ''
        if self.trainer.args.resume_epoch > -1:
            epoch = '_' + str(self.trainer.args.resume_epoch)
        path_name = os.path.join(self.trainer.args.results_path,
                                 f'extracted_features_{self.trainer.args.config_data}_'
                                 f'{self.trainer.args.language_split}lang_{split}subsplit{epoch}.pth')
        print(f'Saving features in {path_name} ...', end=' ')
        torch.save([txt_to_txt_embs, txt_to_img_embs, img_to_txt_embs, img_to_img_embs, info, text_lens,
                         average_count_tokens, idxs], path_name)
        print('Done.')


if __name__ == '__main__':
    main()
