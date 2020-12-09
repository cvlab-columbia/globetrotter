"""
Two different tokenizations. The two of them BPE encodings.

First, code adapted from fairseq/models/roberta/hub_interface.py
Used to make it easier to reuse the dictionary from their xlm model.
This one contains a very big dictionary, so take care and reduce size of embeddings or model size explodes

Second, create our own dictionary just from the text from our datasets, using huggingface tokenization
"""

import os
import shutil
import typing
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from fairseq.data import encoders, Dictionary
from tokenizers import ByteLevelBPETokenizer, Tokenizer
import json

special_tokens = ['<pad>', '<txt>', '<img>', '<mask>', '<sep>']


class FairseqTokenizerBPE(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        self.dict = Dictionary.load(os.path.join(tokenizer_path, 'dict.txt'))
        # <sep> and <pad> already exist in the dictionary
        self.index_special_tokens = {tok: self.dict.add_symbol(tok) for tok in special_tokens}

        args = Namespace(
            bpe='sentencepiece',
            sample_break_mode='complete',
            sentencepiece_vocab=os.path.join(tokenizer_path, 'sentencepiece.bpe.model')
        )
        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))
        self.info = 'fairseq'

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str):
        """
        BPE-encode a sentence (or multiple sentences).

        We simplify the original code, and do not add sentence and sequence separators.

        The BPE encoding follows GPT-2.

        Note that special tokens like <pad>, <sep>, etc. have to be encoded separately. Otherwise self.bpe.encode does
        not understand them as atomic tokens
        """
        bpe_sentence = self.bpe.encode(sentence)
        tokens = self.dict.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long(), bpe_sentence.split(' ')

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.dict.bos():
            tokens = tokens[1:]  # remove <s>. We do not use it anyway
        eos_mask = (tokens == self.dict.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.dict.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def id_to_token(self, id):
        return NotImplemented

    def __len__(self):
        return len(self.dict)

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)


class HuggingfaceTokenizerBPE(nn.Module):
    def __init__(self, text_files, dataset_info_path='', config_data=None):
        super().__init__()
        # The default vocab size in the BERT model is 30522. If we want a number larger than that, we will also have to
        # change the BERT configuration.
        vocab_size = 30000
        self.info = f'hug{vocab_size}'

        with open(f'config/data/{config_data}.json') as json_file:
            tokenizer_from = json.load(json_file)['tokenizer_from']

        config_name = config_data if tokenizer_from == "" else tokenizer_from
        print(os.path.join(dataset_info_path, f'tokenizer_{config_name}_{vocab_size}-vocab.json'))

        # The loading is only properly implemented starting from version 0.8. However, it makes the system use a lot of
        #  CPU for no reason (it is much slower). Maybe it will be fixed in the future.
        if not os.path.isfile(os.path.join(dataset_info_path, f'tokenizer_{config_name}_{vocab_size}-vocab.json')):
            text_files = text_files()
            self.tokenizer = ByteLevelBPETokenizer()
            # Join into a single file. This should NOT be necessary but it does not work properly with a lot of files
            with open('/tmp/text_files.txt', 'wb') as outfile:
                for filename in tqdm(text_files, desc='Joining all files into one for tokenization'):
                    with open(filename, 'rb') as readfile:
                        shutil.copyfileobj(readfile, outfile)
                text_files = '/tmp/text_files.txt'
            self.tokenizer.train(text_files, vocab_size=vocab_size, special_tokens=special_tokens)
            self.tokenizer.save(dataset_info_path, f'tokenizer_{config_name}_{vocab_size}')

        # No "else", always load for consistency
        vocab_file = os.path.join(dataset_info_path, f'tokenizer_{config_name}_{vocab_size}-vocab.json')
        merges_file = os.path.join(dataset_info_path, f'tokenizer_{config_name}_{vocab_size}-merges.txt')
        self.tokenizer = ByteLevelBPETokenizer(vocab_file=vocab_file, merges_file=merges_file)
        self.tokenizer.add_special_tokens(special_tokens)

        self.index_special_tokens = {tok: self.tokenizer.encode(tok).ids[0] for tok in special_tokens}

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str):
        output = self.tokenizer.encode(sentence)
        token_ids = output.ids
        tokens = output.tokens
        return torch.tensor(token_ids), tokens

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = list(tokens.cpu().numpy())
        sentences = self.tokenizer.decode(tokens)
        return sentences

    def id_to_token(self, token_id):
        if type(token_id) != torch.Tensor:
            token_id = torch.tensor(token_id)
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token):
        assert type(token) == str
        return self.tokenizer.token_to_id(token)

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)


def create_tokenizer(tokenizer_type, list_txt_files=None, tokenizer_path=None, dataset_info_path=None,
                     config_data=None):
    if tokenizer_type == 'fairseq':
        tokenizer = FairseqTokenizerBPE(tokenizer_path=tokenizer_path)
    else:
        tokenizer = HuggingfaceTokenizerBPE(text_files=list_txt_files, dataset_info_path=dataset_info_path,
                                            config_data=config_data)
    return tokenizer
