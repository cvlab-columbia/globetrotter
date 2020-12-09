import os
import typing

import torch
import transformers.modeling_bert as mb
from torch import nn
from torchvision.models import resnet18
from transformers import *
import utils
import time
import torch.utils.checkpoint as checkpoint
import numpy as np


class BertVMPredictionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transform = mb.BertPredictionHeadTransform(cfg)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.contrastive_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        # Normalize
        hidden_states = hidden_states / (hidden_states ** 2).sum(-1, keepdim=True).sqrt()

        return hidden_states

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)


class TextEmbeddings(mb.BertEmbeddings):
    def __init__(self, cfg, tok):
        super().__init__(cfg)
        self.cfg = cfg
        self.tokenizer = tok

    def get_token_embedding(self, token, device, bs):
        embedding_token = self.word_embeddings(torch.tensor(self.tokenizer.index_special_tokens[token]).to(device))
        txt_type_embedding = self.token_type_embeddings(torch.tensor(2).to(device))[(None,) * 2]. \
            expand((bs, 1, -1))
        embedding_token = embedding_token + txt_type_embedding
        return embedding_token

    # noinspection PyMethodOverriding
    def forward(self, text_input_ids, img_input=None):
        # Optionally an ID for each language could be added

        assert len(text_input_ids.shape) == 2

        device = text_input_ids.device

        # They are not really word-level embeddings, but we use the BERT code, so they are called word embeddings
        text_word_embeddings = self.word_embeddings(text_input_ids)
        text_type_embeddings = self.token_type_embeddings(torch.ones_like(text_input_ids))
        text_embeddings = text_word_embeddings + text_type_embeddings

        # Add <txt> token at the beginning
        # We label the tokens <img> and <txt> with type embedding 2
        embedding_txt_token = self.get_token_embedding('<txt>', device, text_input_ids.shape[0])

        seq_length = text_input_ids.shape[-1] + 1  # Duration of text, and token <txt>
        embeddings = torch.cat((embedding_txt_token, text_embeddings), dim=1)

        # Add the position embeddings, that start counting from 0 to seq_lenght-1 (the 0 corresponds to <txt>, etc.)
        pos_embeddings = self.position_embeddings(torch.arange(seq_length, dtype=torch.long, device=device)). \
            unsqueeze(0).expand_as(embeddings)
        embeddings = embeddings + pos_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, text_word_embeddings

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)


class GlobetrotterModel(BertPreTrainedModel):
    def __init__(self, path, fn_cfg='config', tok=None, pretrained_cnn=False, momentum_bn=0.1,
                 transductive_bn=False, output_attentions=False, freeze_image=False):
        cfg = BertConfig.from_json_file(os.path.join(path, f'{fn_cfg}.json'))
        if output_attentions:
            cfg.output_attentions = True
        super().__init__(cfg)

        # Text
        self.tokenizer = tok
        self.text_embeddings = TextEmbeddings(cfg, tok)
        self.text_encoder = mb.BertEncoder(cfg)
        self.pooler = mb.BertPooler(cfg)
        self.text_prediction = mb.BertLMPredictionHead(cfg)

        # Image
        self.img_embedder = resnet18(pretrained=pretrained_cnn)
        self.img_embedder.fc = nn.Linear(512, cfg.hidden_size)

        # Heads
        self.txt_to_txt_head = BertVMPredictionHead(cfg)
        self.txt_to_img_head = BertVMPredictionHead(cfg)
        self.img_to_txt_head = BertVMPredictionHead(cfg)
        self.img_to_img_head = BertVMPredictionHead(cfg)

        self.cfg = cfg
        try:
            self.apply(self.init_weights)
        except:
            self.init_weights()
        self.tie_weights()

        for bn_layer in utils.return_all_leaf_children(self, 'BatchNorm'):
            if transductive_bn:
                bn_layer.track_running_stats = False
                # This is not actually transductive-related, but it is related to instability of batchnorm
                bn_layer.weight.requires_grad = False
                bn_layer.bias.requires_grad = False
            else:
                bn_layer.momentum = momentum_bn

        if freeze_image:
            for subnetwork in [self.img_embedder, self.img_to_txt_head, self.img_to_img_head]:
                for param in subnetwork.parameters():
                    param.requires_grad = False

    def tie_weights(self):
        self._tie_or_clone_weights(self.text_prediction.decoder, self.text_embeddings.word_embeddings)

    def forward(self, text_input_ids, img_input=None, language=None, return_hidden=False, not_mean=False):

        # ---------- Process text ----------- #
        word_position_embeddings, word_embeddings = self.text_embeddings(text_input_ids, img_input)
        txt_embeddings = self.text_encoder(word_position_embeddings, head_mask=[None] * self.cfg.num_hidden_layers)
        hidden_states = txt_embeddings[0]

        # Separate output into output for text and separator token
        txt_token_output = hidden_states[:, :1]
        text_seq_output = hidden_states[:, 1:1 + text_input_ids.shape[1]]

        text_predictions = self.text_prediction(text_seq_output)
        all_text_output = utils.indexed_mean(text_seq_output, (text_input_ids != 0).sum(1), dim=1, keepdim=True)

        if not_mean:
            all_text_output = txt_token_output = text_seq_output

        txt_to_txt_emb = self.txt_to_txt_head(all_text_output)

        # ---------- Process images ---------- #
        if img_input is None:  # no images
            img_to_img_emb = img_to_txt_emb = txt_to_img_emb = None
        else:
            img_embeddings = self.img_embedder(img_input.view([-1] + list(img_input.shape[2:])))
            img_to_img_emb = self.img_to_img_head(img_embeddings)
            img_to_img_emb = img_to_img_emb.view(img_input.shape[0], img_input.shape[1], img_to_img_emb.shape[1])
            img_to_txt_emb = self.img_to_img_head(img_embeddings)
            img_to_txt_emb = img_to_txt_emb.view(img_input.shape[0], img_input.shape[1], img_to_txt_emb.shape[1])
            txt_to_img_emb = self.txt_to_img_head(txt_token_output)

        if img_input is None:
            outputs = text_predictions, txt_to_txt_emb
        else:
            outputs = text_predictions, txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb

        if return_hidden:
            outputs += (hidden_states, )

        return outputs

    def named_parameters(self, prefix='', recurse=True, return_no_grad=True):
        """
        Overwrite to only return parameters that require gradient.
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if elem[1].requires_grad or return_no_grad:
                yield elem

    def parameters(self, recurse=True, return_no_grad=True):
        for name, param in self.named_parameters(recurse=recurse, return_no_grad=return_no_grad):
            yield param

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def get_type_device(self):
        return self.device.type


class AdaptLayer(nn.Module):
    """
    Just a linear layer. The orthogonality constraint is enforced in the loss
    """

    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.sigurdsson_hid_size, cfg.sigurdsson_hid_size)

    def forward(self, x):
        x = self.linear(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # "hidden_size" parameters not specified in the sigurdsson paper
        self.linear1 = nn.Linear(cfg.sigurdsson_hid_size, cfg.hidden_size)
        self.linear2 = nn.Linear(cfg.hidden_size, cfg.contrastive_size)
        self.relu = nn.ReLU()

    def forward(self, x, pool=True):
        x_flat = x.view(-1, x.shape[-1])
        x_flat = self.relu(self.linear1(x_flat))
        x = x_flat.view(x.shape[0], x.shape[1], -1)
        if pool:
            x = x.max(1)[0]  # maxpool over the words
        x = self.linear2(x)
        # Normalize
        x = x / (x ** 2).sum(-1, keepdim=True).sqrt()

        return x

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)


class CustomEmbedding(nn.Module):
    """
    All the languages share the same vocabulary, but most tokens never appear in most of them, so having weights
    representing them (weights that will never be used), as well as optimizer statistics for those weights takes too
    much space. So we do an intelligent forward pass based on the pretrained tokens. Those with pretrained features
    are the only ones that are going to be used. We can only do that if we previously know which tokens are used for
    each language.
    """

    def __init__(self, vocab_size, hid_size, pretrained_weights):
        super().__init__()
        indices_lang = pretrained_weights.mean(1) != 0
        vocab_size_lang = indices_lang.sum()
        # The +1 is where new tokens that appear (and are not in training) will go to
        self.embedding_translate = nn.Embedding(vocab_size + 1, 1, padding_idx=0)
        self.embedding_translate.weight.requires_grad = False
        self.embedding_translate.weight.data[:indices_lang.shape[0]][indices_lang, 0] = \
            torch.tensor(range(vocab_size_lang)).float()
        self.embedding_translate.weight.data[:indices_lang.shape[0]][~indices_lang] = vocab_size_lang.float()

        # The last token represents all the new tokens that are not in the word2vec (new words)
        self.embedding = nn.Embedding(vocab_size_lang + 1, hid_size, padding_idx=0)
        self.embedding.weight.data[:-1] = pretrained_weights[indices_lang]

    def forward(self, x):
        x = self.embedding_translate(x).squeeze(-1).long()
        x = self.embedding(x)
        return x


class SigurdssonModel(nn.Module):
    def __init__(self, path, list_lang, fn_cfg='config', pretrained_cnn=False, freeze_image=False, pretrain_path=None):
        cfg = BertConfig.from_json_file(os.path.join(path, f'{fn_cfg}.json'))
        super().__init__()

        self.num_lang = len(list_lang)
        self.config = cfg

        # Text. 300 is the dimensionality used for word embeddings in the Sigurdsson paper
        self.word_embeddings = nn.ModuleDict(
            {f'lang_{i}': nn.Embedding(cfg.vocab_size, cfg.sigurdsson_hid_size, padding_idx=0)
             for i in range(self.num_lang)}
        )
        if pretrain_path is not None:
            assert os.path.isfile(pretrain_path), 'Please precompute word2vec weights to pretrain the embedding'
            pretrained_weights = torch.load(pretrain_path)
            # The order of the languages will be the same because it is the same as in the config .json file
            for lang, idx in list_lang.items():
                if lang in pretrained_weights:
                    self.word_embeddings[f'lang_{idx}'] = \
                        CustomEmbedding(cfg.vocab_size, cfg.sigurdsson_hid_size, torch.tensor(pretrained_weights[lang]))

        self.adapt_layers = nn.ModuleDict(
            {f'lang_{i}': AdaptLayer(cfg) for i in range(self.num_lang)}
        )
        self.text_encoder = TextEncoder(cfg)

        # Image
        self.img_embedder = resnet18(pretrained=pretrained_cnn)
        self.img_embedder.fc = nn.Linear(512, cfg.hidden_size)

        # Heads
        # Text to image head is part of the text encoder
        self.img_to_txt_head = BertVMPredictionHead(cfg)  # Following our model because we use the same visual network

        if freeze_image:
            for subnetwork in [self.img_embedder, self.img_to_txt_head]:
                for param in subnetwork.parameters():
                    param.requires_grad = False

    def forward(self, text_input_ids, img_input=None, language=None, pool=True, return_hidden=False):
        # ---------- Process text ----------- #
        h_txt = torch.zeros((text_input_ids.shape[0], text_input_ids.shape[1], self.config.sigurdsson_hid_size)). \
            to(text_input_ids.device)
        for lang in set([lang.item() for lang in language]):
            indices = torch.where(torch.tensor(language == lang))
            text_input_ids_lang = text_input_ids[indices]
            h_lang = self.word_embeddings[f'lang_{lang}'](text_input_ids_lang)
            h_lang = self.adapt_layers[f'lang_{lang}'](h_lang)
            h_txt[indices] = h_lang.float()  # they may be Half

        txt_to_img_emb = self.text_encoder(h_txt, pool=pool)
        # ---------- Process images ---------- #
        if img_input is None:  # no images
            img_to_txt_emb = None
        else:
            img_embeddings = self.img_embedder(img_input.view([-1] + list(img_input.shape[2:])))
            img_to_txt_emb = self.img_to_txt_head(img_embeddings)
            img_to_txt_emb = img_to_txt_emb.view(img_input.shape[0], img_input.shape[1], img_to_txt_emb.shape[1])

        img_to_img_emb = text_predictions = txt_to_txt_emb = hidden_states = None

        outputs = text_predictions, txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb

        if return_hidden:
            outputs += (hidden_states, )

        return outputs

    def get_adapt_layer_weights(self):
        return [layer.linear.weight for name, layer in self.adapt_layers.items()]

    def named_parameters(self, prefix='', recurse=True, return_no_grad=True):
        """
        Overwrite to only return parameters that require gradient.
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if elem[1].requires_grad or return_no_grad:
                yield elem

    def parameters(self, recurse=True, return_no_grad=True):
        for name, param in self.named_parameters(recurse=recurse, return_no_grad=return_no_grad):
            yield param

    # This is simply for PyCharm to find the correct reference to the methods of the class
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def get_type_device(self):
        return self.text_encoder.linear1.weight.device.type
