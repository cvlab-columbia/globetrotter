import argparse
from datetime import datetime
import os
import random
import socket

import numpy as np
import torch
from pytorch_transformers import AdamW
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import dataset
import models
import tokenization
import utils
from masker import Masker
from trainer import Trainer
import json


def get_args():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on validation set')
    parser.add_argument('--test', action='store_true', help='Test model on test set')
    parser.add_argument('--test_name', type=str, default='', help='Test to be done')
    parser.add_argument('--test_options', type=str, default='', help='Test options depend on test')
    parser.add_argument('--sigurdsson', action='store_true', help='Use Deepmind\'s (Sigurdsson et al.) model')

    # Model
    parser.add_argument('--two_heads_modality', action='store_true',
                        help='The head that goes to the cross-modal loss is different than the modality-specific loss')
    parser.add_argument('--pretrained_cnn', action='store_true',
                        help="Use pretrained CNN to embed images. Overwritten by --resume")

    # Training
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of global training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the outer loop')

    # Paths
    parser.add_argument('--name', type=str, default='', help='Name of the training')
    parser.add_argument('--resume_name', type=str, default='', help='Name of the resumed training')
    parser.add_argument('--resume', action='store_true', help='Load best model from checkpoint')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the pretrained tokenizer. Only for fairseq')
    parser.add_argument('--dataset_info_path', type=str,
                        help='Directory with computed info about dataset and tokenizer')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='Checkpoints directory')
    parser.add_argument('--runs_dir', default='/path/to/your/runs', help='Path to tensorboard information')
    parser.add_argument('--results_dir', default='/path/to/your/results', help='Path to miscellaneous results')
    parser.add_argument('--config_arch', default='config', help='Transformers config file')
    parser.add_argument('--config_data', default='all_lang', help='Languages to work with')

    # Dataset parameters
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--max_txt_seq_len', type=int, default=50, help='Maximum text length')
    parser.add_argument('--tokenizer_type', type=str, default='fairseq', choices=['fairseq', 'huggingface'],
                        help='Which tokenizer to use (see tokenization.py)')
    parser.add_argument('--not_use_images', action='store_true', help='Do not input images to the model')
    parser.add_argument('--augment_image', action='store_true', help='Dataset returns two augmented images')
    parser.add_argument('--language_split', type=str, default='training', choices={'training', 'testing'},
                        help='Which languages to use, from the .json config file')

    # Optimization
    parser.add_argument('--lambda_visual_loss', type=float, default=1.0, help='Weight for the visual loss')
    parser.add_argument('--lambda_xlang_loss', type=float, default=1.0, help='Weight for the cross-language loss')
    parser.add_argument('--lambda_lm_loss', type=float, default=1.0, help='Weight for the cross-language loss')
    parser.add_argument('--lambda_xm_loss', type=float, default=1.0, help='Weight for the cross-language loss')
    parser.add_argument('--lambda_orthogonality_loss', type=float, default=1.0,
                        help='Weight for the orthogonality constraint for the AdaptLayer in Sigurdsson\'s method')
    parser.add_argument('--alpha_xm', action='store_true', help='Use xm similarity to create alphas')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the outer loop')
    parser.add_argument('--workers', type=int, default=4, help='Num parallel workers')
    parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit float precision instead of 32-bit')
    parser.add_argument('--opt_level', default='O1', help='Optimization level for fp16 training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training on gpus')
    parser.add_argument('--momentum_bn', type=float, default=0.1, help='Batch norm momentum')

    # Masking probabilities
    parser.add_argument('--prob_predict_token', type=float, default=1 / 6, help='prob_predict_token. See dataset')
    parser.add_argument('--p_mask', type=float, default=0.8, help='p_mask. See masker')
    parser.add_argument('--p_clobber_other_txt', type=float, default=0, help='p_clobber_other_txt. See masker')

    # Other
    parser.add_argument('--debug', action='store_true', help='Debug (no writing to disk at all)')
    parser.add_argument('--resume_latest', action='store_true', help='Resume latest checkpoint instead of best one')
    parser.add_argument('--output_attentions', action='store_true', help='Model output attention values')
    parser.add_argument('--resume_epoch', type=int, default=-1, help='Epoch to resume')
    parser.add_argument('--resume_but_restart',  action='store_true',
                        help='Load model from another checkpoint, but do not continue with that execution')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--print_freq', type=int, default=1, help='Print/save loss every print_freq batches')

    args = parser.parse_args()

    assert not (args.resume_latest and not args.resume), 'To resume the last model, "--resume" has to be also True'
    assert not (args.resume_latest and args.resume_epoch >= 0), 'Either resume latest, or resume specific epoch'

    assert not ((args.lambda_visual_loss > 0 or args.lambda_xlang_loss > 0) and args.batch_size == 1) or args.test, \
        'With batch size of 1, there will not be negatives for the visual or xlang loss'
    assert not ((args.lambda_visual_loss > 0 or args.lambda_xm_loss > 0) and args.not_use_images), 'Really?'
    assert not (args.lambda_visual_loss > 0 and not args.augment_image) or args.test, \
        'For the visual loss we need augmented images'
    assert not (args.alpha_xm and args.lambda_xm_loss <= 0), "The xm loss is needed to add alpha_xm to the xlang loss"

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not (args.resume and not args.resume_but_restart):
        assert args.name is not None and len(args.name) > 0
        args.name = args.name + '_' + current_time + '_' + socket.gethostname()
    else:
        assert args.resume_name is not None and len(args.resume_name) > 0
        args.name = args.resume_name

    if not args.test:
        if args.sigurdsson:
            assert not args.two_heads_modality and args.lambda_lm_loss == 0 and args.lambda_visual_loss == 0 \
                   and args.lambda_xlang_loss == 0
            assert args.p_clobber_other_txt == 0 and args.prob_predict_token == 0 and args.p_mask == 0
        else:
            assert args.lambda_orthogonality_loss == 0

    args.checkpoint_path = os.path.join(args.checkpoint_dir, args.name)
    args.checkpoint_path_load_from = os.path.join(args.checkpoint_dir, args.resume_name)
    args.results_path = os.path.join(args.results_dir, args.name)

    return args


def main():
    args = get_args()

    # Fix randomness
    seed = args.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not args.debug:
        os.makedirs(args.checkpoint_path, exist_ok=True)
        os.makedirs(args.results_path, exist_ok=True)

    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = args.step_n_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.step_n_gpus = torch.distributed.get_world_size()

    writer = SummaryWriter(log_dir=os.path.join(args.runs_dir, args.name) if not (
            args.debug or args.test or args.evaluate) else '/tmp') if args.local_rank <= 0 else None

    # ---------------------------- Prepare dataset ----------------------------- #
    if args.local_rank <= 0:
        print('Preparing dataset')
    # The "subsplits" (either ['train', 'val'] or ['test']) indicate how we are going to subsplit the language splits
    if args.test and not ('extract_features' in args.test_name and args.test_options != 'test'):
        subsplits = ['test']
    else:
        subsplits = ['train', 'val']

    dataset_dict = {}
    for subsplit in subsplits:
        # No need to randomize sigurdsson validation
        randomize = subsplit != 'test' and (subsplit != 'val' or (not args.test and not args.sigurdsson))
        dataset_dict[subsplit] = dataset.MultipleDatasets(
            dataset_path=args.dataset_path,
            img_size=args.image_size, subsplit=subsplit, max_txt_seq_len=args.max_txt_seq_len,
            dataset_info_path=args.dataset_info_path, config_data=args.config_data, augment_image=args.augment_image,
            prob_predict_token=args.prob_predict_token, language_split=args.language_split,
            not_use_images=args.not_use_images, randomize=randomize)
    list_txt_files = lambda: [txt_file for subsplit in subsplits for txt_file in dataset_dict[subsplit].list_txt_files]
    tokenizer = tokenization.create_tokenizer(args.tokenizer_type, dataset_info_path=args.dataset_info_path,
                                              config_data=args.config_data, list_txt_files=list_txt_files)
    for subsplit in subsplits:
        dataset_dict[subsplit].tokenizer = tokenizer

    def get_sampler(_subsplit):
        _sampler = None
        _shuffle = _subsplit == 'train'
        if args.local_rank != -1:
            _sampler = DistributedSampler(dataset_dict[_subsplit], shuffle=_shuffle)
        _shuffle = _shuffle and (_sampler is None)
        return _sampler, _shuffle

    loader_dict = {}
    for subsplit in subsplits:
        sampler, shuffle = get_sampler(subsplit)
        loader_dict[subsplit] = DataLoader(dataset_dict[subsplit], batch_size=batch_size, sampler=sampler,
                                           shuffle=shuffle, num_workers=args.workers, collate_fn=utils.collate_fn,
                                           pin_memory=True, drop_last=True)

    # ---------------------------- Prepare model ----------------------------- #
    if args.local_rank <= 0:
        print('Preparing model')
    config_from = args.checkpoint_path_load_from if args.resume else 'config/architecture'
    list_lang = None
    if args.sigurdsson:
        list_lang = dataset_dict[subsplits[0]].language_to_id

        config_name, lang_split = args.config_data, args.language_split
        with open(f'config/data/{args.config_data}.json') as json_file:
            word2vec_from = json.load(json_file)['word2vec_from']
            if word2vec_from != "":
                config_name, lang_split = word2vec_from
        pretrain_path = os.path.join(args.dataset_info_path, f'word2vec_weights_{config_name}_{lang_split}lang.pth')

        model = models.SigurdssonModel(config_from, list_lang, fn_cfg=args.config_arch, pretrained_cnn=args.pretrained_cnn,
                                     freeze_image=False, pretrain_path=pretrain_path).to(args.device)
    else:
        model = models.GlobetrotterModel(config_from, fn_cfg=args.config_arch, tok=tokenizer,
                                         pretrained_cnn=args.pretrained_cnn, momentum_bn=args.momentum_bn,
                                         output_attentions=args.output_attentions, freeze_image=False).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False, eps=1e-4)

    if args.resume:
        fn = f'checkpoint_{args.resume_epoch}.pth' if args.resume_epoch >= 0 else 'checkpoint.pth'
        load_best = not (args.resume_latest or args.resume_epoch >= 0)
        epoch, global_step = utils.load_checkpoint(model, optimizer if not args.test else None,
                                                   args.checkpoint_path_load_from, load_best=load_best, fn=fn,
                                                   strict=False, list_lang=list_lang)
        if args.resume_but_restart:
            epoch = global_step = -1
    else:
        epoch = global_step = -1

    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        args.parallel = 'ddp'
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        args.parallel = 'dp'
    else:
        args.parallel = 'none'

    # ---------------------------- Prepare trainer ----------------------------- #
    if args.local_rank <= 0:
        print('Preparing trainer')
    masker = Masker(tokenizer, p_mask=args.p_mask, p_clobber_other_txt=args.p_clobber_other_txt)
    trainer = Trainer(args, model, loader_dict, optimizer, epoch, global_step, writer, masker)

    # ---------------------------- Run ----------------------------- #
    if args.evaluate:
        trainer.run_epoch(epoch=None, train=False)
    elif args.test:
        trainer.test(args.test_name)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
