import time
from collections import defaultdict
import os.path

import torch
from torch.optim import SGD
from tqdm import tqdm
import numpy as np
import pickle
import torch.distributed as D
from torch.cuda.amp import GradScaler, autocast

import losses
import utils
import test


class Trainer:
    def __init__(self, args, model, loaders, optimizer, epoch, global_step, writer, masker):
        self.args = args
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.epoch_initial = epoch
        self.global_step = global_step
        self.writer = writer
        self.masker = masker
        self.scaler = GradScaler()

        self.list_losses = ['total']
        if args.lambda_visual_loss > 0 and not args.not_use_images:
            self.list_losses.append('vm')
        if args.lambda_xm_loss > 0 and not args.not_use_images:
            self.list_losses.append('xm')
        if args.lambda_xlang_loss > 0:
            self.list_losses.append('xl')
        if args.lambda_lm_loss > 0:
            self.list_losses.append('lm')
        if args.lambda_orthogonality_loss > 0:
            self.list_losses.append('orthogonal')

        self.precomp_feat = self.features_per_batch = self.indices_random = None

    def train(self):
        best_eval = 0
        try:
            for epoch in range(self.epoch_initial + 1, self.args.num_epochs):
                if self.args.local_rank != -1:
                    self.loaders['train'].sampler.set_epoch(epoch)
                self.run_epoch(epoch)

                # Evaluate on validation set
                eval_score = self.run_epoch(epoch, train=False)
                # Remember best eval score and save checkpoint
                is_best = eval_score > best_eval
                best_eval = max(eval_score, best_eval)
                if self.args.local_rank <= 0 and not self.args.debug:
                    print('Saving checkpoint')
                    list_lang = None
                    if self.args.sigurdsson:
                        # Needed to match languages to embedding indices
                        list_lang = self.loaders[list(self.loaders.keys())[0]].dataset.language_to_id
                    utils.save_checkpoint(self.model, self.optimizer, is_best, epoch, self.args.checkpoint_path,
                                          global_step=self.global_step, args=self.args, list_lang=list_lang,
                                          save_always=True)

        except KeyboardInterrupt:
            if self.args.local_rank <= 0:
                print(f'You decided to finish the training at epoch {epoch}')

    def run_epoch(self, epoch, train=True):
        if self.args.device == "cuda":
            torch.cuda.synchronize()
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Initialize meters
        avg_batch_time = utils.AverageMeter()
        avg_data_time = utils.AverageMeter()
        average_meters_losses = defaultdict(lambda: utils.AverageMeter())
        avg_top1 = utils.AverageMeter()
        avg_top5 = utils.AverageMeter()

        end = time.time()

        with tqdm(self.loaders['train' if train else 'val'], desc=f'Training epoch {epoch}' if train else
                  f'Evaluating {f"epoch {epoch}" if epoch else ""}', disable=self.args.local_rank > 0) as t:
            for batch_idx, data in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)
                # -------------- Organize inputs ------------- #

                text = data['text'].to(self.args.device)
                text = text.view(-1, text.shape[-1])

                text_len = torch.tensor(data['text_len']).view(-1)
                language = torch.tensor(data['language']).view(-1).to(self.args.device)
                pos_to_predict = [p for pos in data['pos_to_predict'] for p in pos]
                gt_tokens = data['gt_tokens'].view([-1] + list(data['gt_tokens'].shape[2:])).to(self.args.device)

                # Note that this does not mask special tokens
                text = self.masker.mask_text(text, pos_to_predict, text_len, randomize=train)

                images = None
                if not self.args.not_use_images:
                    images = data['imgs'].to(self.args.device)
                    images = images.view([-1] + list(images.shape[2:]))

                # ----------------- Forward model and compute losses ---------- #
                with autocast(enabled=self.args.fp16):
                    with torch.set_grad_enabled(train):
                        output = self.model(text, images, language)
                        if images is None:
                            lm_preds, txt_to_txt_emb = output
                            txt_to_img_emb = img_to_txt_emb = img_to_img_emb = None
                        else:
                            lm_preds, txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb = output
                        adapt_layer_weights = None
                        if self.args.sigurdsson:
                            adapt_layer_weights = self.get_base_model().get_adapt_layer_weights()
                        text_len = torch.tensor(text_len).clamp(max=self.args.max_txt_seq_len).to(self.args.device)

                        if img_to_img_emb is None and self.args.lambda_xlang_loss > 0:
                            img_to_img_emb = torch.tensor(data['idxs']).cuda().view(-1)  # for the supervised case
                        if self.args.parallel == 'ddp':
                            tensors_to_gather = [txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb, text_len]
                            for i, v in enumerate(tensors_to_gather):
                                tensors_to_gather[i] = gather_tensor(v)
                            txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb, text_len = tensors_to_gather

                        text_xm_indices = None

                        loss, loss_dict, accuracy = \
                            losses.compute_losses(lm_preds, gt_tokens, txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb,
                                                  img_to_img_emb, text_len, self.args.lambda_visual_loss,
                                                  self.args.lambda_xlang_loss, self.args.lambda_xm_loss,
                                                  self.args.lambda_lm_loss, self.args.lambda_orthogonality_loss,
                                                  self.args.alpha_xm, self.args.two_heads_modality, adapt_layer_weights,
                                                  language, device=self.args.device, compute_acc=True,
                                                  sigurdsson=self.args.sigurdsson, text_xm_indices=text_xm_indices)

                # --------------- Update model -------------- #
                if train:
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                # ------------- Show information ------------ #

                postfix_kwargs = {}

                if not train:
                    avg_top1.update(*accuracy['top1'])
                    avg_top5.update(*accuracy['top5'])

                for loss_name in self.list_losses:
                    average_meters_losses[loss_name].update(loss_dict[loss_name].item())
                    postfix_kwargs[loss_name] = average_meters_losses[loss_name].val

                t.set_postfix(
                    DataTime=avg_data_time.avg,
                    BatchTime=avg_batch_time.avg,
                    **postfix_kwargs
                )

                if train:
                    self.global_step += 1
                    if self.global_step % self.args.print_freq == 0 and self.writer and not self.args.debug:
                        num_outer_samples = (self.global_step + 1) * self.args.batch_size * \
                                            (1 if 'Parallel' in str(type(self.model)) else self.args.step_n_gpus)
                        self.writer.add_scalars('train/loss', {**postfix_kwargs}, num_outer_samples)

        if not train:
            cnt = average_meters_losses['total'].count
            # For losses and accuracies that are shared between GPUs, the results will be repeated, but it is ok
            values_to_gather = [average_meters_losses[loss_name].avg for loss_name in self.list_losses]
            values_to_gather += [avg_top1.avg, avg_top5.avg]
            labels_gathered = self.list_losses + ['recall_top1', 'recall_top5']
            values_gathered = utils.gather_score(values_to_gather, cnt)
            dict_gathered = {k: v for k, v in zip(labels_gathered, values_gathered)}

            loss_scalars = {}
            for loss_name in self.list_losses:
                loss_scalars[loss_name] = dict_gathered[loss_name]
            acc_scalars = {
                'recall_top1': dict_gathered['recall_top1'],
                'recall_top5': dict_gathered['recall_top5'],
            }
            if self.args.local_rank <= 0:
                print(f"Evaluation loss: {'; '.join([f'{k}: {v:.02f}' for k, v in loss_scalars.items()])}")
                print(f"Evaluation accuracy: {'; '.join([f'{k}: {v:.02f}' for k, v in acc_scalars.items()])}")
            if self.writer and not self.args.debug:
                self.writer.add_scalars('eval/loss', loss_scalars, epoch)
                self.writer.add_scalars('eval/acc', acc_scalars, epoch)

            return dict_gathered['recall_top5']

    def test(self, test_name):
        tester = test.Tester(trainer=self)
        if test_name == 'extract_features':
            tester.extract_features()
        else:
            print(f"Sorry but the test {test_name} is not implemented")

    def get_base_model(self):
        if 'DataParallel' in str(type(self.model)):  # both the ones from apex and from torch.nn
            return self.model.module
        else:
            return self.model


def gather_tensor(v):
    if v is None:
        return None

    gather_dest = [torch.empty_like(v) * i for i in range(D.get_world_size())]  # list where each element is [N x H_DIM]
    D.all_gather(gather_dest, v.contiguous())  # as far as i recall, this loses gradient information completely

    gather_dest[D.get_rank()] = v  # restore tensor with gradient information
    gather_dest = torch.cat(gather_dest)

    # gather_dest is now a tensor of [(N*N_GPUS) x H_DIM], as if you ran everything on one GPU, except only N samples
    # corresponding to GPU i inputs will have gradient information
    return gather_dest
