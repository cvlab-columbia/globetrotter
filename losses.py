import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import utils


def compute_lm_loss(lm_preds, gt_tokens):
    # default reduction is 'mean'
    loss = F.cross_entropy(lm_preds.view(-1, lm_preds.shape[-1]), gt_tokens.view(-1), ignore_index=-1)
    return loss


def compute_contrastive_loss(vm_preds, img_embeddings, use_also_anchor=False, reverse=False):
    """
    Simple contrastive loss, where positive is the input embedding of an augmented version of the image, and negatives
    come from other samples across the batch (both the regular ones and the augmented ones -- the regular ones can also
    be augmented, but it is not the same augmentation because of randomization). If there is more than one image per
    example, these are considered like they come from different examples.
    """
    tau = 0.1  # Temperature
    vm_preds = vm_preds.view([-1] + list(vm_preds.shape[-1:]))
    img_embeddings_ = torch.cat([img_embeddings[:, 0], img_embeddings[:, 1]]) if img_embeddings.shape[1] == 2 \
        else img_embeddings[:, 0]  # in case of not augmentation. this will happen only for xmodal loss
    loss = torch.tensor(0).to(vm_preds.device)
    similarities_diag = None
    if vm_preds.shape[0] > 1:  # If at least more than one element in the batch
        similarities = torch.mm(vm_preds, img_embeddings_.transpose(0, 1))  # Note we use dot product
        similarities_diag = similarities.diag()
        if not use_also_anchor:
            similarities.fill_diagonal_(-1e4)  # Not interested in exactly the same image. We want its bis
        similarities = similarities / tau
        log_softmax = F.log_softmax(similarities, dim=1)
        if reverse:
            log_softmax = log_softmax/2 + F.log_softmax(similarities, dim=0)/2
        if img_embeddings.shape[1] == 2:
            if use_also_anchor:
                loss = - log_softmax[:, log_softmax.shape[0]:].diagonal().mean()/2 + \
                       - log_softmax[:, :log_softmax.shape[0]].diagonal().mean()/2
            else:
                loss = - log_softmax[:, log_softmax.shape[0]:].diagonal().mean()
        else:
            loss = - log_softmax.diagonal().mean()

    return loss, similarities_diag


def compute_xlang_loss(txt_to_txt_emb, img_to_img_emb, text_len, alpha_xm=False, xm_similarities=None):
    img_to_img_emb = img_to_img_emb.detach()
    xm_similarities = xm_similarities.detach() if xm_similarities is not None else None

    margin = 0.4
    tau = 0.1  # small temperature can result in precision errors and overflow.

    # txt_token_output = hidden_states[:, :1]
    max_txt_len = torch.min(text_len.max(), torch.tensor(txt_to_txt_emb.shape[1]).to(text_len.device))
    batch_size = txt_to_txt_emb.shape[0]
    # text_seq_output = hidden_states[:, 1:1 + max_txt_len]
    text_seq_output = txt_to_txt_emb[:, :max_txt_len]
    text_seq_output = text_seq_output.contiguous().view(-1, text_seq_output.shape[-1])
    betas = torch.matmul(text_seq_output, text_seq_output.transpose(0, 1))
    betas = betas.view(batch_size, max_txt_len, batch_size, max_txt_len)
    betas = utils.indexed_max(betas, text_len, dim=-1)
    betas = utils.indexed_mean(betas, text_len, dim=1)
    betas.fill_diagonal_(-1e4)

    if len(img_to_img_emb.shape) == 1:
        img_to_img_emb = img_to_img_emb.unsqueeze(1)
        alphas = (img_to_img_emb.expand(img_to_img_emb.shape[0], img_to_img_emb.shape[0]) -
                 img_to_img_emb.transpose(1, 0).expand(img_to_img_emb.shape[0], img_to_img_emb.shape[0])) == 0
        alphas.fill_diagonal_(False)
        alphas = alphas.float()
    else:
        img_to_img_emb = img_to_img_emb[:, 0]
        # The alphas weigh the loss.
        alphas = torch.matmul(img_to_img_emb.squeeze(1), img_to_img_emb.squeeze(1).transpose(0, 1))

        if alpha_xm:
            # Measures how close the image is to the text. If very close, then the signal makes more sense
            # It will need the xm loss to start converging before making sense
            # We rescale this value so that it is at most (100*x)% wrt to the actual alpha
            x = 0.25
            alpha_sim_xm = torch.mm(xm_similarities.unsqueeze(1), xm_similarities.unsqueeze(0))
            # rescaling
            alpha_sim_xm = alpha_sim_xm - alpha_sim_xm.min()
            alpha_sim_xm = alpha_sim_xm/alpha_sim_xm.max() * 2*x + (1-x)
            alphas *= alpha_sim_xm

        alphas.fill_diagonal_(-1e4)
        # We don't want to normalize image_outputs individually because in the visual loss functions we use the dot
        # product similarity and not the cosine similarity. So we just divide by maximum of the alphas to have a
        # controlled range
        alphas = alphas/alphas.max()
        # alphas could be negative, but this makes the system unstable, and there is no need to push in that direction
        alphas = torch.relu(alphas - margin) / (1-margin)

    score = torch.log_softmax(betas/tau, dim=1)
    score = score*alphas
    loss = -score.sum() / alphas.sum()

    return loss


def compute_sigurdsson_loss(txt_to_img_emb, img_to_txt_emb, languages):
    loss = 0
    for lang in set([lang.item() for lang in languages]):
        # If len(indices_lang) == 1, it's not a problem. The softmax will always be 1, so it's not informative.
        indices_lang = torch.where(torch.tensor(languages == lang))[0]
        txt_to_img_emb_lang = txt_to_img_emb[indices_lang]
        img_to_txt_emb_lang = img_to_txt_emb[indices_lang]
        similarities = torch.matmul(txt_to_img_emb_lang, img_to_txt_emb_lang.transpose(0, 1))
        tau = 0.1
        similarities = similarities / tau
        log_softmax = F.log_softmax(similarities, dim=1)/2 + F.log_softmax(similarities, dim=0)/2
        loss_lang = - log_softmax.diagonal().mean()
        loss += loss_lang
    return loss


def compute_orthogonality_loss(weights):
    """ This is the penalty used in the sigurdsson paper"""
    loss = 0
    for w in weights:
        identity = torch.eye(w.shape[0]).to(w.device)
        loss += (w*w.transpose(1, 0)-identity).pow(2).sum()
    return loss


def compute_accuracy(input_1, input_2, topk=(1,), dict_by_token=None, retrieval=False):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    :param retrieval: whether the accuracy is retrieval, or prediction
    :param input_2: first tensor to be compared. It can be either the lm_preds or the txt features
    :param input_1: second tensor to be compared. It can either be the gt text labels, or the img features
    :param topk: tuple of k to consider
    :param dict_by_token: results dict, used when we have separate gt tokens
    """
    assert not (dict_by_token is not None and retrieval), 'dict_by_token does not make sense in the retrieval case'

    maxk = max(topk)

    if not retrieval:
        # tensor of dim [batch_size, text_len, embedding_size]
        lm_preds = input_1
        # list of batch_size tensors, each one containing integers with the gt_tokens to predict in the
        # positions to be predicted, and -1 elsewhere
        gt_tokens = input_2
        prediction = lm_preds[gt_tokens != -1]
        gt = gt_tokens[(gt_tokens != -1)]
        dims = [1]

    else:
        # Note that these values will depend on the number of negatives! (ie batch size)
        prediction = torch.matmul(input_1, input_2.transpose(1, 0))
        gt = torch.tensor(range(input_1.shape[0]))
        dims = [0, 1]

    correct = []
    for dim in dims:
        _, pred = prediction.topk(maxk, dim, True, True)  # text to image
        if dim == 1:
            pred = pred.t()
        correct.append(pred.eq(gt.to(input_1.device)[None]))
    correct = torch.cat(correct, 1)

    res = {} if dict_by_token is None else dict_by_token
    for k in topk:
        if dict_by_token is not None:
            for i, gt_token in enumerate(gt):
                res[gt_token.cpu().item()][f'top{k}'].update(correct[:k, i].max().cpu().int().numpy())
        else:
            if input_1.sum() == 0:
                val = np.array([0, 0])
            else:
                correct_k = correct[:k].max(dim=0).values.sum(dtype=torch.float32)
                val = np.array([(correct_k * (100.0 / correct.shape[1])).item(), correct.shape[1]])
            res[f'top{k}'] = val

    return res


def compute_losses(lm_preds, gt_tokens, txt_to_txt_emb, txt_to_img_emb, img_to_txt_emb, img_to_img_emb, text_len,
                   lambda_visual_loss, lambda_xlang_loss, lambda_xm_loss, lambda_lm_loss, lambda_orthogonality_loss,
                   alpha_xm=False, two_heads_modality=False, adapt_layer_weights=None, languages=None, device="cpu",
                   compute_acc=False, sigurdsson=False, text_xm_indices=None):

    if not two_heads_modality and not sigurdsson:
        txt_to_img_emb = txt_to_txt_emb
        img_to_txt_emb = img_to_img_emb

    loss_total = torch.zeros(1).to(device)
    vm_loss = torch.zeros(1).to(device)
    xmodal_loss = torch.zeros(1).to(device)
    xlang_loss = torch.zeros(1).to(device)
    lm_loss = torch.zeros(1).to(device)
    orthogonality_loss = torch.zeros(1).to(device)

    # language model loss
    if lambda_lm_loss > 0:
        lm_loss = compute_lm_loss(lm_preds, gt_tokens.to(lm_preds.device))
        loss_total += lambda_lm_loss * lm_loss

    # Visual loss
    if lambda_visual_loss > 0:
        vm_loss, _ = compute_contrastive_loss(img_to_img_emb[:, 0:1], img_to_img_emb)
        loss_total += lambda_visual_loss * vm_loss

    # Cross-modal loss
    xm_similarities = None
    if lambda_xm_loss > 0:
        if sigurdsson:
            xmodal_loss = compute_sigurdsson_loss(txt_to_img_emb, img_to_txt_emb[:, 0], languages)
            xm_similarities = None
        else:
            txt_to_img_emb_xm = txt_to_img_emb
            if text_xm_indices is not None:
                # We do not want texts that come from a different distribution to be negatives. This will make the model
                # to pull the two distribution
                txt_to_img_emb_xm = txt_to_img_emb[:text_xm_indices]
            xmodal_loss, xm_similarities = \
                compute_contrastive_loss(txt_to_img_emb_xm, img_to_txt_emb, use_also_anchor=True, reverse=True)
        loss_total += lambda_xm_loss * xmodal_loss

    # Cross-language loss
    if lambda_xlang_loss > 0:
        xlang_loss = compute_xlang_loss(txt_to_txt_emb, img_to_img_emb, text_len, alpha_xm=alpha_xm,
                                        xm_similarities=xm_similarities)
        loss_total += lambda_xlang_loss * xlang_loss

    # Orthogonality constraint
    if lambda_orthogonality_loss > 0:
        orthogonality_loss = compute_orthogonality_loss(adapt_layer_weights)
        loss_total += lambda_orthogonality_loss * orthogonality_loss

    # Accuracy
    accuracy = None
    if compute_acc:  # Just for visualization purposes
        if sigurdsson:
            # Just orientative measure, because we combine all languages here
            accuracy = compute_accuracy(txt_to_img_emb.detach(), img_to_txt_emb[:, 0].detach(), topk=(1, 5),
                                        retrieval=True)
        else:
            accuracy = compute_accuracy(lm_preds.detach(), gt_tokens.detach(), topk=(1, 5))

    specific_losses = {'total': loss_total.detach(),
                       'lm': lm_loss.detach(),
                       'vm': vm_loss.detach(),
                       'xm': xmodal_loss.detach(),
                       'xl': xlang_loss.detach(),
                       'orthogonal': orthogonality_loss.detach()}
    return loss_total, specific_losses, accuracy

