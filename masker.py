import torch


class Masker:
    def __init__(self, tok, p_mask=0.8, p_clobber_other_txt=1 / 6):
        """

        :param tok: tokenizer
        :param p_mask: probability of replacing with the token <mask> the tokens that are labeled as 'clobbered'.
        :param p_clobber_other_txt: probability of labeling as 'clobbered' the non-target text tokens.
        """
        self.tokenizer = tok
        self.p_mask = p_mask
        self.p_clobber_other_txt = p_clobber_other_txt

    def mask_text(self, text, positions_to_predict, text_len, randomize=True):
        """
        :param text: input tokens
        :param positions_to_predict: positions from this text that we have to predict
        :param text_len: length of the text sequences. Used to not mask the padding
        :param randomize: if False, we use 1 instead of self.p_mask (always replace with <mask>), and never clobber
        images or tokens that are not the target ones (replace p_clobber_other_text with 0)
        """
        # Note: this assumes all text, except for the padding, is original text, not special tokens (added later)
        prob_mask = self.p_mask if randomize else 1.
        prob_clobber_other_text = self.p_clobber_other_txt if randomize else 0
        masked_text = text.clone()
        # p contains the probability of clobbering
        p = torch.zeros_like(masked_text).float()
        for i in range(masked_text.shape[0]):
            p[i, :text_len[i]] = prob_clobber_other_text  # The padding is not masked
            p[i, positions_to_predict[i]] = 1
        # clobber is  True in the positions that are labeled as clobbered, which means that with probability
        # prob_mask will be replaced with <mask>, and with probability (1-prob_mask)/2 will be replaced with a
        # random word
        clobber = torch.bernoulli(p).bool()

        # With prob_mask probability, we replace clobbered input tokens with the token (<mask>)
        indices_masked = torch.bernoulli(torch.full(masked_text.shape, prob_mask).to(text.device)).bool() & clobber
        masked_text[indices_masked] = self.tokenizer.index_special_tokens['<mask>']

        #  With (1-prob_mask)/2 probability, we replace clobbered input tokens tokens with a random token
        indices_random = torch.bernoulli(
            torch.full(masked_text.shape, (1 - prob_mask) / 2)).bool().to(text.device) & clobber & ~indices_masked
        random_words = torch.randint(len(self.tokenizer), masked_text.shape, dtype=torch.long, device=text.device)
        masked_text[indices_random] = random_words[indices_random]

        return masked_text

    def mask_text_all_tokens(self, text, text_len):
        assert text.shape[0] == 1  # Only works for batch sizes of 1
        masked_text = text.clone().repeat((text_len, 1))
        for i in range(text_len):
            masked_text[i, i] = self.tokenizer.index_special_tokens['<mask>']
        return masked_text
