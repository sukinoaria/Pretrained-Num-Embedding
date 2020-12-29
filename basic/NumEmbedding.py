import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.NumProjEmbedding import NumProjEmbedding

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class NumEmbedding(nn.Module):
    def __init__(self, config):
        super(NumEmbedding, self).__init__()
        # params
        self.ifgpu = config.gpu
        self.dec_dim = config.decoder_dim

        self.tgt_emb_dim = config.tgt_emb_dim

        self.eps = 1e-20
        self.force_copy = False
        self.normalize_by_length = True

        # modules
        self.numProj = NumProjEmbedding(config)

        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, self.tgt_emb_dim, padding_idx=config.pad_ind)

        # generation
        self.gen_linear = nn.Linear(self.dec_dim, config.tgt_vocab_size)

        self.drop = nn.Dropout(config.dropout_rate)


    def forward(self, src):

        '''
        :param src: batch * src_len * feat_num
        :param src_lengths: batch
        :param tgt: batch * tgt_len
        :param dec_state:
        :return:
        '''
        proj_state = self.numProj(src)

        # transformer layer

        decoder_outputs, attns, dec_state = self.decode(tgt, enc_state if dec_state is None else dec_state,
                                                        src_lengths, memory_bank)
        return decoder_outputs, attns, dec_state

    def compute_loss(self, decoder_outputs, attns, tgt, src_map, alignment, copy_vocab):
        '''
        :param decoder_outputs: tgt_len * batch * dim
        :param attns: tgt_len * batch * src_len for each
        :param tgt: batch * tgt_len
        :param src_map: batch * src_len * copy_vocab_len
        :param alignment: batch * tgt_len
        :param copy_vocab:
        :return:
        '''
        batch_size = tgt.size(0)
        tlen = tgt.size(-1)

        decoder_outputs = decoder_outputs.transpose(0, 1).contiguous()  # batch * tgt_len * dim
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))  # (batch * tgt_len) * dim

        copy_attn = attns['copy'].transpose(0, 1).contiguous()  # batch * tgt_len * src_len
        copy_attn = copy_attn.view(-1, copy_attn.size(-1))  # (batch * tgt_len) * src_len

        target = tgt.contiguous().view(-1)  # (batch * tgt_len)
        align = alignment.contiguous().view(-1)  # (batch * tgt_len)
        assert target.size(0) == align.size(0)

        prob = self.compute_prob_w_copy(decoder_outputs, copy_attn, src_map)
        # (batch * tgt_len) * (tgt_vocab_len+copy_vocab_len)

        # Compute unks in align and target for readability
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()  # (batch * tgt_len)

        # Copy probability of tokens in source
        out = prob.gather(1, align.view(-1, 1) + self.offset).view(-1)  # (batch * tgt_len)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = prob.gather(1, target.view(-1, 1)).view(-1)  # (batch * tgt_len)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())  # (batch * tgt_len)

        scores_data = prob.data.clone()  # (batch * tgt_len) * (tgt_vocab_len+copy_vocab_len)
        scores_data = scores_data.view(batch_size, tlen, -1)
        scores_data = self.collapse_copy_scores(scores_data, copy_vocab)
        scores_data = scores_data.view(batch_size * tlen, -1)

        # Correct target copy token instead of <unk>
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.vocab['tgt'].itos)) * correct_mask.long()
        target_data = target_data + correct_copy  # (batch * tgt_len)

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        pred = scores_data.max(1)[1]
        non_padding = target.ne(self.pad)
        num_correct = pred.eq(target_data).masked_select(non_padding).sum()  # (batch * tgt_no_pad_len)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            tgt_lens = tgt.ne(self.pad).float().sum(1) + 1  # batch
            # Compute Total Loss per sequence in batch
            loss = loss.view(batch_size, -1).sum(1)  # batch
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()
        stats = util.Statistics(loss.item(), non_padding.sum(), num_correct)

        return loss, stats

