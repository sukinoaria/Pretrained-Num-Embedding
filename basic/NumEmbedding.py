import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.copy = config.copy_attn
        self.reuse_copy_attn = config.reuse_copy_attn

        self.eps = 1e-20
        self.force_copy = False
        self.normalize_by_length = True

        # encoder and decoder
        #self.src_embedding = Embedding(config)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, self.tgt_emb_dim, padding_idx=config.pad_ind)

        # generation
        self.gen_linear = nn.Linear(self.dec_dim, config.tgt_vocab_size)
        self.ifcopy_linear = nn.Linear(self.dec_dim, 1)
        self.drop = nn.Dropout(config.dropout_rate)

    def encoder(self, src):
        emb = self.src_embedding(src)  # batch_size * seq_len * emb_dim
        # encoder_hidden = emb.mean(1).unsqueeze(0).repeat(2, 1, 1)  # batch_size * emb_dim
        # repeat self.decoder_layer times, different dec layer input, replace decoder hidden state
        encoder_hidden = emb.mean(1).unsqueeze(0).repeat(self.decoder.num_layer, 1, 1)  # batch_size * emb_dim
        # decoder hidden state, h_0 and c_0,each have #num_layer layers
        encoder_final = (encoder_hidden, encoder_hidden)
        memory_bank = emb
        enc_state = RNNDecoderState(encoder_hidden.size(2), encoder_final, self.ifgpu)
        return enc_state, memory_bank

    def decode(self, tgt_ids, state, input_length, memory_bank):
        '''

        :param tgt_ids: batch * tgt_len
        :param state: save hidden(tuple, 2x(num_layer, batch, dim)),
                      and input_feed(batch, enc_dim)
        :param input_length: batch
        :param memory_bank: batch * src_len * src_emb_dim
        :return: decoder_outputs(tgt_len * batch * dim), attns(tgt_len * batch * src_len), state
        '''

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self.copy:
            attns["copy"] = []

        tgt_emb = self.tgt_embeddings(tgt_ids)
        assert tgt_emb.dim() == 3  # batch * seq_len * tgt_emb_dim

        hidden = state.hidden  # (layer * batch * dim)*2
        input_feed = state.input_feed.squeeze(0)  # batch * dim
        # tgt emb use idx i to select generation position tgt_emb[:,i,:]
        for i, emb_t in enumerate(tgt_emb.split(1, dim=1)):
            emb_t = emb_t.squeeze(1)  # batch * tgt_emb_dim
            try:
                decoder_input = torch.cat([emb_t, input_feed], 1)  # batch * (tgt_emb_dim + dim)
            except:
                decoder_input = torch.cat([emb_t, input_feed.unsqueeze(0)], 1)
            rnn_output, hidden = self.decoder(decoder_input, hidden)  # batch * dim; (layer * batch * dim)*2
            decoder_output, p_attn = self.attn(rnn_output, memory_bank, input_length)
            # batch * dim; batch * 1 * src_len

            decoder_output = self.drop(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]  # (batch * dim) list
            attns["std"] += [p_attn.squeeze(1)]  # (batch * src_len) list

            #  copy attention
            if self.copy and not self.reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output, memory_bank, input_length)
                attns["copy"] += [copy_attn.squeeze(1)]
            elif self.copy:
                attns["copy"] = attns["std"]  # (batch * src_len) list
        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)  # tgt_len * batch * dim
        for k in attns:
            attns[k] = torch.stack(attns[k])  # tgt_len * batch * src_len

        # Update the state with the result.
        state.update_state(hidden, input_feed)

        return decoder_outputs, attns, state

    def compute_prob_w_copy(self, decoder_outputs, copy_attn, src_map):
        '''
        :param decoder_outputs: (batch * tgt_len) * dim
        :param copy_attn: (batch * tgt_len) * src_len
        :param src_map: batch * src_len * copy_vocab_len
        :param cvocab:
        :return: (batch * tgt_len) * (tgt_vocab_len+copy_vocab_len)
        '''

        batch_size = src_map.size(0)
        slen = copy_attn.size(-1)
        cvocab = src_map.size(-1)

        # Probability of generation
        gen_prob = self.gen_linear(decoder_outputs)  # (batch * tgt_len) * tgt_vocab_len
        gen_prob[:, self.pad] = -float('inf')
        gen_prob = F.softmax(gen_prob)

        # Probability of if copy
        ifcopy_prob = F.sigmoid(self.ifcopy_linear(decoder_outputs))  # (batch * tgt_len) * 1

        # Probibility of final
        out_prob = torch.mul(gen_prob, 1 - ifcopy_prob.expand_as(gen_prob))  # (batch * tgt_len) * tgt_vocab_len
        mul_attn = torch.mul(copy_attn, ifcopy_prob.expand_as(copy_attn))    # (batch * tgt_len) * src_len
        copy_prob = torch.bmm(mul_attn.view(batch_size, -1, slen), src_map.float())  # batch * tgt_len * copy_vocab_len
        copy_prob = copy_prob.view(-1, cvocab)
        prob = torch.cat([out_prob, copy_prob], 1)  # (batch * tgt_len) * (tgt_vocab_len+copy_vocab_len)
        return prob

    def collapse_copy_scores(self, scores_data, copy_vocab):
        '''
        :param scores_data: batch * tgt_len * (tgt_vocab_len+copy_vocab_len)
        :param copy_vocab: list
        :param batch_size:
        :return: batch * tgt_len * (tgt_vocab_len+copy_vocab_len)
        '''
        batch_size = scores_data.size(0)
        for b in range(batch_size):
            blank = []
            fill = []
            src_vocab = copy_vocab[b]
            for i in range(1, len(src_vocab.itos)):
                sw = src_vocab.get_str(i)
                ti = self.vocab['tgt'].get_id(sw)
                if ti != 0:
                    blank.append(self.offset + i)  # index_from
                    fill.append(ti)  # index_to
            if blank:
                blank = torch.Tensor(blank).long()
                fill = torch.Tensor(fill).long()
                if self.ifgpu:
                    blank = blank.cuda()
                    fill = fill.cuda()
                scores_data[b].index_add_(1, fill, scores_data[b].index_select(1, blank))  # add blank to fill
                scores_data[b].index_fill_(1, blank, 1e-10)  # set blank to 1e-10
        return scores_data

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

    def forward(self, src, tgt, src_lengths, dec_state):

        '''
        :param src: batch * src_len * feat_num
        :param src_lengths: batch
        :param tgt: batch * tgt_len
        :param dec_state:
        :return:
        '''
        enc_state, memory_bank = self.encoder(src)
        decoder_outputs, attns, dec_state = self.decode(tgt, enc_state if dec_state is None else dec_state,
                                                        src_lengths, memory_bank)
        return decoder_outputs, attns, dec_state
