import argparse
from basic import Data, util
import Model
import torch
import torch.optim as optim
import os
import random
import numpy as np
import logging.config
from basic.Beam import Beam
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(1)
# device = torch.device("cuda:0")

seed_num = 57
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class vocab(object):
    def __init__(self):
        self.itos = []
        self.stoi = {}
        self.freq = {}

    def add_item(self, word):
        if word in self.stoi:
            self.freq[word] = self.freq[word] + 1
        else:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)
            self.freq[word] = 1

    def get_id(self, word):
        if word in self.stoi:
            return self.stoi[word]
        else:
            return self.stoi[UNK_WORD]

    def get_str(self, idx):
        if idx > len(self.itos):
            print(1)
        return self.itos[idx]


class sample(object):
    def __init__(self):
        self.src = []
        self.src_feat_1 = []
        self.src_feat_2 = []
        self.src_feat_3 = []
        self.tgt = []
        self.src_map = []
        self.alignment = []
        self.src_id = []
        self.src_fid_1 = []
        self.src_fid_2 = []
        self.src_fid_3 = []
        self.tgt_id = []
        self.copy_vocab = vocab()


def valid(model, batch_generator_dev):
    model.eval()

    stats = util.Statistics()

    for id, (step, batch_num, src, tgt_inp, tgt_oup, src_lengths, tgt_lengths, src_map, alignment, src_mask, tgt_mask,
             copy_vocab) in enumerate(batch_generator_dev):
        dec_state = None
        stats.n_src_words += src_lengths.sum()
        trunc_size = tgt_lengths.max()
        decoder_outputs, attns, dec_state = model(src, tgt_inp, src_lengths, dec_state)
        loss, batch_stats = model.compute_loss(decoder_outputs, attns, tgt_oup, src_map,
                                               alignment, copy_vocab)
        stats.update(batch_stats)
    return stats


def from_beam(beam):
    ret = {"predictions": [],
           "scores": [],
           "attention": []}
    for b in beam:
        n_best = 1
        if len(b.finished)==0:
            for i in range(b.size):
                s = (b.scores[i]).item()/len(b.next_ys)
                b.finished.append((s, len(b.next_ys)-1, i))
        scores, ks = b.sort_finished()
        hyps, attn = [], []
        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att = b.get_hyp(times, k)
            hyps.append(hyp)
            attn.append(att)
        ret["predictions"].append(hyps)
        ret["scores"].append(scores[0])
        ret["attention"].append(attn)
    return ret


def from_batch(translation_batch, tgt_vocab):
    # assume nbest==1

    copy_vocab = translation_batch["copy_vocab"]
    assert (len(translation_batch["scores"]) ==
            len(translation_batch["predictions"]))
    batch_size = len(copy_vocab)

    pred = translation_batch["predictions"]
    pred_score = translation_batch["scores"]

    # Sorting
    translations = []
    for b in range(batch_size):
        src_vocab = copy_vocab[b]
        tokens = []
        for tok in pred[b][0]:
            if tok < len(tgt_vocab.itos):
                tokens.append(tgt_vocab.get_str(tok))
            else:
                tokens.append(src_vocab.get_str(tok - len(tgt_vocab.itos)))
            if tokens[-1] == EOS_WORD:
                tokens = tokens[:-1]
                break
        # print(" ".join(tokens))
        translation = {'pred_sent': tokens, 'score': pred_score[b]}
        translations.append(translation)

    return translations

def test_greedy(model, batch_generator_test, beam_size, min_length, max_length, tgt_vocab, output_path, ifgpu):
    model.eval()
    all_sents = []
    for id, (step, batch_num, src, tgt_inp, tgt_oup, src_lengths, tgt_lengths, src_map, alignment, src_mask, tgt_mask, copy_vocab) in enumerate(batch_generator_test):
        batch_size = tgt_inp.size(0)
        dec_states, memory_bank = model.encoder(src)
        inp = torch.LongTensor(batch_size, 1).fill_(tgt_vocab.get_id(BOS_WORD))
        temp_sents = []
        for i in range(batch_size):
            temp_sents.append([])
        if ifgpu:
            inp = inp.cuda()
        for l_index in range(max_length):
            dec_out, attn, dec_states = model.decode(inp, dec_states, src_lengths, memory_bank)
            dec_out = dec_out.squeeze(0)  # batch * dim
            copy_attn = attn['copy'].transpose(0, 1).contiguous().squeeze(1) # batch * 1 * src_len
            out = model.compute_prob_w_copy(dec_out, copy_attn, src_map)
            # batch * (tgt_vocab_len+copy_vocab_len)

            out = model.collapse_copy_scores(out.unsqueeze(1), copy_vocab).squeeze(1)
            # batch * (tgt_vocab_len+copy_vocab_len)
            out = out.log()

            if l_index < min_length-1:
                if ifgpu:
                    out[:, tgt_vocab.get_id(EOS_WORD)] = torch.FloatTensor(batch_size).fill_(-1e20)
            # find best
            best_scores, best_scores_id = out.topk(1, 1, True, True)
            best_id = best_scores_id.squeeze(1)
            break_num = 0
            for i in range(batch_size):
                if len(temp_sents[i])==0 or temp_sents[i][-1]!=tgt_vocab.get_id(EOS_WORD):
                    temp_sents[i].append(best_id[i])
                else:
                    break_num+=1
            if break_num == batch_size:
                break
            inp = best_scores_id
            inp = inp.masked_fill(inp.gt(len(tgt_vocab.itos) - 1), 0)

        # Sorting
        for b in range(batch_size):
            src_vocab = copy_vocab[b]
            tokens = []
            for tok in temp_sents[b]:
                if tok < len(tgt_vocab.itos):
                    tokens.append(tgt_vocab.get_str(tok))
                else:
                    tokens.append(src_vocab.get_str(tok - len(tgt_vocab.itos)))
                if tokens[-1] == EOS_WORD:
                    tokens = tokens[:-1]
                    break
            all_sents.append(" ".join(tokens))
    output_file = open(output_path, 'w', encoding='utf-8')
    for s in all_sents:
        output_file.write(s + "\n")
    output_file.close()
    eval_str = 'perl ./multi-bleu.perl ./data/tgt_standard/test.tgt < '+output_path
    a = os.popen(eval_str)
    for line in a:
        logging.info(line.strip())

def test(model, batch_generator_test, beam_size, min_length, max_length, tgt_vocab, output_path, ifgpu):
    model.eval()
    pred_score_total, pred_words_total = 0, 0
    all_sents = []
    for id, (step, batch_num, src, tgt_inp, tgt_oup, src_lengths, tgt_lengths, src_map, alignment, src_mask, tgt_mask, copy_vocab) in enumerate(batch_generator_test):
        # print(id)
        batch_size = tgt_inp.size(0)
        beam = [Beam(beam_size, tgt_vocab.get_id(PAD_WORD), tgt_vocab.get_id(BOS_WORD),
                     tgt_vocab.get_id(EOS_WORD), gpu=ifgpu, min_length=min_length) for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def rvar(a):
            return a.repeat(beam_size, 1, 1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1).transpose(0, 1)

        dec_states, memory_bank = model.encoder(src)
        src_map = rvar(src_map) if model.copy else None  # (batch * beam) * src_len * copy_vocab_len
        memory_bank = rvar(memory_bank)  # (batch * beam) * src_len * src_emb_dim
        memory_lengths = src_lengths.repeat(beam_size)  # (batch * beam)
        dec_states.repeat_beam_size_times(beam_size)  # 2x(num_layer, batch * beam, dim); 1 * (batch * beam) * dim

        for i in range(max_length):
            # print(i)
            # if i == 150:
            #     print(0)
            if all((b.done() for b in beam)):
                break

            # Get all the pending current beam words and arrange for forward.
            inp = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1)  # (batch * beam)
            if ifgpu:
                inp = inp.cuda()
            # Turn copied words to UNKs
            if model.copy:
                inp = inp.masked_fill(inp.gt(len(tgt_vocab.itos) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation in the decoder
            inp = inp.unsqueeze(1)   # (batch * beam) * 1

            # Run one step.
            dec_out, attn, dec_states = model.decode(inp, dec_states, memory_lengths, memory_bank)

            dec_out = dec_out.squeeze(0)  # (batch * beam) * dim
            copy_attn = attn['copy'].squeeze(0)  # (batch * beam) * src_len
            out = model.compute_prob_w_copy(dec_out, copy_attn, src_map)
            # (batch * beam) * (tgt_vocab_len+copy_vocab_len)

            out = model.collapse_copy_scores(unbottle(out), copy_vocab)
            # batch * beam * (tgt_vocab_len+copy_vocab_len)
            out = out.log()
            beam_attn = unbottle(attn["copy"])  # batch * beam * src_len

            # (c) Advance each beam.
            temp_pred_list = []
            for j, b in enumerate(beam):
                b.advance(out[j, :],
                          beam_attn.data[j, :, :memory_lengths[j]])
                temp_pred_list.append(b.get_current_origin())
            dec_states.beam_update(temp_pred_list, beam_size)
        ret = from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        ret["copy_vocab"] = copy_vocab
        translations = from_batch(ret, tgt_vocab)

        for trans in translations:
            pred_score_total += trans['score']
            pred_words_total += len(trans['pred_sent'])

            all_sents.append(" ".join(trans['pred_sent']))

    output_file = open(output_path, 'w', encoding='utf-8')
    for s in all_sents:
        output_file.write(s+"\n")
    output_file.close()

    logging.info("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        'PRED', pred_score_total / pred_words_total,
        'PRED', math.exp(-pred_score_total / pred_words_total)))
    eval_str = 'perl ./multi-bleu.perl ./data/tgt_standard/test.tgt < '+output_path
    a = os.popen(eval_str)
    for line in a:
        logging.info(line.strip())

def main():
    args = parse_args()
    # make dir
    args.data_path = args.data_path + args.data_name + "/data.pt"
    args.vocab_path = args.vocab_path + args.data_name + "/vocab.pt"
    args.model_path = args.model_path + args.data_name + "/" + args.model_name + "/"
    args.output_path = args.output_path + args.data_name + "/" + args.model_name + "/"
    args.log_path = args.log_path + args.data_name + "/"
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    args.log_path = args.log_path + args.model_name + ".log"
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info(args)

    # read data
    data_dict = torch.load(args.data_path)
    vocab_dict = torch.load(args.vocab_path)
    train_data = data_dict['train']
    dev_data = data_dict['dev']
    test_data = data_dict['test']
    args.src_vocab_size.append(len(vocab_dict['src'].itos))
    args.src_vocab_size.append(len(vocab_dict['src_feat_1'].itos))
    args.src_vocab_size.append(len(vocab_dict['src_feat_2'].itos))
    args.src_vocab_size.append(len(vocab_dict['src_feat_3'].itos))
    args.tgt_vocab_size = len(vocab_dict['tgt'].itos)
    args.pad_ind = vocab_dict['src'].get_id(PAD_WORD)
    assert args.pad_ind == vocab_dict['tgt'].get_id(PAD_WORD) == vocab_dict['src_feat_1'].get_id(PAD_WORD) == \
           vocab_dict['src_feat_2'].get_id(PAD_WORD) == vocab_dict['src_feat_3'].get_id(PAD_WORD)
    logging.info('Data loaded...')

    # train batch
    train_dataset = Data.ScoreDataset(train_data, 'train', args.batch_size, args.pad_ind, args.gpu)
    dev_dataset = Data.ScoreDataset(dev_data, 'dev', args.batch_size, args.pad_ind, args.gpu)
    test_dataset = Data.ScoreDataset(test_data, 'test', args.batch_size, args.pad_ind, args.gpu)

    # model
    model = Model.S2SModel(args, vocab_dict)
    if args.gpu:
        model = model.cuda()
    # checkpoint = torch.load(
    #     '/home/chenshaowei/AAAI2021/model/rotowire/2_0.005lr_0trunc/20.pth')
    # model.load_state_dict(checkpoint['net'])
    # batch_generator_test = test_dataset.get_batch_data()
    # test(model, batch_generator_test, args.beam_size, args.min_length, args.max_length,
    #      vocab_dict['tgt'], args.output_path + str(28) + "_beam_2.txt", args.gpu)
    # exit(0)

    # optimizer
    param_list = list(model.named_parameters())
    for n, p in param_list:
        if p.requires_grad == True:
            print(n)
    grouped_params = [p for n, p in param_list if p.requires_grad == True]

    # make optimizer
    if args.optim_type == "adam":
        optimizer = optim.Adam(grouped_params, lr=args.learning_rate, betas=[0.9, 0.999], eps=1e-09)
    elif args.optim_type == "sgd":
        optimizer = optim.SGD(grouped_params, lr=args.learning_rate)
    elif args.optim_type == "adagrad":
        optimizer = optim.Adagrad(grouped_params, lr=args.learning_rate,
                                  initial_accumulator_value=args.adagrad_accum)
    elif args.optim_type == "adadelta":
        optimizer = optim.Adadelta(grouped_params, lr=args.learning_rate)
    elif args.optim_type == "rmsprop":
        optimizer = optim.RMSprop(grouped_params, lr=args.learning_rate)
    else:
        print('Error Optimizer!')

    # load saved model, optimizer and epoch num
    if args.reload and os.path.exists(args.model_path + '{}.pth'.format(args.saved_epoch_num)):
        checkpoint = torch.load(args.model_path + '{}.pth'.format(args.saved_epoch_num))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
    else:
        start_epoch = 1
        print('New model and optimizer from epoch 0')

    best_dev_ppl = 0.
    last_ppl = None
    ppl = None
    lr = args.learning_rate
    for epoch in range(start_epoch, start_epoch + args.epoch_num):
        model.train()

        # updata learning rate
        start_decay = False
        if args.start_decay_at is not None and epoch >= args.start_decay_at:
            start_decay = True
        if last_ppl is not None and ppl is not None:
            if ppl > last_ppl:
                start_decay = True

        if start_decay:
            lr = lr * args.learning_rate_decay
            optimizer.param_groups[0]["lr"] = lr
            logging.info("Decaying learning rate to %g" % lr)
        if ppl is not None:
            last_ppl = ppl

        batch_generator = train_dataset.get_batch_data()

        total_stats = util.Statistics()
        report_stats = util.Statistics()
        for id, (step, batch_num, src, tgt_inp, tgt_oup, src_lengths, tgt_lengths, src_map, alignment, src_mask, tgt_mask,
                 copy_vocab) in enumerate(batch_generator):
            if args.trunc_size:
                trunc_size = args.trunc_size
            else:
                trunc_size = tgt_lengths.max().item()
            dec_state = None
            report_stats.n_src_words += src_lengths.sum().item()
            total_stats.n_src_words += src_lengths.sum().item()
            for j in range(0, tgt_lengths.max().item() - 1, trunc_size):
                # 1. Create truncated target.
                tgt_inp_temp = tgt_inp[:, j: j + trunc_size]
                tgt_oup_temp = tgt_oup[:, j: j + trunc_size]
                align_temp = alignment[:, j: j + trunc_size]
                optimizer.zero_grad()

                decoder_outputs, attns, dec_state = model(src, tgt_inp_temp, src_lengths, dec_state)
                loss, batch_stats = model.compute_loss(decoder_outputs, attns, tgt_oup_temp, src_map,
                                                       align_temp, copy_vocab)
                loss.backward()  # no normalization
                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(grouped_params, args.max_grad_norm)
                optimizer.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            # log
            if step % 100 == 0:
                report_stats.output(epoch, step, batch_num)
                report_stats = util.Statistics()
            # if step / 100 == 3:
            #     state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            #     torch.save(state, args.model_path + str(epoch) + '.pth')
            #     batch_generator_test = test_dataset.get_batch_data()
            #
            #     test_greedy(model, batch_generator_test, args.beam_size, args.min_length, args.max_length,
            #          vocab_dict['tgt'], args.output_path + str(epoch) + ".txt", args.gpu)

        # validation
        batch_generator_dev = dev_dataset.get_batch_data()
        dev_state = valid(model, batch_generator_dev)
        dev_state.output_dev(epoch)
        # logging.info(dev_state.ppl())

        # save checkpoint
        if epoch > args.start_checkpoint_at:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, args.model_path + str(epoch) + '.pth')

        # test
        batch_generator_test = test_dataset.get_batch_data()
        test(model, batch_generator_test, args.beam_size, args.min_length, args.max_length,
            vocab_dict['tgt'], args.output_path+str(epoch)+".txt", args.gpu)


def parse_args():
    # config
    parser = argparse.ArgumentParser(description='dual learning on data-to-text')
    parser.add_argument('--model_name', type=str, default="0.005lr_0trunc_adam_mlp")
    parser.add_argument('--data_name', type=str, default="rotowire", choices=["rotowire", "mlb", "rotowire_new"])
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--vocab_path', type=str, default="data/")
    parser.add_argument('--model_path', type=str, default="model/")
    parser.add_argument('--output_path', type=str, default="output/")
    parser.add_argument('--log_path', type=str, default="log/")

    parser.add_argument('--pad_ind', type=int, default=0)
    parser.add_argument('--src_vocab_size', type=list, default=[])
    parser.add_argument('--tgt_vocab_size', type=int, default=1000)
    parser.add_argument('--emb_size', type=int, default=600)
    parser.add_argument('--emb_out_size', type=int, default=600)
    parser.add_argument('--tgt_emb_dim', type=int, default=500)
    parser.add_argument('--encoder_dim', type=int, default=600)

    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--decoder_dim', type=int, default=600)
    parser.add_argument('--coverage_attn', type=bool, default=True)
    parser.add_argument('--attn_type', type=str, default="mlp", choices=["dot", "general", "mlp"])
    parser.add_argument('--copy_attn', type=bool, default=True)
    parser.add_argument('--reuse_copy_attn', type=bool, default=True)

    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--optim_type', type=str, default="adam",
                        choices=["adam", "sgd", "adagrad", "adadelta", "rmsprop"])
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--adagrad_accum', type=float, default=0.15)
    parser.add_argument('--learning_rate_decay', type=float, default=0.98)
    parser.add_argument('--max_grad_norm', type=float, default=5)

    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--saved_epoch_num', type=int, default=0)
    parser.add_argument('--start_decay_at', type=int, default=15)
    parser.add_argument('--trunc_size', type=int, default=0)
    parser.add_argument('--start_checkpoint_at', type=int, default=0)

    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--gen_length', type=int, default=200)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--min_length', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=400)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()