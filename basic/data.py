import torch
import random

UNK_WORD = '<unk>'

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

class ScoreDataset(object):
    def __init__(self, data, data_type, batch_size, pad_id, ifgpu=True):

        self._dataset = data
        self._batch_size = batch_size
        if data_type == 'train':
            random.shuffle(self._dataset)
        self.sample_num = len(self._dataset)
        self.ifgpu = ifgpu
        self.pad_id = pad_id

    def get_batch_num(self):
        if self.sample_num % self._batch_size == 0:
            return self.sample_num // self._batch_size
        else:
            return self.sample_num // self._batch_size + 1

    def get_batch_data(self):
        for step in range(self.get_batch_num()):
            start = step * self._batch_size
            end = (step + 1) * self._batch_size
            if end > self.sample_num:
                end = self.sample_num

            instances = self._dataset[start:end]
            batch = len(instances)
            src_lengths = torch.LongTensor([len(inc) for inc in instances])
            max_seq_len = src_lengths.max()

            src = (torch.ones(batch, max_seq_len)*self.pad_id).long()
            src_mask = (torch.zeros(batch, max_seq_len)).float()

            for idx in range(len(instances)):
                temp_src_len = len(instances[idx])
                src[idx, :temp_src_len] = torch.LongTensor(instances[idx])
                src_mask[idx, :temp_src_len] = torch.ones(temp_src_len)

            if self.ifgpu:
                src = src.cuda()
                src_mask = src_mask.cuda()
                src_lengths = src_lengths.cuda()
            yield step, self.get_batch_num(), src, src_lengths, src_mask
