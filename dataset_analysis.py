
import torch

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

def main():

    data_path = "data/rotowire/data.pt"
    vocab_path = "data/rotowire/vocab.pt"

    # read data
    data_dict = torch.load(data_path)
    vocab_dict = torch.load(vocab_path)
    train_data = data_dict['train']
    dev_data = data_dict['dev']
    test_data = data_dict['test']
    src_vocab_size = []
    src_vocab_size.append(len(vocab_dict['src'].itos))
    src_vocab_size.append(len(vocab_dict['src_feat_1'].itos))
    src_vocab_size.append(len(vocab_dict['src_feat_2'].itos))
    src_vocab_size.append(len(vocab_dict['src_feat_3'].itos))
    tgt_vocab_size = len(vocab_dict['tgt'].itos)
    pad_ind = vocab_dict['src'].get_id(PAD_WORD)

    print(1)


if __name__ == "__main__":
    main()