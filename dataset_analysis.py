
import torch
from basic.data import *

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


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

    all_data = train_data + dev_data + test_data
    cnt = 0
    for instance in all_data:
        tgt_str = " ".join(instance.tgt)
        find_str = "The Atlanta Hawks defeated the Miami Heat"

        if find_str in tgt_str:
            #print(1)
            cnt +=1
    print(cnt)


if __name__ == "__main__":
    main()