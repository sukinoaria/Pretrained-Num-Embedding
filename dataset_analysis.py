
import torch
from basic.data import *

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def main():

    data_path = "data/rotowire/nums.pt"


    # read data
    data_dict = torch.load(data_path)

    train_data = data_dict['train']
    dev_data = data_dict['dev']
    test_data = data_dict['test']

    all_data = train_data + dev_data + test_data

    array_list = []
    num_count = 0
    for instance in all_data:
        num_count += len(instance)
        array_list.extend(instance)

    print(max(array_list),min(array_list))


if __name__ == "__main__":
    main()