# -*- coding: utf-8 -*-

import torch
from basic.data import vocab,sample

def build_texts(all_data):
    num_texts = []
    for instance in all_data:
        contents, feat_1,feat_2 = instance.src,instance.src_feat_1,instance.src_feat_2

        # row level nums extraction
        last_idx = None
        last_nums = set()
        for feat_name,value in zip(feat_1,contents):
            if feat_name != last_idx:
                if len(last_nums)>=2:
                    num_texts.append(list(last_nums))
                    last_nums = set()
                last_idx = feat_name
                if value.isnumeric():
                    last_nums.add(int(value))
            else:
                if value.isnumeric():
                    last_nums.add(int(value))
        if last_nums: num_texts.append(list(last_nums))
        # col level
        uni_cols = set(feat_2)
        candidates = {key:set() for key in uni_cols}
        for feat_name,value in zip(feat_2,contents):
            if value.isnumeric():
                candidates[feat_name].add(int(value))
        for key,val in candidates.items():
            if len(val) >= 2:
                num_texts.append(list(val))
    return num_texts

def main():
    INPUT_FILE = "./data/rotowire/data.pt"
    OUTPUT_FILE = './data/rotowire/nums.pt'

    data_dict = torch.load(INPUT_FILE)
    res = dict()
    res['train'] = build_texts(data_dict['train'])
    res['dev'] = build_texts(data_dict['dev'])
    res['test'] = build_texts(data_dict['test'])

    # save
    print("count: train:{}, dev:{}, test:{}".format(len(res['train']),len(res['dev']),len(res['test'])))
    torch.save(res, OUTPUT_FILE)

if __name__ == '__main__':
    main()