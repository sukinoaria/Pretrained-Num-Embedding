import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class NumProjEmbedding(nn.Module):
    def __init__(self, config):
        super(NumProjEmbedding, self).__init__()
        # params
        self.ifgpu = config.gpu
        self.proj_parts = torch.Tensor(config.proj_parts).long()
        self.part_dim = len(config.proj_parts)
        self.out_dim = config.proj_dim

        self.weight = nn.Parameter(torch.Tensor(self.part_dim, self.out_dim))
        self.bias = config.proj_bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.part_dim,self.out_dim))
        else:
            self.register_parameter('bias', None)

        self.quantify = nn.Parameter(torch.Tensor(self.out_dim,self.out_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.quantify, a=math.sqrt(5))

    def forward(self, src):

        '''
        :param src: batch * src_len * feat_num
        :param src_lengths: batch
        :param tgt: batch * tgt_len
        :param dec_state:
        :return:
        '''
        # build mask for weight selection
        batch_size,src_len = src.shape
        proj_parts = self.proj_parts.unsqueeze(0).repeat(batch_size*src_len,1)
        src_tmp = src.view(batch_size*src_len,-1).repeat(1,self.part_dim)
        proj_idx = torch.sum(torch.gt(src_tmp,proj_parts),dim=1)

        # select proj weights, calc project result
        weights = torch.index_select(self.weight,0,proj_idx)
        bias = torch.index_select(self.bias,0,proj_idx)

        # todo wx+b





        return proj_state



