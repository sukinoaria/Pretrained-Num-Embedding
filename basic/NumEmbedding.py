import torch
import torch.nn as nn
from basic.NumProjLayer import NumProjLayer

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
        self.hinge_loss_delta = torch.tensor(config.hinge_loss_delta)

        # modules
        self.numProj = NumProjLayer(config)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.proj_dim,nhead=config.num_emb_ahead)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=config.num_emb_layer)

        # mlp layer for loss calc
        self.score_linear = nn.Linear(config.proj_dim, 1)

    def forward(self, src):
        proj_state = self.numProj(src)
        out = self.encoder(proj_state)
        return out

    def compute_loss(self, encoder_state, tgt, tgt_lengths, src_mask):
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)

        out_scores = torch.sigmoid(self.score_linear(encoder_state)).squeeze()

        # calc hinge loss
        scores_minus = out_scores.unsqueeze(2).repeat(1,1,tgt_len) - out_scores.unsqueeze(1).repeat(1,tgt_len,1)

        nums_minus = tgt.unsqueeze(2).repeat(1,1,tgt_len) - tgt.unsqueeze(1).repeat(1,tgt_len,1)
        T_nums = torch.ge(nums_minus,0) # true,false for >=, <
        T_gt = T_nums.float() + (~T_nums).float()*-1

        # build mask for loss sumarizition, not conside i to i loss
        loss_mask = src_mask.unsqueeze(1).repeat(1,tgt_len,1) * src_mask.unsqueeze(2).repeat(1,1,tgt_len)

        # only use half value to calc loss
        loss_mask = torch.triu(loss_mask)

        eye_matrix = torch.stack([torch.eye(tgt_len) for _ in range(batch_size)])
        if self.ifgpu: eye_matrix = eye_matrix.cuda()

        loss_mask -= eye_matrix * loss_mask

        # calc each loss for token i and j
        loss_matrix = self.hinge_loss_delta - T_gt * scores_minus
        zero_matrix = torch.zeros(batch_size,tgt_len,tgt_len)
        if self.ifgpu: zero_matrix = zero_matrix.cuda()
        pos_loss = torch.max(loss_matrix,zero_matrix)

        #sum up and divide lenth^2 to get average loss
        masked_pos_loss = pos_loss * loss_mask
        pos_loss_sum = torch.sum(masked_pos_loss.view(batch_size,-1),dim=1)
        averaged_loss = torch.sum(pos_loss_sum / (tgt_lengths ** 2))

        # get static information
        pred_bool_mask = torch.ge(scores_minus,0)
        pred_result = pred_bool_mask.float() + (~pred_bool_mask).float()*-1
        masked_pred_result = pred_result * T_gt * loss_mask
        correct_nums = torch.sum(torch.gt(masked_pred_result,0))

        #stats = Statistics(averaged_loss.item(),loss_mask.sum(),correct_nums)
        stats = (averaged_loss.item(), loss_mask.sum(), correct_nums)

        return averaged_loss, stats, masked_pred_result

