import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class EnhancedObj(nn.Module):
    def __init__(self, opt, baseline=False):
        super(EnhancedObj, self).__init__()
        self.baseline = baseline

        self.obj_embed = nn.Linear(opt.region_projected_size, opt.region_projected_size)
        self.obj_norm = nn.Sequential(
                nn.Tanh(),
                nn.LayerNorm(opt.region_projected_size)
            )
        visual_input_size = opt.hidden_size*2

        self.visual_embed = nn.Linear(visual_input_size, opt.hidden_size*2)

        self.visual_norm = nn.Sequential(
            nn.Tanh(),
            nn.LayerNorm(opt.hidden_size*2)
        )
        self.obj_visual_norm = nn.Sequential(
            nn.Tanh(),
            nn.LayerNorm(opt.hidden_size*2),
            # nn.Dropout(args.dropout)
        )



    def forward(self, visual_feats, obj_feats):
        bs, win_len, obj_num, obj_size = obj_feats.size()

        visual_embed = self.visual_embed(visual_feats)

        visual_embed = self.visual_norm(visual_embed)

        obj_embed = self.obj_embed(obj_feats).view(bs, win_len*obj_num, -1)

        obj_embed = self.obj_norm(obj_embed)

        adj_matrix = torch.div(torch.matmul(obj_embed, visual_embed.transpose(-1,-2)), torch.tensor(np.sqrt(obj_size)))
        adj_matrix = F.softmax(adj_matrix, dim=1)

        # bs * win_len * d
        obj_agg = torch.matmul(obj_embed.transpose(-1,-2), adj_matrix).transpose(-1,-2)
        obj_visual = obj_agg + visual_embed
        obj_visual = self.obj_visual_norm(obj_visual)


        return obj_visual
