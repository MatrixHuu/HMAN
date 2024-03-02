import torch
import torch.nn as nn

from models.attention import SemanticAlignment, SemanticAttention


class Feat_Decoder(nn.Module):
    def __init__(self, vis_feat_size, feat_len, embedding_size, sem_align_hidden_size,
                 sem_attn_hidden_size, hidden_size):
        super(Feat_Decoder, self).__init__()
        self.vis_feat_size = vis_feat_size
        self.feat_len = feat_len
        self.embedding_size = embedding_size
        self.sem_align_hidden_size = sem_align_hidden_size
        self.sem_attn_hidden_size = sem_attn_hidden_size
        self.hidden_size = hidden_size

        self.semantic_alignment = SemanticAlignment(
            query_size=self.embedding_size,
            feat_size=self.vis_feat_size,
            bottleneck_size=self.sem_align_hidden_size)
        self.semantic_attention = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.embedding_size + self.vis_feat_size,
            bottleneck_size=self.sem_attn_hidden_size)


    def forward(self, embedded, last_hidden, vis_feats, phr_feats, phr_masks):
        semantic_group_feats, semantic_align_weights, semantic_align_logits = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=vis_feats)
        feat, semantic_attn_weights, semantic_attn_logits = self.semantic_attention(
            query=last_hidden,
            keys=semantic_group_feats,
            values=semantic_group_feats,
            masks=phr_masks)

        feat = torch.cat((
            feat,
            embedded), dim=1)

        return feat, (semantic_align_weights, semantic_attn_weights), \
               (semantic_align_logits, semantic_attn_logits)
