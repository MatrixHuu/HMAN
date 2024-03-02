'''
    pytorch implementation of our RMN model
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gumbel as gumbel
from models.allennlp_beamsearch import BeamSearch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionShare(nn.Module): # 512 -> 512opt.hidden_size + 300opt.word_size
    def __init__(self, input_value_size, input_key_size, output_size, dropout=0.5):
        super(AttentionShare, self).__init__()
        self.input_value_size = input_value_size
        self.input_key_size = input_key_size
        self.attention_size = output_size
        self.dropout = dropout

        self.K = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.Q = nn.Linear(in_features=input_key_size, out_features=output_size, bias=False)
        self.V = nn.Linear(in_features=input_value_size, out_features=input_value_size, bias=False)    #out_features=output_size-->out_features=input_value_size
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.input_value_size, out_features=input_value_size, bias=False),
            nn.Tanh(),
            nn.LayerNorm(input_value_size),
            nn.Dropout(self.dropout)
        )
    # query
    def forward(self, meta_state, hidden_previous):

        K = self.K(meta_state)
        Q = self.Q(hidden_previous).unsqueeze(2)
        V = self.V(meta_state).transpose(-1, -2)

        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        weight = F.softmax(logits, dim=1)
        # weight = F.sigmoid(logits)
        mid_step = torch.matmul(V, weight)
        # mid_step = torch.matmul(V, weight)

        attention = mid_step.squeeze(2)

        attention = self.output_layer(attention)

        return attention, weight


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global

class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global

class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """
    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SCAN_attention(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """
    def __init__(self):
        super(SCAN_attention, self).__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, query, context, smooth, eps=1e-8):
        """
            query: (n_context, queryL, d)
            context: (n_context, sourceL, d)
            """
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)

        attn = self.relu(attn)
        attn = l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch, queryL, sourceL)
        attn = self.softmax(attn * smooth)

        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)
        weightedContext = l2norm(weightedContext, dim=-1)

        return weightedContext

'''    
def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext
'''

# ------------------------------------------------------
# ------------ Soft Attention Mechanism ----------------
# ------------------------------------------------------

class SoftAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(SoftAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

    def forward(self, feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha

# ------------------------------------------------------
# ------------ Gumbel Attention Mechanism --------------
# ------------------------------------------------------

class GumbelAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(GumbelAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)

    def forward(self, feats, key):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        v = self.wv(feats)
        inputs = self.wh(key).unsqueeze(1).expand_as(v) + v
        outputs = self.wa(torch.tanh(inputs)).squeeze(-1)
        if self.training:
            alpha = gumbel.st_gumbel_softmax(outputs)
        else:
            alpha = gumbel.greedy_select(outputs).float()

        att_feats = torch.bmm(alpha.unsqueeze(1), feats).squeeze(1)
        return att_feats, alpha

# ------------------------------------------------------
# --- Pre-process visual features by Bi-LSTM Encoder ---
# ------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.a_feature_size = opt.a_feature_size
        self.m_feature_size = opt.m_feature_size
        self.hidden_size = opt.hidden_size
        self.use_multi_gpu = opt.use_multi_gpu

        # frame feature embedding
        self.frame_feature_embed = nn.Linear(opt.a_feature_size, opt.frame_projected_size)
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        self.bi_lstm1 = nn.LSTM(opt.frame_projected_size, opt.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_drop1 = nn.Dropout(p=opt.dropout)

        # i3d feature embedding
        self.i3d_feature_embed = nn.Linear(opt.m_feature_size, opt.frame_projected_size)
        nn.init.xavier_normal_(self.i3d_feature_embed.weight)
        self.bi_lstm2 = nn.LSTM(opt.frame_projected_size, opt.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_drop2 = nn.Dropout(p=opt.dropout)

        # region feature embedding
        self.region_feature_embed = nn.Linear(opt.region_feature_size, opt.region_projected_size)
        nn.init.xavier_normal_(self.region_feature_embed.weight)

        # location feature embedding
        self.spatial_feature_embed = nn.Linear(opt.spatial_feature_size, opt.spatial_projected_size)
        nn.init.xavier_normal_(self.spatial_feature_embed.weight)

        # time embedding matrix
        self.time_feats = nn.Parameter(torch.randn(opt.max_frames, opt.time_size))

        # fuse region+loc+time
        in_size = opt.spatial_projected_size + opt.time_size
        self.visual_embed = nn.Linear(opt.region_projected_size + in_size, opt.region_projected_size)
        nn.init.xavier_normal_(self.visual_embed.weight)
        self.visual_drop = nn.Dropout(p=opt.dropout)

    # def mask(self, A):
    #
    #     bsz, _, _, fsize = A.shape
    #
    #     AA = A.permute(0, 1, 3, 2)
    #
    #     dot_matrix = A.matmul(AA)
    #
    #     v1_row_norm = torch.norm(A, p=2, dim=3, keepdim=True)
    #
    #     v2_col_norm = torch.norm(AA, p=2, dim=2, keepdim=True)
    #
    #     norm_matrix = v1_row_norm.matmul(v2_col_norm)
    #
    #     res = dot_matrix / norm_matrix
    #
    #     res[torch.isneginf(res)] = 0
    #
    #     sum1 = res.sum(axis=3, keepdim=False)
    #
    #     # ones = sum1.mean(dim=2, keepdim=True).expand_as(sum1)*2
    #     #
    #     # mask_k = torch.gt(sum1, ones)
    #     #
    #     # mask_res_per_frame = mask_k.sum(axis=2)
    #     # mask_res_per_video = mask_res_per_frame.sum(axis=1)
    #     # mask_res_per_batch = mask_res_per_video.sum(axis=0)
    #     # print('mask_res_per_batch', mask_res_per_batch)
    #     # result = A.masked_fill(mask_k, value=torch.tensor(-1e9))
    #
    #     _, top2 = sum1.topk(18, dim=2, largest=True, sorted=True)
    #
    #     dim0, dim1, dim2 = top2.shape
    #
    #     mask_k = torch.zeros(bsz, 26, 36, 1).to(DEVICE)
    #
    #     for i in range(dim0):
    #         for j in range(dim1):
    #             for k in range(dim2):
    #                 mask_k[i, j, top2[i, j, k]] = 1
    #
    #     mask_k = mask_k.bool()
    #     #
    #     mask_k.to(DEVICE)
    #
    #     result = A.masked_fill(mask_k, value=torch.tensor(-1e9))
    #
    #     result = result[result[:, :, :, 0] != -1e9].reshape(bsz, 26, 18, fsize)
    #
    #
    #     return result

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c



    def forward(self, cnn_feats, region_feats, spatial_feats):
        '''
        :param cnn_feats: (batch_size, max_frames, m_feature_size + a_feature_size)
        :param region_feats: (batch_size, max_frames, num_boxes, region_feature_size)
        :param spatial_feats: (batch_size, max_frames, num_boxes, spatial_feature_size)
        :return: output of Bidirectional LSTM and embedded region features
        '''
        # 2d cnn or 3d cnn or 2d+3d cnn
        assert self.a_feature_size + self.m_feature_size == cnn_feats.size(2)
        frame_feats = cnn_feats[:, :, :self.a_feature_size].contiguous()
        i3d_feats = cnn_feats[:, :, -self.m_feature_size:].contiguous()

        # flatten parameters if use multiple gpu
        if self.use_multi_gpu:
            self.bi_lstm1.flatten_parameters()
            self.bi_lstm2.flatten_parameters()

        # frame feature embedding
        embedded_frame_feats = self.frame_feature_embed(frame_feats)
        lstm_h1, lstm_c1 = self._init_lstm_state(frame_feats)
        # bidirectional lstm encoder
        frame_feats, _ = self.bi_lstm1(embedded_frame_feats, (lstm_h1, lstm_c1))
        frame_feats = self.lstm_drop1(frame_feats)

        # i3d feature embedding
        embedded_i3d_feats = self.i3d_feature_embed(i3d_feats)
        lstm_h2, lstm_c2 = self._init_lstm_state(i3d_feats)
        # bidirectional lstm encoder
        i3d_feats, _ = self.bi_lstm2(embedded_i3d_feats, (lstm_h2, lstm_c2))
        i3d_feats = self.lstm_drop2(i3d_feats)

        # region feature embedding
        region_feats = self.region_feature_embed(region_feats)
        # spatial feature embedding
        loc_feats = self.spatial_feature_embed(spatial_feats)
        # time feature embedding
        bsz, _, num_boxes, _ = region_feats.size()
        time_feats = self.time_feats.unsqueeze(0).unsqueeze(2).repeat(bsz, 1, num_boxes, 1)
        # object feature
        object_feats = torch.cat([region_feats, loc_feats, time_feats], dim=-1)
        object_feats = self.visual_drop(torch.tanh(self.visual_embed(object_feats)))
        # object_feats = object_feats[:, :, :18, :]
        # object_feats = self.mask(object_feats)
        return frame_feats, i3d_feats, object_feats


# ------------------------------------------------------
# -------------------- LOCATE Module -------------------
# ------------------------------------------------------

class LOCATE(nn.Module):
    def __init__(self, opt):
        super(LOCATE, self).__init__()
        '''
        self.spatial_attn = SoftAttention(opt.region_projected_size, opt.hidden_size, opt.hidden_size)
        self.FSAF_module = AttentionFiltration(2024)
        '''
        '''
        @211 :(
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.objt_global_w = TextSA(1000, 0.4)
        self.OSAF_module = AttentionFiltration(1000)
        self.SCAN_attention = SCAN_attention()
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)

        feat_size = opt.region_projected_size + opt.hidden_size * 2
        self.temp_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)
        '''
        '''
        @29
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.objt_global_w = TextSA(1000, 0.4)
        self.OSGR_module = nn.ModuleList([GraphReasoning(1000) for i in range(3)])
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)

        feat_size = opt.region_projected_size + opt.hidden_size * 2
        self.temp_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)
        '''
        '''
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.framev_global_w = VisualSA(2024, 0.4, 26)
        self.objt_global_w = TextSA(1000, 0.4)
        self.framet_global_w = TextSA(2024, 0.4)
        self.OSGR_module = nn.ModuleList([GraphReasoning(1000) for i in range(3)])
        self.FSGR_module = nn.ModuleList([GraphReasoning(2024) for i in range(3)])
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.hidden_fc2 = nn.Linear(opt.hidden_size, 2024)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)
        self.sim_tranframe_w = nn.Linear(2024, 2024)
        self.sim_tranframeglo_w = nn.Linear(2024, 2024)
        '''

        # spatial soft attention module
        self.spatial_attn = SoftAttention(opt.region_projected_size, opt.hidden_size, opt.hidden_size)

        # temporal soft attention module
        feat_size = opt.region_projected_size + opt.hidden_size * 2
        self.temp_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)


    def forward(self, frame_feats, object_feats, hidden_state):
        """
        :param real_frame_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: loc_feat: (batch_size, feat_size)
        """
        '''
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        frame_feats = torch.cat([object_feats_att, real_frame_feats], dim=-1)
        frame_vec = self.FSAF_module(frame_feats)
        '''
        '''
        @211 :(
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = self.SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        obj_vec = self.OSAF_module(obj_emb)

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        feat = torch.cat([osgr_feats, real_frame_feats], dim=-1)
        loc_feat, _ = self.temp_attn(feat, hidden_state)
        '''
        '''
        @29
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        for module in self.OSGR_module:
            obj_emb = module(obj_emb)
        obj_vec = obj_emb[:, 0, :]

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        feat = torch.cat([osgr_feats, real_frame_feats], dim=-1)
        loc_feat, _ = self.temp_attn(feat, hidden_state)
        '''
        '''
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        for module in self.OSGR_module:
            obj_emb = module(obj_emb)
        obj_vec = obj_emb[:, 0, :]

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        hidden_femb = self.hidden_fc2(hidden_state.unsqueeze(1).repeat(1, max_frames, 1))
        frame_feats = torch.cat([osgr_feats, real_frame_feats], dim=-1)
        frame_ave = torch.mean(frame_feats, 1)
        frame_glo = self.framev_global_w(frame_feats, frame_ave)
        hidden_fave = torch.mean(hidden_femb, 1)
        hidden_fglo = self.framet_global_w(hidden_femb, hidden_fave)

        Context_frame = SCAN_attention(hidden_femb, frame_feats, smooth=9.0)
        sim_frame_loc = torch.pow(torch.sub(Context_frame, hidden_femb), 2)
        sim_frame_loc = l2norm(self.sim_tranframe_w(sim_frame_loc), dim=-1)

        sim_frame_glo = torch.pow(torch.sub(frame_glo, hidden_fglo), 2)
        sim_frame_glo = l2norm(self.sim_tranframeglo_w(sim_frame_glo), dim=-1)

        frame_emb = torch.cat([sim_frame_glo.unsqueeze(1), sim_frame_loc], 1)
        for module in self.FSGR_module:
            frame_emb = module(frame_emb)
        frame_vec = frame_emb[:, 0, :]
        '''


        # spatial attention
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # temporal attention
        feat = torch.cat([object_feats_att, frame_feats], dim=-1)
        loc_feat, _ = self.temp_attn(feat, hidden_state)

        return loc_feat
        # return frame_vec

# ------------------------------------------------------
# -------------------- RELATE Module -------------------
# ------------------------------------------------------

class RELATE(nn.Module):
    def __init__(self, opt):
        super(RELATE, self).__init__()
        '''
        region_feat_size = opt.region_projected_size
        self.spatial_attn = SoftAttention(region_feat_size, opt.hidden_size, opt.hidden_size)
        self.FSAF_module = AttentionFiltration(2024)
        '''
        '''
        @211-1-2 :(
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.objt_global_w = TextSA(1000, 0.4)
        self.OSAF_module = AttentionFiltration(1000)
        self.SCAN_attention = SCAN_attention()
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)

        region_feat_size = opt.region_projected_size
        feat_size = region_feat_size + opt.hidden_size * 2
        self.relation_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)
        '''
        '''
        @211 :(
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.framev_global_w = VisualSA(2024, 0.4, 26)
        self.objt_global_w = TextSA(1000, 0.4)
        self.framet_global_w = TextSA(2024, 0.4)
        self.OSAF_module = AttentionFiltration(1000)
        self.FSAF_module = AttentionFiltration(2024)
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.hidden_fc2 = nn.Linear(opt.hidden_size, 2024)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)
        self.sim_tranframe_w = nn.Linear(2024, 2024)
        self.sim_tranframeglo_w = nn.Linear(2024, 2024)
        '''
        '''
        @29
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.objt_global_w = TextSA(1000, 0.4)
        self.OSGR_module = nn.ModuleList([GraphReasoning(1000) for i in range(3)])
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)

        region_feat_size = opt.region_projected_size
        feat_size = region_feat_size + opt.hidden_size * 2
        self.relation_attn = SoftAttention(feat_size, opt.hidden_size, opt.hidden_size)
        '''
        '''
        region_feat_size = opt.region_projected_size
        self.spatial_attn = SoftAttention(region_feat_size, opt.hidden_size, opt.hidden_size)
        self.hidden_fc2 = nn.Linear(opt.hidden_size, 2024)
        self.framev_global_w = VisualSA(2024, 0.4, 26)
        self.framet_global_w = TextSA(2024, 0.4)
        self.FSGR_module = nn.ModuleList([GraphReasoning(2024) for i in range(3)])
        self.sim_tranframe_w = nn.Linear(2024, 2024)
        self.sim_tranframeglo_w = nn.Linear(2024, 2024)
        '''
        '''
        @28,210 :(
        self.objv_global_w = VisualSA(1000, 0.4, 36)
        self.framev_global_w = VisualSA(2024, 0.4, 26)
        self.objt_global_w = TextSA(1000, 0.4)
        self.framet_global_w = TextSA(2024, 0.4)
        self.OSGR_module = nn.ModuleList([GraphReasoning(1000) for i in range(3)])
        self.FSGR_module = nn.ModuleList([GraphReasoning(2024) for i in range(3)])
        self.hidden_fc1 = nn.Linear(opt.hidden_size, 1000)
        self.hidden_fc2 = nn.Linear(opt.hidden_size, 2024)
        self.sim_tranobj_w = nn.Linear(1000, 1000)
        self.sim_tranobjglo_w = nn.Linear(1000, 1000)
        self.sim_tranframe_w = nn.Linear(2024, 2024)
        self.sim_tranframeglo_w = nn.Linear(2024, 2024)
        '''

        # spatial soft attention module
        region_feat_size = opt.region_projected_size
        self.spatial_attn = SoftAttention(region_feat_size, opt.hidden_size, opt.hidden_size)

        # temporal soft attention module
        feat_size = region_feat_size + opt.hidden_size * 2
        self.relation_attn = SoftAttention(2*feat_size, opt.hidden_size, opt.hidden_size)


    def forward(self, i3d_feats, object_feats, hidden_state):
        '''
        :param i3d_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: rel_feat
        '''
        '''
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        frame_feats = torch.cat([object_feats_att, i3d_feats], dim=-1)
        frame_vec = self.FSAF_module(frame_feats)
        '''
        '''
        @211-1 :(
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = self.SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        obj_vec = self.OSAF_module(obj_emb)

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        feat = torch.cat([osgr_feats, i3d_feats], dim=-1)
        rel_feat, _ = self.relation_attn(feat, hidden_state)
        '''
        '''
        @211 :(
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        obj_vec = self.OSAF_module(obj_emb)

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        hidden_femb = self.hidden_fc2(hidden_state.unsqueeze(1).repeat(1, max_frames, 1))
        frame_feats = torch.cat([osgr_feats, i3d_feats], dim=-1)
        frame_ave = torch.mean(frame_feats, 1)
        frame_glo = self.framev_global_w(frame_feats, frame_ave)
        hidden_fave = torch.mean(hidden_femb, 1)
        hidden_fglo = self.framet_global_w(hidden_femb, hidden_fave)

        Context_frame = SCAN_attention(hidden_femb, frame_feats, smooth=9.0)
        sim_frame_loc = torch.pow(torch.sub(Context_frame, hidden_femb), 2)
        sim_frame_loc = l2norm(self.sim_tranframe_w(sim_frame_loc), dim=-1)

        sim_frame_glo = torch.pow(torch.sub(frame_glo, hidden_fglo), 2)
        sim_frame_glo = l2norm(self.sim_tranframeglo_w(sim_frame_glo), dim=-1)

        frame_emb = torch.cat([sim_frame_glo.unsqueeze(1), sim_frame_loc], 1)
        frame_vec = self.FSAF_module(frame_emb)
        '''
        '''
        @29
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        for module in self.OSGR_module:
            obj_emb = module(obj_emb)
        obj_vec = obj_emb[:, 0, :]

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        feat = torch.cat([osgr_feats, i3d_feats], dim=-1)
        rel_feat, _ = self.relation_attn(feat, hidden_state)
        '''
        '''
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        frame_feats = torch.cat([object_feats_att, i3d_feats], dim=-1)
        hidden_femb = self.hidden_fc2(hidden_state.unsqueeze(1).repeat(1, max_frames, 1))
        frame_ave = torch.mean(frame_feats, 1)
        frame_glo = self.framev_global_w(frame_feats, frame_ave)
        hidden_fave = torch.mean(hidden_femb, 1)
        hidden_fglo = self.framet_global_w(hidden_femb, hidden_fave)

        Context_frame = SCAN_attention(hidden_femb, frame_feats, smooth=9.0)
        sim_frame_loc = torch.pow(torch.sub(Context_frame, hidden_femb), 2)
        sim_frame_loc = l2norm(self.sim_tranframe_w(sim_frame_loc), dim=-1)

        sim_frame_glo = torch.pow(torch.sub(frame_glo, hidden_fglo), 2)
        sim_frame_glo = l2norm(self.sim_tranframeglo_w(sim_frame_glo), dim=-1)

        frame_emb = torch.cat([sim_frame_glo.unsqueeze(1), sim_frame_loc], 1)
        for module in self.FSGR_module:
            frame_emb = module(frame_emb)
        frame_vec = frame_emb[:, 0, :]
        '''
        '''
        @28,210 :(
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        obj_feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden_emb = self.hidden_fc1(hidden_state.unsqueeze(1).repeat(max_frames, num_boxes, 1))
        obj_ave = torch.mean(obj_feats, 1)
        obj_glo = self.objv_global_w(obj_feats, obj_ave)
        hidden_ave = torch.mean(hidden_emb, 1)
        hidden_glo = self.objt_global_w(hidden_emb, hidden_ave)

        Context_obj = SCAN_attention(hidden_emb, obj_feats, smooth=9.0)
        sim_obj_loc = torch.pow(torch.sub(Context_obj, hidden_emb), 2)
        sim_obj_loc = l2norm(self.sim_tranobj_w(sim_obj_loc), dim=-1)

        sim_obj_glo = torch.pow(torch.sub(obj_glo, hidden_glo), 2)
        sim_obj_glo = l2norm(self.sim_tranobjglo_w(sim_obj_glo), dim=-1)

        obj_emb = torch.cat([sim_obj_glo.unsqueeze(1), sim_obj_loc], 1)
        for module in self.OSGR_module:
            obj_emb = module(obj_emb)
        obj_vec = obj_emb[:, 0, :]

        osgr_feats = obj_vec.reshape(bsz, max_frames, fsize)
        hidden_femb = self.hidden_fc2(hidden_state.unsqueeze(1).repeat(1, max_frames, 1))
        frame_feats = torch.cat([osgr_feats, i3d_feats], dim=-1)
        frame_ave = torch.mean(frame_feats, 1)
        frame_glo = self.framev_global_w(frame_feats, frame_ave)
        hidden_fave = torch.mean(hidden_femb, 1)
        hidden_fglo = self.framet_global_w(hidden_femb, hidden_fave)


        Context_frame = SCAN_attention(hidden_femb, frame_feats, smooth=9.0)
        sim_frame_loc = torch.pow(torch.sub(Context_frame, hidden_femb), 2)
        sim_frame_loc = l2norm(self.sim_tranframe_w(sim_frame_loc), dim=-1)

        sim_frame_glo = torch.pow(torch.sub(frame_glo, hidden_fglo), 2)
        sim_frame_glo = l2norm(self.sim_tranframeglo_w(sim_frame_glo), dim=-1)

        frame_emb = torch.cat([sim_frame_glo.unsqueeze(1), sim_frame_loc], 1)
        for module in self.FSGR_module:
            frame_emb = module(frame_emb)
        frame_vec = frame_emb[:, 0, :]
        '''

        # spatial atttention
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        feats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = hidden_state.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _ = self.spatial_attn(feats, hidden)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # generate pair-wise feature
        feat = torch.cat([object_feats_att, i3d_feats], dim=-1)
        feat1 = feat.repeat(1, max_frames, 1)
        feat2 = feat.repeat(1, 1, max_frames).reshape(bsz, max_frames*max_frames, -1)
        pairwise_feat = torch.cat([feat1, feat2], dim=-1)

        # temporal attention
        rel_feat, _ = self.relation_attn(pairwise_feat, hidden_state)

        return rel_feat
        # return frame_vec

# ------------------------------------------------------
# -------------------- FUNC Module ---------------------
# ------------------------------------------------------

class FUNC(nn.Module):
    def __init__(self, opt):
        super(FUNC, self).__init__()
        self.cell_attn = SoftAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)

    def forward(self, cells, hidden_state):
        '''
        :param cells: previous memory states of decoder LSTM
        :param hidden_state: (batch_size, hidden_size)
        :return: func_feat
        '''
        func_feat, _ = self.cell_attn(cells, hidden_state)
        return func_feat


class VATTR(nn.Module):
    def __init__(self, opt):
        super(VATTR, self).__init__()

        #  attentionshare module # opt.hidden_size + opt.word_size
        self.query_atten = AttentionShare(opt.region_projected_size, opt.hidden_size, opt.hidden_size)

    def forward(self, object_feats, hidden_state):
        """
        :param object_feats: (batch_size, max_frames, num_boxes, region_projected_size)
        :param hidden_state: (batch_size, hidden_size)
        :return: obj_v_feats: (batch_size, hidden_size)
        """
        bsz, max_frames, num_boxes, fsize = object_feats.size()

        #  attention share
        feats = object_feats.reshape(bsz, max_frames*num_boxes, fsize)
        object_feats_att, _ = self.query_atten(feats, hidden_state)
        return object_feats_att


# ------------------------------------------------------
# ------------------- Module Selector ------------------
# ------------------------------------------------------

class ModuleSelection(nn.Module):
    def __init__(self, opt):
        super(ModuleSelection, self).__init__()
        self.use_loc = opt.use_loc
        self.use_rel = opt.use_rel
        self.use_func = opt.use_func

        if opt.use_loc:
            loc_feat_size = opt.region_projected_size + opt.hidden_size * 2
            self.loc_fc = nn.Linear(loc_feat_size, opt.hidden_size)
            nn.init.xavier_normal_(self.loc_fc.weight)

        if opt.use_rel:
            rel_feat_size = 2 * (opt.region_projected_size + 2 * opt.hidden_size)
            self.rel_fc = nn.Linear(rel_feat_size, opt.hidden_size)
            # self.rel_fc = nn.Linear(opt.region_projected_size + opt.hidden_size * 2, opt.hidden_size)
            nn.init.xavier_normal_(self.rel_fc.weight)

        if opt.use_func:
            func_size = opt.hidden_size
            self.func_fc = nn.Linear(func_size, opt.hidden_size)
            nn.init.xavier_normal_(self.func_fc.weight)

        if opt.use_loc and opt.use_rel and opt.use_func:
            if opt.attention == 'soft':
                self.module_attn = SoftAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            elif opt.attention == 'gumbel':
                self.module_attn = GumbelAttention(opt.hidden_size, opt.hidden_size, opt.hidden_size)

    def forward(self, loc_feats, rel_feats, func_feats, hidden_state):
        '''
        soft attention: Weighted sum of three features
        gumbel attention: Choose one of three features
        '''
        loc_feats = self.loc_fc(loc_feats) if self.use_loc else None
        rel_feats = self.rel_fc(rel_feats) if self.use_rel else None
        func_feats = self.func_fc(func_feats) if self.use_func else None

        if self.use_loc and self.use_rel and self.use_func:
            feats = torch.stack([loc_feats, rel_feats, func_feats], dim=1)
            feats, module_weight = self.module_attn(feats, hidden_state)

        elif self.use_loc and not self.use_rel:
            feats = loc_feats
            module_weight = torch.tensor([0.3, 0.3, 0.4]).cuda()
        elif self.use_rel and not self.use_loc:
            feats = rel_feats
            module_weight = torch.tensor([0.3, 0.3, 0.4]).cuda()

        return feats, module_weight


# ------------------------------------------------------
# --------------- Language LSTM Decoder ----------------
# ------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, opt, vocab):
        super(Decoder, self).__init__()
        self.region_projected_size = opt.region_projected_size
        self.hidden_size = opt.hidden_size
        self.word_size = opt.word_size
        self.max_words = opt.max_words
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.beam_size = opt.beam_size
        self.use_multi_gpu = opt.use_multi_gpu
        self.use_loc = opt.use_loc
        self.use_rel = opt.use_rel
        self.use_func = opt.use_func
        self.batch_size = 32

        # modules
        self.loc = LOCATE(opt)
        self.rel = RELATE(opt)
        self.func = FUNC(opt)

        # self.vatrri = VATTR(opt)
        self.module_selection = ModuleSelection(opt)

        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        self.word_drop = nn.Dropout(p=opt.dropout)

        # attention lstm
        visual_feat_size = opt.hidden_size * 4 + opt.region_projected_size
        att_insize = opt.hidden_size + opt.word_size + visual_feat_size
        self.att_lstm = nn.LSTMCell(att_insize, opt.hidden_size)
        self.att_lstm_drop = nn.Dropout(p=opt.dropout)

        # self.query_embed = nn.Linear(opt.hidden_size + opt.word_size, self.hidden_size)
        #
        # self.vattr_fc = nn.Linear(opt.region_projected_size, self.hidden_size)

        # language lstm
        self.lang_lstm = nn.LSTMCell(opt.hidden_size * 2, opt.hidden_size)
        self.lstm_drop = nn.Dropout(p=opt.dropout)

        # final output layer
        self.out_fc = nn.Linear(opt.hidden_size * 3, self.hidden_size)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.word_restore = nn.Linear(opt.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        self.beam_search = BeamSearch(vocab('<end>'), self.max_words, self.beam_size, per_node_beam_size=self.beam_size)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def forward(self, frame_feats, i3d_feats, object_feats, captions, teacher_forcing_ratio=1.0):
        self.batch_size = frame_feats.size(0)
        infer = True if captions is None else False

        # visual input of attention lstm
        global_frame_feat = torch.mean(frame_feats, dim=1)
        global_i3d_feat = torch.mean(i3d_feats, dim=1)
        global_object_feat = torch.mean(torch.mean(object_feats, dim=2), dim=1)
        global_feat = torch.cat([global_frame_feat, global_i3d_feat, global_object_feat], dim=1)

        # initialize lstm state
        lang_lstm_h, lang_lstm_c = self._init_lstm_state(global_feat)
        att_lstm_h, att_lstm_c = self._init_lstm_state(global_feat)

        # add a '<start>' sign
        start_id = self.vocab('<start>')
        start_id = global_feat.data.new(global_feat.size(0)).long().fill_(start_id)
        word = self.word_embed(start_id)
        word = self.word_drop(word)  # b*w
        # query = torch.cat([lang_lstm_h, word], dim=1)

        # training stage
        outputs = []
        previous_cells = []
        previous_cells.append(lang_lstm_c)
        module_weights = []
        if not infer or self.beam_size == 1:
            for i in range(self.max_words):
                if not infer and not self.use_multi_gpu and captions[:, i].data.sum() == 0:
                    break

                # attention lstm
                att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                                       (att_lstm_h, att_lstm_c))
                att_lstm_h = self.att_lstm_drop(att_lstm_h)
                # query = self.query_embed(torch.cat([lang_lstm_h, word], dim=1))
                # lstm decoder with attention model
                word_logits, module_weight, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                                    previous_cells, att_lstm_h, lang_lstm_h,
                                                                    lang_lstm_c)
                module_weights.append(module_weight)
                previous_cells.append(lang_lstm_c)

                # teacher_forcing: a training trick
                use_teacher_forcing = not infer and (random.random() < teacher_forcing_ratio)
                if use_teacher_forcing:
                    word_id = captions[:, i]
                else:
                    word_id = word_logits.max(1)[1]
                word = self.word_embed(word_id)
                word = self.word_drop(word)

                if infer:
                    outputs.append(word_id)
                else:
                    outputs.append(word_logits)

            outputs = torch.stack(outputs, dim=1)  # b*m*v(train) or b*m(infer)
            module_weights = torch.stack(module_weights, dim=1)
        else:
            # apply beam search if beam size > 1 during testing
            start_state = {'att_lstm_h': att_lstm_h, 'att_lstm_c': att_lstm_c, 'lang_lstm_h': lang_lstm_h,
                           'lang_lstm_c': lang_lstm_c, 'global_feat': global_feat, 'frame_feats': frame_feats,
                           'i3d_feats': i3d_feats, 'object_feats': object_feats, 'previous_cells': previous_cells}
            # start_state = {'lang_lstm_h': lang_lstm_h,
            #                'lang_lstm_c': lang_lstm_c, 'global_feat': global_feat, 'frame_feats': frame_feats,
            #                'i3d_feats': i3d_feats, 'object_feats': object_feats, 'previous_cells': previous_cells,
            #                'query': query}
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)  # b*1
            max_index = max_index.squeeze(1)  # b
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)
            module_weights = None

        return outputs, module_weights

    def beam_step(self, last_predictions, current_state):
        '''
        A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        '''
        group_size = last_predictions.size(0)  # batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            att_lstm_h = current_state['att_lstm_h'][:, i, :]
            att_lstm_c = current_state['att_lstm_c'][:, i, :]
            lang_lstm_h = current_state['lang_lstm_h'][:, i, :]
            lang_lstm_c = current_state['lang_lstm_c'][:, i, :]
            global_feat = current_state['global_feat'][:, i, :]
            frame_feats = current_state['frame_feats'][:, i, :]
            i3d_feats = current_state['i3d_feats'][:, i, :]
            object_feats = current_state['object_feats'][:, i, :]
            previous_cells = current_state['previous_cells'][:, i, :]
            # query = current_state['query'][:, i, :]

            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)
            # attention lstm
            att_lstm_h, att_lstm_c = self.att_lstm(torch.cat([lang_lstm_h, global_feat, word], dim=1),
                                                   (att_lstm_h, att_lstm_c))
            att_lstm_h = self.att_lstm_drop(att_lstm_h)
            # query = self.query_embed(torch.cat([lang_lstm_h, word], dim=1))
            # language lstm decoder
            word_logits, module_weight, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                                               previous_cells, att_lstm_h, lang_lstm_h,
                                                                               lang_lstm_c)
            previous_cells = torch.cat([previous_cells, lang_lstm_c.unsqueeze(1)], dim=1)
            # store log probabilities
            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            # update new state
            new_state['att_lstm_h'].append(att_lstm_h)
            new_state['att_lstm_c'].append(att_lstm_c)
            new_state['lang_lstm_h'].append(lang_lstm_h)
            new_state['lang_lstm_c'].append(lang_lstm_c)
            new_state['global_feat'].append(global_feat)
            new_state['frame_feats'].append(frame_feats)
            new_state['i3d_feats'].append(i3d_feats)
            new_state['object_feats'].append(object_feats)
            new_state['previous_cells'].append(previous_cells)
            # new_state['query'].append(query)


            # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0)  # (beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size)  # (batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims)  # (batch_size*beam_size, *)
        return (log_probs, new_state)

    def decode(self, frame_feats, i3d_feats, object_feats, previous_cells, att_lstm_h, lang_lstm_h, lang_lstm_c):
        if isinstance(previous_cells, list):
            previous_cells = torch.stack(previous_cells, dim=1)

        # LOCATE, RELATE, FUNC modules
        if not self.use_rel and not self.use_loc:
            raise ValueError('use locate or relation, all use both')
        loc_feats = self.loc(frame_feats, object_feats, att_lstm_h) if self.use_loc else None
        rel_feats = self.rel(i3d_feats, object_feats, att_lstm_h) if self.use_rel else None
        func_feats = self.func(previous_cells, att_lstm_h) if self.use_func else None
        feats, module_weight = self.module_selection(loc_feats, rel_feats, func_feats, att_lstm_h)

        # visual_attributes = self.vatrri(object_feats, query)
        # obj_v_feats = self.vattr_fc(visual_attributes)

        # language lstm decoder
        decoder_input = torch.cat([feats, att_lstm_h], dim=1)
        lstm_h, lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
        lstm_h = self.lstm_drop(lstm_h)
        decoder_output = torch.tanh(self.out_fc(torch.cat([lstm_h, decoder_input], dim=1)))
        word_logits = self.word_restore(decoder_output)  # b*v
        return word_logits, module_weight, lstm_h, lstm_c

    def decode_tokens(self, tokens):
        '''
        convert word index to caption
        :param tokens: input word index
        :return: capiton
        '''
        words = []
        for token in tokens:
            if token == self.vocab('<end>'):
                break
            word = self.vocab.idx2word[token]
            words.append(word)
        captions = ' '.join(words)
        return captions


# ------------------------------------------------------
# ----------------- Captioning Model -------------------
# ------------------------------------------------------

class CapModel(nn.Module):
    def __init__(self, opt, vocab):
        super(CapModel, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt, vocab)

    def forward(self, cnn_feats, region_feats, spatial_feats, captions, teacher_forcing_ratio=1.0):
        frame_feats, i3d_feats, object_feats = self.encoder(cnn_feats, region_feats, spatial_feats)
        outputs, module_weights = self.decoder(frame_feats, i3d_feats, object_feats, captions, teacher_forcing_ratio)
        return outputs, module_weights