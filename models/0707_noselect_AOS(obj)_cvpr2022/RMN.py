'''
    pytorch implementation of our RMN model
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gumbel as gumbel
from models.allennlp_beamsearch import BeamSearch
from torch.autograd import Variable
from models.decoder import Feat_Decoder
from models.transformer.Models import Encoder as PhraseEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.attention import SemanticAlignment, SemanticAttention


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
        self.visual_embed = nn.Linear(opt.region_projected_size + in_size, opt.object_projected_size)
        nn.init.xavier_normal_(self.visual_embed.weight)
        self.visual_drop = nn.Dropout(p=opt.dropout)
        #
        # self.app_linear = nn.Linear(1024, 300)
        # self.mot_linear = nn.Linear(1024, 300)
        # self.obj_linear = nn.Linear(1000, 300)

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

        # #############
        # app_outputs = self.app_linear(frame_feats)
        # mot_outputs = self.mot_linear(i3d_feats)
        # vis_feats = torch.cat([frame_feats, i3d_feats], dim=2)

        # region feature embedding
        region_feats = self.region_feature_embed(region_feats)
        # spatial feature embedding
        loc_feats = self.spatial_feature_embed(spatial_feats)
        # time feature embedding
        bsz, _, num_boxes, _ = region_feats.size()
        time_feats = self.time_feats.unsqueeze(0).unsqueeze(2).repeat(bsz, 1, num_boxes, 1)
        # object feature
        object_feats = torch.cat([region_feats, loc_feats, time_feats], dim=-1)
        # object_feats = object_feats[:, :, :18, :].contiguous()
        object_feats = self.visual_drop(torch.tanh(self.visual_embed(object_feats)))

        return frame_feats, i3d_feats, object_feats


class reasoning(nn.Module):
    def __init__(self, opt):
        super(reasoning, self).__init__()

        self.embedding_size = opt.word_size
        self.vis_feat_size = opt.feature_size
        self.sem_align_hidden_size = opt.hidden_size
        self.sem_attn_hidden_size = opt.hidden_size
        self.hidden_size = opt.hidden_size

        self.spatial_attn = SemanticAttention(opt.hidden_size, opt.hidden_size * 2, opt.hidden_size)

        self.semantic_alignment = SemanticAlignment(
            query_size=self.embedding_size,  # b,step,300
            feat_size=self.vis_feat_size,  # b,26,36,1024  or  b,26,1024
            bottleneck_size=self.sem_align_hidden_size)

        self.semantic_attention = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.embedding_size * 3,
            bottleneck_size=self.sem_attn_hidden_size)

        self.cnnembedded = nn.Linear(opt.hidden_size * 2, opt.word_size)
        nn.init.xavier_normal_(self.cnnembedded.weight)

        self.motembedded = nn.Linear(opt.hidden_size * 2, opt.word_size)
        nn.init.xavier_normal_(self.motembedded.weight)

        self.fraembedded = nn.Linear(opt.hidden_size * 4, opt.word_size)
        nn.init.xavier_normal_(self.fraembedded.weight)

    def forward(self, frame_feats, i3d_feats, object_feats, phr_feats, last_hidden, phr_masks):
        """
        :param frame_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, object_projected_size)
        :param phr_feats: (batch_size, word_size)
        :return: loc_feat: (batch_size, feat_size)
        """

        bsz, max_frames, num_boxes, fsize = object_feats.size()

        objfeats = object_feats.reshape(bsz * max_frames, num_boxes, fsize)
        hidden = last_hidden.repeat(1, max_frames).reshape(bsz * max_frames, -1)
        object_feats_att, _, _ = self.spatial_attn(hidden, objfeats, objfeats)
        object_feats_att = object_feats_att.reshape(bsz, max_frames, fsize)

        # obj aligned
        semantic_obj_feats, semantic_obj_weights, semantic_obj_logits = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=object_feats_att)

        phr_feats = self.cnnembedded(semantic_obj_feats)

        semantic_i3d_feats, _, _ = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=i3d_feats)

        phr_motfeats = self.motembedded(semantic_i3d_feats)

        semantic_frameobj_feats, _, _ = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=frame_feats)

        semantic_framemot_feats, _, _ = self.semantic_alignment(
            phr_feats=phr_motfeats,
            vis_feats=frame_feats)

        phr_framefeats = self.fraembedded(
            torch.cat([semantic_frameobj_feats, semantic_framemot_feats], dim=-1))

        semantic_group_feats = torch.cat([phr_framefeats, phr_motfeats, phr_feats], dim=2)

        feat, semantic_attn_weights, semantic_attn_logits = self.semantic_attention(
            query=last_hidden,
            keys=semantic_group_feats,
            values=semantic_group_feats,
            masks=phr_masks)

        return feat


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

        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        self.word_drop = nn.Dropout(p=opt.dropout)

        # language lstm
        self.lang_lstm = nn.LSTMCell(opt.word_size * 4, opt.hidden_size)
        self.lstm_drop = nn.Dropout(p=opt.dropout)

        # final output layer
        self.out_fc = nn.Linear(opt.word_size * 4 + opt.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.word_restore = nn.Linear(opt.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        self.reasoning = reasoning(opt)

        self.phr_encoder = PhraseEncoder(
            len_max_seq=opt.max_words,
            d_word_vec=opt.word_size,
            n_layers=1,
            n_head=1,
            d_k=32,
            d_v=32,
            d_model=opt.word_size,
            d_inner=opt.hidden_size,
            dropout=0.1)

    def get_rnn_init_hidden(self, batch_size, hidden_size):
        return (
            torch.zeros(batch_size, hidden_size).cuda(),
            torch.zeros(batch_size, hidden_size).cuda())

    def phr_mask(self, step, phr_attns, threshod):
        self.batch_size = phr_attns.size(0)
        if step == 0:
            phr_masks = torch.zeros(self.batch_size, 1).to(DEVICE)
        else:
            A = torch.bmm(phr_attns, phr_attns.transpose(1, 2))
            A_mask = torch.eye(step, step).cuda().type(torch.bool)
            A.masked_fill_(A_mask, 0)
            A_sum = A.sum(dim=2)

            indices = (A >= threshod).nonzero()  # Obtain indices of phrase pairs that
            # are highly overlapped with each other
            indices = indices[
                indices[:, 1] < indices[:, 2]]  # Leave only the upper triangle to prevent duplication
            phr_masks = torch.zeros_like(A_sum)
            if len(indices) > 0:
                redundancy_masks = torch.zeros_like(phr_masks).long()
                indices_b = indices[:, 0]
                indices_i = indices[:, 1]
                indices_j = indices[:, 2]
                indices_ij = torch.stack((indices_i, indices_j), dim=1)
                A_sum_i = A_sum[indices_b, indices_i]
                A_sum_j = A_sum[indices_b, indices_j]
                A_sum_ij = torch.stack((A_sum_i, A_sum_j), dim=1)
                _, i_or_j = A_sum_ij.max(dim=1)
                i_or_j = i_or_j.type(torch.bool)
                indices_i_or_j = torch.zeros_like(indices_b)
                indices_i_or_j[i_or_j] = indices_j[i_or_j]
                indices_i_or_j[~i_or_j] = indices_i[~i_or_j]
                redundancy_masks[indices_b, indices_i_or_j] = 1  # Mask phrases that are more redundant
                # than their counterpart
                phr_masks = redundancy_masks > 0.5

        return phr_masks

    def forward(self, frame_feats, i3d_feats, object_feats, captions, teacher_forcing_ratio=1.0):
        infer = True if captions is None else False
        batch_size = frame_feats.size(0)
        caption_EOS_table = captions == self.vocab.word2idx['<end>']
        caption_PAD_table = captions == self.vocab.word2idx['<pad>']
        caption_end_table = ~(~caption_EOS_table * ~caption_PAD_table)

        lang_lstm_h, lang_lstm_c = self.get_rnn_init_hidden(batch_size, 512)
        # outputs = Variable(torch.zeros(self.max_words, batch_size, vocab_size)).cuda()
        outputs = []
        caption_lens = torch.zeros(batch_size).cuda().long()
        output = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<start>']))
        if not infer or self.beam_size == 1:
            for t in range(0, self.max_words):
                embedded = self.word_embed(output.view(1, -1)).squeeze(0)
                if t == 0:
                    embedded_list = embedded[:, None, :]
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                elif t == 1:
                    embedded_list = embedded[:, None, :]
                    caption_lens += 1
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                else:
                    embedded_list = torch.cat([embedded_list, embedded[:, None, :]], dim=1)
                    caption_lens += ((output.long().squeeze() != self.vocab.word2idx['<pad>']) * \
                                     (output.long().squeeze() != self.vocab.word2idx['<end>'])).long()
                    src_pos = torch.arange(1, t + 1).repeat(batch_size, 1).cuda()
                    src_pos[src_pos > caption_lens[:, None]] = 0
                phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
                phr_attns = phr_attns[0]

                phr_masks = self.phr_mask(t, phr_attns, 0.2)

                # output, hidden, (sem_align_weights, _), (sem_align_logits, _) = self.decoder(
                #    embedded, hidden, vis_feats, phr_feats, phr_masks)
                output, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats, lang_lstm_h,
                                                               lang_lstm_c,
                                                               embedded, phr_feats, phr_masks)

                # Choose the next word
                # outputs[t] = output

                top1 = output.data.max(1)[1]

                if infer:
                    outputs.append(top1)
                else:
                    outputs.append(output)

                is_teacher = not infer and (random.random() < teacher_forcing_ratio)
                output = Variable(captions.data[t] if is_teacher else top1).cuda()

                # Early stop
                if not infer and torch.all(caption_end_table[t]).item():
                    break

            outputs = torch.stack(outputs, dim=1)

        else:
            outputs = self.beam_search(batch_size, self.vocab_size, frame_feats, i3d_feats, object_feats,
                                       self.beam_size)
        return outputs

    def decode(self, frame_feats, i3d_feats, object_feats, lang_lstm_h, lang_lstm_c, embedded, phr_feats, phr_masks):

        feats = self.reasoning(frame_feats, i3d_feats, object_feats, phr_feats, lang_lstm_h, phr_masks)

        # language lstm decoder
        decoder_input = torch.cat([feats, embedded], dim=1)
        lstm_h, lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
        lstm_h = self.lstm_drop(lstm_h)
        decoder_output = torch.tanh(self.out_fc(torch.cat([lstm_h, decoder_input], dim=1)))
        word_logits = self.word_restore(decoder_output)  # b*v
        word_logits = torch.log_softmax(word_logits, dim=1)

        return word_logits, lstm_h, lstm_c

    def beam_search(self, batch_size, vocab_size, frame_feats, i3d_feats, object_feats, width):
        hidden = self.get_rnn_init_hidden(batch_size, 512)

        input_list = [torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<start>'])]
        hidden_list = [hidden]
        cum_prob_list = [torch.ones(batch_size).cuda()]
        cum_prob_list = [torch.log(cum_prob) for cum_prob in cum_prob_list]
        EOS_idx = self.vocab.word2idx['<end>']

        output_list = [[[]] for _ in range(batch_size)]
        for t in range(0, self.max_words):
            beam_output_list = []
            normalized_beam_output_list = []
            beam_hidden_list = ([], [])
            next_output_list = [[] for _ in range(batch_size)]

            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                caption_list = [output_list[b][i] for b in range(batch_size)]
                if t == 0:
                    words_list = input.transpose(0, 1)
                else:
                    words_list = torch.cuda.LongTensor(caption_list)

                embedded_list = self.word_embed(words_list)
                if t == 0:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                elif t == 1:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                else:
                    caption_lens = torch.cuda.LongTensor([[idx.item() for idx in caption].index(EOS_idx) if EOS_idx in [
                        idx.item() for idx in caption] else t for caption in caption_list])
                    src_pos = torch.arange(1, t + 1).repeat(batch_size, 1).cuda()
                    src_pos[src_pos > caption_lens[:, None]] = 0
                phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
                phr_attns = phr_attns[0]

                phr_masks = self.phr_mask(t, phr_attns, 0.2)

                embedded = self.word_embed(input.view(1, -1)).squeeze(0)
                # output, next_hidden, _, _ = self.decoder(embedded, hidden, vis_feats, phr_feats, phr_masks)

                output, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                               hidden[0], hidden[1],
                                                               embedded, phr_feats, phr_masks)
                EOS_mask = [1 if EOS_idx in [idx.item() for idx in caption] else 0 for caption in caption_list]
                EOS_mask = torch.cuda.BoolTensor(EOS_mask)
                output[EOS_mask] = 0.

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [[idx.item() for idx in caption].index(EOS_idx) + 1 if EOS_idx in [idx.item() for idx in
                                                                                                  caption] else t + 1
                                for caption in caption_list]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** 1.6) / ((5 + 1) ** 1.6)
                normalized_output = output / normalizing_factor[:, None]
                normalized_beam_output_list.append(normalized_output)
                beam_hidden_list[0].append(lang_lstm_h)
                beam_hidden_list[1].append(lang_lstm_c)
            beam_output_list = torch.cat(beam_output_list, dim=1)  # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:,
                                          :width]  # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size  # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size  # ( 100, width )

            topk_output_list = [topk_output_index[:, i] for i in range(width)]  # width * ( 100, )
            topk_hidden_list = (
                [[] for _ in range(width)],
                [[] for _ in range(width)])  # 2 * width * (1, 100, 512)
            topk_cum_prob_list = [[] for _ in range(width)]  # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    topk_hidden_list[0][k].append(beam_hidden_list[0][bi][i, :])
                    topk_hidden_list[1][k].append(beam_hidden_list[1][bi][i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [oi])
            output_list = next_output_list

            input_list = [topk_output.unsqueeze(0) for topk_output in topk_output_list]  # width * ( 1, 100 )
            hidden_list = (
                [torch.stack(topk_hidden, dim=0) for topk_hidden in topk_hidden_list[0]],
                [torch.stack(topk_hidden, dim=0) for topk_hidden in
                 topk_hidden_list[1]])  # 2 * width * ( 1, 100, 512 )
            hidden_list = [(hidden, context) for hidden, context in zip(*hidden_list)]
            cum_prob_list = [torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in
                             topk_cum_prob_list]  # width * ( 100, )

        SOS_idx = self.vocab.word2idx['<start>']
        outputs = [[SOS_idx] + o[0] for o in output_list]
        return outputs

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
        outputs = self.decoder(frame_feats, i3d_feats, object_feats, captions, teacher_forcing_ratio)
        return outputs
