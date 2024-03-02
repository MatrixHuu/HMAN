'''
    pytorch implementation of our RMN model
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gumbel as gumbel
from models.allennlp_beamsearch import BeamSearch
from models.transformer.Models import Encoder as PhraseEncoder
from models.decoder import Feat_Decoder
from models.attention import SemanticAlignment, SemanticAttention
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        object_feats = object_feats[:,:,:18,:].contiguous()
        object_feats = self.visual_drop(torch.tanh(self.visual_embed(object_feats)))

        return frame_feats, i3d_feats, object_feats


# ------------------------------------------------------
# -------------------- LOCATE Module -------------------
# ------------------------------------------------------

class reasoning(nn.Module):
    def __init__(self, opt):
        super(reasoning, self).__init__()

        self.embedding_size = opt.word_size
        self.vis_feat_size = opt.feature_size
        self.sem_align_hidden_size = opt.hidden_size
        self.sem_attn_hidden_size = opt.hidden_size
        self.hidden_size = opt.hidden_size

        self.semantic_alignment = SemanticAlignment(
            query_size=self.embedding_size,  # b,step,300
            feat_size=self.vis_feat_size,    # b,26,36,1024  or  b,26,1024
            bottleneck_size=self.sem_align_hidden_size)

        self.semantic_attention = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.embedding_size + 2*self.vis_feat_size,
            bottleneck_size=self.sem_attn_hidden_size)

        self.cnnembedded = nn.Linear(opt.hidden_size*4, opt.hidden_size*2)

    def forward(self, frame_feats, motion_feats, object_feats, phr_feats, last_hidden, phr_masks):
        """
        :param frame_feats: (batch_size, max_frames, 2*hidden_size)
        :param object_feats: (batch_size, max_frames, num_boxes, object_projected_size)
        :param phr_feats: (batch_size, word_size)
        :return: loc_feat: (batch_size, feat_size)
        """
        # obj aligned
        bsz, max_frames, num_boxes, fsize = object_feats.size()
        all_obj = object_feats.view(bsz, max_frames * num_boxes, -1)
        semantic_obj_feats, semantic_obj_weights, semantic_obj_logits = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=all_obj)
        cnn_feats = torch.cat([frame_feats, motion_feats], dim=2)
        cnn_feats = self.cnnembedded(cnn_feats)
        semantic_cnn_feats, semantic_cnn_weights, semantic_cnn_logits = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=cnn_feats)

        semantic_group_feats = torch.cat([semantic_obj_feats, semantic_cnn_feats, phr_feats], dim=2)

        feat, semantic_attn_weights, semantic_attn_logits = self.semantic_attention(
            query=last_hidden,
            keys=semantic_group_feats,
            values=semantic_group_feats,
            masks=phr_masks)

        return feat   # b, 1500


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
        self.batch_size = 32

        # modules
        self.reasoning = reasoning(opt)

        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        self.word_drop = nn.Dropout(p=opt.dropout)

        # attention lstm
        visual_feat_size = opt.hidden_size * 4 + opt.region_projected_size
        att_insize = opt.hidden_size + opt.word_size + visual_feat_size
        self.att_lstm = nn.LSTMCell(att_insize, opt.hidden_size)
        self.att_lstm_drop = nn.Dropout(p=opt.dropout)

        # language lstm
        self.lang_lstm = nn.LSTMCell(opt.word_size*2 + opt.hidden_size*4, opt.hidden_size)
        self.lstm_drop = nn.Dropout(p=opt.dropout)

        # final output layer
        self.out_fc = nn.Linear(opt.hidden_size * 5 + opt.word_size*2, self.hidden_size)
        nn.init.xavier_normal_(self.out_fc.weight)
        self.word_restore = nn.Linear(opt.hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        # beam search: The BeamSearch class is imported from Allennlp
        # DOCUMENTS: https://docs.allennlp.org/master/api/nn/beam_search/
        self.beam_search = BeamSearch(vocab('<end>'), self.max_words, self.beam_size, per_node_beam_size=self.beam_size)

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


    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

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
        self.batch_size = frame_feats.size(0)
        self.PS_threshold = 0.2

        infer = True if captions is None else False


        dim2_feats = torch.mean(frame_feats,dim=1)
        # initialize lstm state
        lang_lstm_h, lang_lstm_c = self._init_lstm_state(dim2_feats)

        # add a '<start>' sign
        start_id = self.vocab('<start>')
        start_id = dim2_feats.data.new(dim2_feats.size(0)).long().fill_(start_id)
        word = self.word_embed(start_id)
        word = self.word_drop(word)  # b*w

        # training stage
        outputs = []

        caption_lens = torch.zeros(self.batch_size).cuda().long()
        phr_feats = word[:, None, :]

        phr_masks = torch.zeros(self.batch_size, 1).to(DEVICE)
        if not infer or self.beam_size == 1:
            for i in range(self.max_words):
                if not infer and not self.use_multi_gpu and captions[:, i].data.sum() == 0:
                    break

                if i == 0:
                    embedded_list = word[:, None, :]
                    src_pos = torch.arange(1, 2).repeat(self.batch_size, 1).cuda()
                elif i == 1:
                    embedded_list = word[:, None, :]
                    caption_lens += 1
                    src_pos = torch.arange(1, 2).repeat(self.batch_size, 1).cuda()
                else:
                    embedded_list = torch.cat([embedded_list, word[:, None, :]], dim=1)
                    caption_lens += ((word_id.long().squeeze() != self.vocab.word2idx['<pad>']) * \
                                     (word_id.long().squeeze() != self.vocab.word2idx['<end>'])).long()
                    src_pos = torch.arange(1, i+1).repeat(self.batch_size, 1).cuda()
                    src_pos[src_pos > caption_lens[:, None]] = 0

                # attention transformer
                phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
                phr_attns = phr_attns[0]

                phr_masks = self.phr_mask(i, phr_attns, self.PS_threshold)

                # lstm decoder with attention model
                word_logits, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                                    lang_lstm_h, lang_lstm_c,
                                                                    word, phr_feats, phr_masks)


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

        else:
            # apply beam search if beam size > 1 during testing
            start_state = {'lang_lstm_h': lang_lstm_h, 'lang_lstm_c': lang_lstm_c,
                           'frame_feats': frame_feats,'i3d_feats': i3d_feats,
                           'object_feats': object_feats, 'phr_feats': phr_feats, 'phr_masks': phr_masks}

            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)  # b*1
            max_index = max_index.squeeze(1)  # b
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)


        return outputs

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
        caption_lens = torch.zeros(self.batch_size).cuda().long()
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            if isinstance(state, list):
                state = torch.stack(state, dim=1)
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            lang_lstm_h = current_state['lang_lstm_h'][:, i, :]
            lang_lstm_c = current_state['lang_lstm_c'][:, i, :]
            frame_feats = current_state['frame_feats'][:, i, :]
            i3d_feats = current_state['i3d_feats'][:, i, :]
            object_feats = current_state['object_feats'][:, i, :]
            phr_feats = current_state['phr_feats'][:, i, :]
            phr_masks = current_state['phr_masks'][:, i, :]


            # decoding stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)

            if i == 0:
                embedded_list = word[:, None, :]
                src_pos = torch.arange(1, 2).repeat(self.batch_size, 1).cuda()
            elif i == 1:
                embedded_list = word[:, None, :]
                caption_lens += 1
                src_pos = torch.arange(1, 2).repeat(self.batch_size, 1).cuda()
            else:
                embedded_list = torch.cat([embedded_list, word[:, None, :]], dim=1)
                caption_lens += ((word_id.long().squeeze() != self.vocab.word2idx['<pad>']) * \
                                 (word_id.long().squeeze() != self.vocab.word2idx['<end>'])).long()
                src_pos = torch.arange(1, i+1).repeat(self.batch_size, 1).cuda()
                src_pos[src_pos > caption_lens[:, None]] = 0

            # attention transformer
            phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
            phr_attns = phr_attns[0]

            phr_masks = self.phr_mask(i, phr_attns, self.PS_threshold)

            # language lstm decoder
            word_logits, lang_lstm_h, lang_lstm_c = self.decode(frame_feats, i3d_feats, object_feats,
                                                                               lang_lstm_h, lang_lstm_c,
                                                                               word, phr_feats, phr_masks)

            # store log probabilities
            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            # update new state

            new_state['lang_lstm_h'].append(lang_lstm_h)
            new_state['lang_lstm_c'].append(lang_lstm_c)
            new_state['frame_feats'].append(frame_feats)
            new_state['i3d_feats'].append(i3d_feats)
            new_state['object_feats'].append(object_feats)
            new_state['phr_feats'].append(phr_feats)
            new_state['phr_masks'].append(phr_masks)


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

    def decode(self, frame_feats, i3d_feats, object_feats, lang_lstm_h, lang_lstm_c, word, phr_feats, phr_masks):


        feats = self.reasoning( frame_feats, i3d_feats, object_feats, phr_feats, lang_lstm_h, phr_masks)

        # language lstm decoder
        decoder_input = torch.cat([feats, word], dim=1)
        lstm_h, lstm_c = self.lang_lstm(decoder_input, (lang_lstm_h, lang_lstm_c))
        lstm_h = self.lstm_drop(lstm_h)
        decoder_output = torch.tanh(self.out_fc(torch.cat([lstm_h, decoder_input], dim=1)))
        word_logits = self.word_restore(decoder_output)  # b*v
        word_logits = torch.log_softmax(word_logits, dim=1)
        return word_logits, lstm_h, lstm_c

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