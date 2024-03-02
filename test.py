# -*-coding:utf-8-*-
# coding:utf-8
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import sys

import pickle
import numpy as np
import time
import myopts
from collections import OrderedDict
import random
import h5py
import torch.utils.data as data
from utils.opt import parse_opt

import shutil
import pickle
import time
import random

from utils.utils import *
from utils.data import get_train_loader
from utils.opt import parse_opt
import models
import torch
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format
from tensorboard_logger import configure, log_value


def load_pkl(pkl_file):
    f = open(pkl_file, 'rb')
    try:
        result = pickle.load(f)
    finally:
        f.close()
    return result


def get_sub_frames(frames, K):
    # from all frames, take K of them, then add end of video frame
    if len(frames) < K:
        # frames_ = np.zeros([K, frames.shape[1]])
        # frames_[:len(frames),:] = frames
        temp_zeros = np.zeros([K - frames.shape[0], frames.shape[1]])
        frames_ = np.concatenate((frames, temp_zeros), axis=0)
    else:
        index = np.linspace(0, len(frames), K, endpoint=False, dtype=int)
        frames_ = frames[index]
    return frames_


def filt_word_category(cate_pkl, words):
    # load the category file
    category_words = load_pkl(cate_pkl)  # {NN:[cat, dog, pig]}
    # make word and category conpends  {cat:NN, take:VB, ...}
    words_category = {}
    for category, wordlist in category_words.items():
        for word in wordlist:
            words_category[word] = category
    # give each category a ID
    category_name_un = ['FW', '-LRB-', '-RRB-', 'LS']  # 1不明白
    category_name_vb = ['VB', 'VBD', 'VBP', 'VBG', 'VBN', 'VBZ']  # 2动词
    category_name_nn = ['NN', 'NNS', 'NNP']  # 3名词
    category_name_jj = ['JJ', 'JJR', 'JJS']  # 4形容词
    category_name_rb = ['RB', 'RBS', 'RBR', 'WRB', 'EX']  # 5副词
    category_name_cc = ['CC']  # 6连词
    category_name_pr = ['PRP', 'PRP$', 'WP', 'POS', 'WP$']  # 7代词
    category_name_in = ['IN', 'TO']  # 8介词
    category_name_dt = ['DT', 'WDT', 'PDT']  # 9冠词
    category_name_rp = ['RP', 'MD']  # 10助词
    category_name_cd = ['CD']  # 11数字
    category_name_sy = ['SYM', ':', '``', '#', '$']  # 12符号
    category_name_uh = ['UH']  # 13叹词

    all_category = category_words.keys()
    category_id = {}  # {VB:2, VBS:2, NN:3, NNS:3 ...}
    for category in all_category:
        if category in category_name_vb:
            category_id[category] = 2
        elif category in category_name_nn:
            category_id[category] = 3
        elif category in category_name_jj:
            category_id[category] = 4
        elif category in category_name_rb:
            category_id[category] = 5
        elif category in category_name_cc:
            category_id[category] = 6
        elif category in category_name_pr:
            category_id[category] = 7
        elif category in category_name_in:
            category_id[category] = 8
        elif category in category_name_dt:
            category_id[category] = 9
        elif category in category_name_rp:
            category_id[category] = 10
        elif category in category_name_cd:
            category_id[category] = 11
        elif category in category_name_sy:
            category_id[category] = 12
        elif category in category_name_uh:
            category_id[category] = 13
        else:
            category_id[category] = 1
    # turn words' category from str to ID
    all_words_in_category = words_category.keys()
    filted_words_categoryid = {}  # {'<EOS>':0, '<UNK>':1, 'cat':3, 'take':2, 'log_vir':1}
    for key in words:
        if key in all_words_in_category:
            the_key_category = words_category[key]
            filted_words_categoryid[key] = category_id[the_key_category]
        else:
            filted_words_categoryid[key] = 1
    filted_words_categoryid['<end>'] = 0
    filted_words_categoryid['<pad>'] = 0
    filted_words_categoryid['<unk'] = 1
    # take out the unmasked category ids
    unmasked_categoryid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # VB, NN, JJ, and RB needn't be masked
    return filted_words_categoryid, words_category, category_id, category_words, unmasked_categoryid


class V2TDataset(data.Dataset):
    def __init__(self, cap_pkl, cate_pkl, frame_feature_h5, region_feature_h5, msvd_vocab):
        with open(cap_pkl, 'rb') as f:
            self.captions, _, self.lengths, self.video_ids = pickle.load(f)
        i2word = []
        for i in msvd_vocab.idx2word:
            if i in ['<pad>', '<start>', '<end>', '<unk>']:
                continue
            i2word.append(i)
        filted_class, words_class, class_id, class_words, unmasked_classid = filt_word_category(cate_pkl, i2word)
        self.category = filted_class
        category_keys = self.category.keys()
        pos_tags = []
        # class_mask = []
        for _, cap in enumerate(self.captions):
            dict_ = []
            cls_mask = {}
            for i in cap:
                token = msvd_vocab.idx2word[i]  # 是否pad：填充为0
                if token in category_keys:
                    dict_.append(self.category[token])

            # cls_mask = [1 if index in unmasked_classid else 0 for index in dict_]
            pos_tags.append(torch.from_numpy(np.array(dict_)))
            # class_mask.append(torch.from_numpy(np.array(cls_mask)))
        self.pos_tags = pos_tags
        # self.class_mask = class_mask

        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        h5 = h5py.File(region_feature_h5, 'r')
        self.region_feats = h5[opt.region_visual_feats]
        self.spatial_feats = h5[opt.region_spatial_feats]

    def __getitem__(self, index):
        caption = self.captions[index]
        pos_tag = self.pos_tags[index]
        length = self.lengths[index]
        video_id = self.video_ids[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        region_feat = torch.from_numpy(self.region_feats[video_id])
        spatial_feat = torch.from_numpy(self.spatial_feats[video_id])
        return video_feat, region_feat, spatial_feat, caption, pos_tag, length, video_id

    def __len__(self):
        return len(self.captions)


class custom_dset_train(Dataset):
    def get_itow(self):

        wtoi = self.wtoi

        itow = {}
        for key, val in wtoi.iteritems():
            itow[val] = key
        return itow

    def __init__(self, train_pkl, cap_pkl, cate_pkl, feat_path1, feat_path2, wtoi_path, nwords=10000, K=28, opt=None):
        self.nwords = nwords
        self.K = K
        data_name_list = load_pkl(train_pkl)  # [vid1_0,vid1_2, ...]
        caps = load_pkl(cap_pkl)
        wtoi = load_pkl(wtoi_path)
        wtoi['<EOS>'] = 0
        wtoi['UNK'] = 1  # because 'wtoi_path' start from 2.
        wtoi_keys = wtoi.keys()
        self.wtoi = wtoi
        filted_class, words_class, class_id, class_words, unmasked_classid = filt_word_category(cate_pkl, wtoi)
        self.category = filted_class
        category_keys = self.category.keys()

        temp_cap_list = []
        for i, ID in enumerate(data_name_list):
            vidid, capid = ID.split('_')  # vidid='vid1', capid=0
            temp_cap_list.append(caps[vidid][int(capid)])

        data_list = []
        cap_list = []
        for data, cap in zip(data_name_list, temp_cap_list):
            token = cap['tokenized'].split()
            if 0 < len(token) <= opt.seq_length:
                data_list.append(data)
                new_cap = {}
                # new_cap['image_id'] = cap['image_id']
                new_cap['caption'] = cap['caption']
                new_cap['tokenized'] = cap['tokenized']
                new_cap['numbered'] = [wtoi[w] if w in wtoi_keys else 1 for w in token]
                new_cap['category'] = [self.category[w] if w in category_keys else 1 for w in token]
                new_cap['category_mask'] = [1 if index in unmasked_classid else 0 for index in new_cap['category']]
                cap_list.append(new_cap)

        gts_list = []
        for i, ID in enumerate(data_list):
            sub_gts_list = []
            vidid, _ = ID.split('_')
            for cap in caps[vidid]:
                token = cap['tokenized'].split()
                numbered = [wtoi[w] if w in wtoi_keys else 1 for w in token]
                sub_gts_list.append(numbered)
            sub_gts_list.sort(key=lambda x: len(x), reverse=True)
            tmp_gts_arr = np.zeros([len(sub_gts_list), len(sub_gts_list[0])], dtype=int)
            for x in range(len(sub_gts_list)):
                tmp_gts_arr[x, :len(sub_gts_list[x])] = sub_gts_list[x]
            gts_list.append(tmp_gts_arr)

        self.data_list = data_list  # [vid1_0,vid1_2, ...]
        self.cap_list = cap_list  # [{},{},...]
        self.gts_list = gts_list  # [[str,str,...],...]
        self.feat_path1 = feat_path1
        self.feat_path2 = feat_path2
        print('got %d data and %d labels' % (len(self.data_list), len(self.cap_list)))

    def __getitem__(self, index):
        data = self.data_list[index]
        cap = self.cap_list[index]['numbered']
        cap_class = self.cap_list[index]['category']
        class_mask = self.cap_list[index]['category_mask']
        gts = self.gts_list[index]

        # feat = np.load(self.feat_path +'train/' + data.split('_')[0] + '.npy')
        feat1 = self.feat_path1[data.split('_')[0]][:]
        feat1 = get_sub_frames(feat1, self.K)
        feat1 = torch.from_numpy(feat1).float()  # turn numpy data to Tensor

        feat2 = self.feat_path2[data.split('_')[0]][:]
        feat2 = get_sub_frames(feat2, self.K)
        feat2 = torch.from_numpy(feat2).float()
        # feat_mask = (torch.sum(feat, dim=1, keepdim=True) != 0).float().transpose(1,0) # for fc features
        feat_mask = (torch.sum(feat1.view(feat1.size(0), -1), dim=1, keepdim=True) != 0).float().transpose(1, 0)

        return data, cap, cap_class, class_mask, feat1, feat2, feat_mask, gts

    def __len__(self):
        return len(self.cap_list)


if __name__ == '__main__':
    opt = parse_opt()
    cap_pkl = '/home/valca509/Mark_work/RMN/data/MSVD/msvd_captions_train.pkl'
    cate_pkl = '/home/valca509/Mark_work/RMN/data/MSVD/category.pkl'
    frame_feature_h5 = '/home/valca509/Mark_work/RMN/data/MSVD/msvd_features.h5'
    region_feature_h5 = '/home/valca509/Mark_work/RMN/data/MSVD/msvd_region_feature.h5'
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    v2t = V2TDataset(cap_pkl, cate_pkl, frame_feature_h5, region_feature_h5, vocab)
