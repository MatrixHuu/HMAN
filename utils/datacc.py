import pickle
import h5py
import torch
import torch.utils.data as data
import numpy as np
from utils.opt import parse_opt

opt = parse_opt()

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


class VideoDataset(data.Dataset):
    def __init__(self, eval_range, frame_feature_h5, region_feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5 = h5py.File(frame_feature_h5, 'r')
        self.video_feats = h5[opt.feature_h5_feats]
        h5 = h5py.File(region_feature_h5, 'r')
        self.region_feats = h5[opt.region_visual_feats]
        self.spatial_feats = h5[opt.region_spatial_feats]

    def __getitem__(self, index):
        video_id = self.eval_list[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        region_feat = torch.from_numpy(self.region_feats[video_id])
        spatial_feat = torch.from_numpy(self.spatial_feats[video_id])
        return video_feat, region_feat, spatial_feat, video_id

    def __len__(self):
        return len(self.eval_list)


def train_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, regions, spatials, captions, pos_tags, lengths, video_ids = zip(*data)

    videos = torch.stack(videos, 0)
    regions = torch.stack(regions, 0)
    spatials =torch.stack(spatials, 0)

    captions = torch.stack(captions, 0)
    pos_tags = torch.stack(pos_tags, 0)
    return videos, regions, spatials, captions, pos_tags, lengths, video_ids


def eval_collate_fn(data):
    data.sort(key=lambda x: x[-1], reverse=False)

    videos, regions, spatials, video_ids = zip(*data)

    videos = torch.stack(videos, 0)
    regions = torch.stack(regions, 0)
    spatials = torch.stack(spatials, 0)

    return videos, regions, spatials, video_ids


def get_train_loader(cap_pkl, frame_feature_h5, region_feature_h5, batch_size=100, shuffle=True, num_workers=3, pin_memory=True):
    v2t = V2TDataset(cap_pkl, frame_feature_h5, region_feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory,
                                              drop_last=True)
    return data_loader


def get_eval_loader(cap_pkl, frame_feature_h5, region_feature_h5, batch_size=100, shuffle=False, num_workers=1, pin_memory=False):
    vd = VideoDataset(cap_pkl, frame_feature_h5, region_feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory,
                                              drop_last=True)
    return data_loader


if __name__ == '__main__':
    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size())
    print(d[1].size())
    print(d[2].size())
    print(d[3].size())
    print(len(d[4]))
    print(d[5])
