import sys
import os
sys.path.insert(0, '../caption-eval')
import torch
import pickle
import models
from utils.utils import Vocabulary
# from utils.data import get_eval_loader
from HMN.utils.build_loaders import get_eval_loader
from cocoeval import COCOScorer, suppress_stdout_stderr
from utils.opt import parse_opt
from tqdm import tqdm
from HMN.configs.settings import get_settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_data_to_coco_scorer_format(reference):
    reference_json = {}
    non_ascii_count = 0
    with open(reference, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            try:
                sent.encode('ascii', 'ignore').decode('ascii')
            except UnicodeDecodeError:
                non_ascii_count += 1
                continue
            if vid in reference_json:
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
            else:
                reference_json[vid] = []
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print("=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20)
    return reference_json

def convert_prediction(prediction):
    prediction_json = {}
    with open(prediction, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            prediction_json[vid] = [{u'video_id': vid, u'caption': sent}]
    return prediction_json

def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence

def evaluate(opt, net, eval_range, prediction_txt_path, reference, vocab, test=False):
    cfgs = get_settings()
    eval_loader = get_eval_loader(cfgs)

    result = {}
    for i, (feature2ds, feature3ds, objects, object_masks, captions, cap_lens, nouns, video_ids) in tqdm(enumerate(eval_loader)):
        frames = feature2ds.to(DEVICE)
        motion = feature3ds.to(DEVICE)
        objects = objects.to(DEVICE)
        object_masks = object_masks.to(DEVICE)

        objects_pending, outputs = net(frames, motion, objects, object_masks, None)
        if test:
            for (tokens, vid) in zip(outputs, video_ids):
                s = net.decoder.decode_tokens(tokens.data)
                result[vid] = s
        else:
            captions = [idxs_to_sentence(caption, vocab.idx2word, vocab.word2idx['<end>']) for caption in outputs]
            for (s, vid) in zip(captions, video_ids):
                result[vid] = s

    with open(prediction_txt_path, 'w') as f:
        for vid, s in result.items():
            f.write('%s\t%s\n' % (vid, s))

    prediction_json = convert_prediction(prediction_txt_path)

    # compute scores
    scorer = COCOScorer()

    # TODO
    file_path = '/home/valca3090/Mark/HMAN/viewData/vidname_list.pkl'

    # 使用pickle模块加载.pkl文件
    with open(file_path, 'rb') as file:
        name_to_id = pickle.load(file)

    with open('HMN/data/MSVD/MSVD_splits/MSVD_test_list.pkl', 'rb') as f:
        video_ids = pickle.load(f)

    new_dict = {}
    for key in video_ids:
        # 访问字典的键，修改值
        new_dict[name_to_id[key]] = prediction_json[key]

    with suppress_stdout_stderr():
        scores, sub_category_score = scorer.score(reference, new_dict, new_dict.keys())
    for metric, score in scores.items():
        print('%s: %.6f' % (metric, score * 100))

    if sub_category_score is not None:
        print('Sub Category Score in Spice:')
        for category, score in sub_category_score.items():
            print('%s: %.6f' % (category, score * 100))
    return scores


if __name__ == '__main__':
    opt = parse_opt()

    with open('HMN/data/MSVD/language/idx2word.pkl', 'rb') as f:
        vocab = pickle.load(f)

    net = models.setup(opt, vocab)
    print(net)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    if not opt.eval_metric:
        net.load_state_dict(torch.load(opt.model_pth_path))
    elif opt.eval_metric == 'METEOR':
        net.load_state_dict(torch.load(opt.best_meteor_pth_path))
    elif opt.eval_metric == 'CIDEr':
        net.load_state_dict(torch.load(opt.best_cider_pth_path))
    elif opt.eval_metric == '0':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '3_model.pth')))
    elif opt.eval_metric == '1':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '7_model.pth')))
    elif opt.eval_metric == '2':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '11_model.pth')))
    elif opt.eval_metric == '3':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '15_model.pth')))
    elif opt.eval_metric == '4':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '19_model.pth')))
    elif opt.eval_metric == '5':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '23_model.pth')))
    elif opt.eval_metric == '6':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '27_model.pth')))
    elif opt.eval_metric == '7':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '31_model.pth')))
    elif opt.eval_metric == '8':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '35_model.pth')))
    elif opt.eval_metric == '9':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '39_model.pth')))
    elif opt.eval_metric == '10':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '43_model.pth')))
    elif opt.eval_metric == '11':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '46_model.pth')))
    elif opt.eval_metric == '12':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '51_model.pth')))
    elif opt.eval_metric == '13':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '55_model.pth')))
    elif opt.eval_metric == '14':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '59_model.pth')))
    elif opt.eval_metric == '15':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '63_model.pth')))
    elif opt.eval_metric == '16':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '67_model.pth')))
    elif opt.eval_metric == '17':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '71_model.pth')))
    elif opt.eval_metric == '18':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '75_model.pth')))
    elif opt.eval_metric == '19':
        net.load_state_dict(torch.load(os.path.join(opt.result_dir, opt.dataset + '79_model.pth')))
    else:
        raise ValueError('Please choose the metric from METEOR|CIDEr')
    net.to(DEVICE)
    net.eval()

    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)
    metrics = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference, vocab)
    with open(opt.test_score_txt_path, 'a') as f:
        f.write('\nBEST ' + str(opt.eval_metric) + '(beam size = {}):\n'.format(opt.beam_size))
        for k, v in metrics.items():
            f.write('\t%s: %.2f\n' % (k, 100 * v))