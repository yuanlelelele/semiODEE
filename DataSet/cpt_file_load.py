import numpy as np
import os
from collections import defaultdict
import pickle
import random
import logging
import torch

import config
import data_utils.pkl as pkl
from DataSet.FormatDataset import FormatDataSet

logger = logging.getLogger()

concept_data_dir = "/data/xxin/workspace/data"
x_path = "/data/xxin/workspace/CptODEE_new/data"


class CptEmbedding(object):

    def __init__(self):
        cpt2words, cpt2id, et2id, sememes, cpt_tree = load_annotated_concepts_new()
        # self.verb_cpts = load(os.path.join(x_path, "total_verb_cpts.pkl"))

        self.sememe2id = {sem: idx for idx, sem in enumerate(sememes)}
        self.cpt2words = cpt2words
        self.cpt2id = cpt2id
        self.cpt_list = [cc for cc in self.cpt2id.keys()]
        self.cpt_num = len(self.cpt2id)
        self.sememe_num = len(self.sememe2id)
        # self.et2cpts = et2cpts
        self.et2id = et2id
        self.cpt_tree = cpt_tree
        self.id2cpt = {self.cpt2id[x]: x for x in self.cpt2id.keys()}
        self.cpt_id_list = [ii for ii in self.id2cpt.keys()]

        # concpet to its center sememe
        self.cpt2center_sem = self.get_center_sem_from_cpts()

        # self.sememe_embedding_file = "%s/sememe_embedding.pkl" % x_path
        self.cpt_id2sememe_ids_file = "%s/cpt_id2sememe_ids.pkl" % x_path

        self.cpt_id2embeddings_file = "%s/cpt_id2embeddings.pkl" % x_path

        # load cpt_id 2 sememe_ids in a dict
        self.cpt_id2sememe_ids = load(self.cpt_id2sememe_ids_file)

        self.complement_cpts = self.load_add_condition_concepts()

        self.word2cpt_idx_file = "%s/word2cpt_ids_without_subcpt.pkl" % x_path
        self.eventType2cpts_file = "%s/eventType2cpts_without_subcpt.pkl" % x_path

        self.word2cpt_idx = load(self.word2cpt_idx_file)
        self.et2cpts = load(self.eventType2cpts_file)

        self.cpt_vec_in_bert_file = "%s/cpt_embeddings_in_bert" % x_path

    def get_child_concpet_id(self, tree):
        sub_cpt_ids = []

        if tree.child:
            for cc in tree.child:
                sub_cpt_ids.append(self.cpt2id[cc.node])
                sub_cpt_ids.extend(self.get_child_concpet_id(cc))

        return sub_cpt_ids

    def load_add_condition_concepts(self):
        file_path = '%s/add_condition_for_concepts.txt' % x_path
        complement_cpts = []
        for line in open(file_path, 'r', encoding='utf-8'):
            line = line.strip().split('\t')
            complement_cpts.append(line[1])

        print("complement concept nums: {}".format(len(complement_cpts)))

        return complement_cpts

    def add_sub_concepts_version2(self, tree):
        sub_cpt_list = []
        if tree.child:
            for cc in tree.child:
                if cc.node in self.complement_cpts:
                    sub_cpt_list.append(self.cpt2id[cc.node])
                sub_cpt_list.extend(self.add_sub_concepts_version2(cc))

        return sub_cpt_list

    def add_sub_concepts(self, tree):
        sub_cpt_list = []
        if tree.child:
            for cc in tree.child:
                sub_cpt_list.append(self.cpt2id[cc.node])
                sub_cpt_list.extend(self.add_sub_concepts(cc))

        return sub_cpt_list

    def get_center_sem_from_cpts(self):

        concept2center_sem = {}

        for cc in self.cpt_list:
            center_sem = cc.split('={')[1].split('|')[1].split('}')[0].split(':')[0]
            concept2center_sem[cc] = center_sem
        # for cc in self.cpt_list[:20]:
        #     print(cc, concept2center_sem[cc])

        return concept2center_sem

    def get_random_vecs(self, dim):
        # scale = np.sqrt(6. / (1 + dim))
        scale = 0.1
        vec = np.random.uniform(low=-scale, high=scale, size=[dim])
        vec = vec / np.sqrt(np.sum(vec * vec))
        assert abs(np.sum(vec * vec) - 1.0) < 0.1
        return vec

    def get_zero_vects(self, dim):
        return np.array([0.0] * dim)


class PrintFile(object):

    def __init__(self):
        pass

    def check_cpt2words(self, cpt2words):
        print("\ncheck top 10 cpt2words ...\n")
        num = 0
        for cc, wws in cpt2words.items():
            print(cc , "\t:\t", wws)
            num += 1
            if num > 10:
               break


def save(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def load_annotated_concepts_new():
    eventType_file = os.path.join(x_path, "label2id.dat")
    et2id = {}
    for line in open(eventType_file, 'r', encoding='utf-8'):
        line = line.strip().split('\t')
        et2id[line[0].strip()] = int(line[1].strip())

    cpt2words = load(os.path.join(x_path, "total_cpt2words_with_add_condition_cpts_1125.pkl"))
    sememes = load(os.path.join(x_path, "sememe_list.pkl"))
    cpt_tree = 'none'

    cpts = [cc for cc in cpt2words.keys()]
    print("cc num: ", len(cpts), len(set(cpts)))

    cpt2id = {tmp_cpt: idx for idx, tmp_cpt in enumerate(cpt2words.keys())}
    # print(id2cpt[0])

    return cpt2words, cpt2id, et2id, sememes, cpt_tree


def load_cached_samples_for_mlm():
    dev_sample_length = 4
    cached_sample_dir = os.path.join(config.cached_data_dir, "cached_span_sample_without_mlm")

    inputs = []
    for d_file in os.listdir(cached_sample_dir):
        if d_file != "span_samples_1.pkl":  # span_samples_test span_samples_1
            continue
        print("d_file: ", d_file)
        file_path = os.path.join(cached_sample_dir, d_file)
        if os.path.getsize(file_path) > 0:
            inputs.extend(pkl._load(file_path))
    # torch.save(inputs, os.path.join(cached_sample_dir, "inputs_for_check"))

    # random.shuffle(inputs)
    # random.seed(233)
    # random.shuffle(inputs)
    print("total sample length : {}".format(len(inputs)))

    return inputs, cached_sample_dir


def load_cached_predACE_samples():
    cached_sample_file = os.path.join(config.cached_data_dir, "cached_testACE_span_samples")

    labeled_inputs = torch.load(cached_sample_file)

    return labeled_inputs


def load_cached_testACE_samples():
    cached_sample_file = os.path.join(config.cached_data_dir, "cached_devACE_span_samples")
    labeled_inputs = torch.load(cached_sample_file)

    return labeled_inputs

def format_cpt_inf(cpt2id, et2cpts, et2id):
    id2cpt = {idx: cc for cc, idx in cpt2id.items()}
    et2id = et2id
    id2et = {id: et for et, id in et2id.items()}
    anno_cpt2et = defaultdict(list)
    et_id2cpt_ids = defaultdict(list)
    for et, cpts in et2cpts.items():
        # print(self.et_id2cpt_ids)
        for cc in cpts:
            anno_cpt2et[cc].append(et)
            et_id2cpt_ids[et2id[et]].append(cpt2id[cc])
    cpt_id2et_id = {cpt2id[cc]: [et2id[et] for et in ets] for cc, ets in anno_cpt2et.items()}

    return id2cpt, id2et, cpt_id2et_id

def load_main(args):
    print("Load prep data...\n")

    CptEmb = CptEmbedding()
    word2cpt_ids = CptEmb.word2cpt_idx
    cpt2id = CptEmb.cpt2id
    et2id = CptEmb.et2id
    # verb_cpts = CptEmb.verb_cpts
    et2cpts = CptEmb.et2cpts

    id2cpt, id2et, cpt_id2et_id = format_cpt_inf(cpt2id, et2cpts, et2id)

    # verb_cpt_ids = [cpt2id[cc] for cc in verb_cpts]

    cpt_vec = torch.load(CptEmb.cpt_vec_in_bert_file)  # [:10907]   # [:34442]    # 34443 * 768, padding index = 34442
    logger.info("cpt embedding file: {}".format(CptEmb.cpt_vec_in_bert_file))
    logger.info("cpt vec length: {}".format(len(cpt_vec)))

    args.cpt_num = CptEmb.cpt_num
    print("cpt nums: {}\n".format(args.cpt_num))
    print("HowNet words cnt: {}".format(len(word2cpt_ids)))

    print("Load training data...\n")
    print("Load dev_1 data...\n")
    train_data_samples, cached_selected_sample_filename = load_cached_samples_for_mlm()
    print("Load testing data...\n")
    test_data_samples = load_cached_testACE_samples()

    dataset = FormatDataSet(args, train_data_samples, len(train_data_samples),
                                 shuffle=False, mode='Training')
    # dev_dataset = FormatDataSet(args, dev_data_samples, len(dev_data_samples),
    #                                  shuffle=False, mode='dev')
    test_dataset = FormatDataSet(args, test_data_samples, len(test_data_samples),
                                      shuffle=False, mode='Testing')

    logger.info("training corpus type: {}".format(cached_selected_sample_filename))
    logger.info('training data length: {}\ttesting data length: {}'.format(
        dataset.sample_size, test_dataset.sample_size))
    logger.info('learning rate: {}'.format(args.lr))
    logger.info('warmup_ratio: {}'.format(args.warmup_ratio))
    logger.info('batch size: {}'.format(args.train_batch_size))
    logger.info('gradient_accumulation_steps: {}'.format(args.gradient_accumulation_steps))
    logger.info('train_epoch_num: {}'.format(args.train_epoch_num))
    logger.info('negative sampled size: {}'.format(args.random_cpt_num))

    # pred_data_samples = load_cached_predACE_samples()

    return {"id2cpt": id2cpt,
            "id2et": id2et,
            "word2cpt_ids": word2cpt_ids,
            "cpt2id": cpt2id,
            "et2id": et2id,
            "cpt_vec": cpt_vec,
            "et2cpts": et2cpts,
            "cpt_id2et_id": cpt_id2et_id,
            "train_data": dataset,
            "cached_selected_sample_filename": cached_selected_sample_filename,
            "test_data": test_dataset,
            }


if __name__ == '__main__':
    inputs, _, _ = load_cached_samples_for_mlm()
    exm = inputs[5]
    print(exm.sent_str)
    print(exm.sent_str_token)
    print(exm.sent_seg_token)
    print(exm.anchor)
    print(exm.anchor_idx)
    print(exm.anchor_cpt_ids)



