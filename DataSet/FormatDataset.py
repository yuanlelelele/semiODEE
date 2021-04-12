#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/11/26 21:27
import copy
import random
import numpy as np
import torch


class SpanMap:
    def __init__(self):
        self.id2span = {
                        0: (0, 1),
                        1: (0, 2), 2: (1, 2),
                        3: (0, 3), 4: (1, 3), 5: (2, 3)}

        self.span2id = {self.id2span[x]: x for x in self.id2span.keys()}
        self.span_types = [x for x in self.span2id.keys()]  # (char_idx, token_length)


class FormatDataSet(object):

    def __init__(self, args, samples, sample_size, shuffle, mode):
        self.args = args
        self.mode = mode
        self.samples = samples
        self.sample_size = sample_size
        self.num_samples = sample_size

        self.span_types = SpanMap().span_types

        self.step = 0
        self.shuffle = shuffle

        # self.span_labels = self.define_span_labels()

        if shuffle:
            random.shuffle(self.samples)

        if mode == 'Training':
            self.batch_num = int(self.sample_size / self.args.train_batch_size)
            self.batch_size = self.args.train_batch_size
        elif mode == 'dev':
            self.batch_num = int(self.sample_size / self.args.dev_batch_size)
            self.batch_size = self.args.dev_batch_size
        else:
            self.batch_num = int(self.sample_size / self.args.test_batch_size)
            self.batch_size = self.args.test_batch_size

        # self.cpt_emb = cpt_emb
        print("dataset predefined! with batch num: {}\n".format(self.batch_num))
        self.neg_cpt_id = 34442
        self.max_span_length = 4
        self.max_span_num = 10

    def empty_batch(self):
        batch = {}

        batch["sent_string"] = []
        batch["sent_ori_string"] = []

        batch["input_ids"] = []
        batch["input_mask"] = []
        batch["input_segment_ids"] = []

        if self.mode == 'Training':
            batch["anchor_masks"] = []
            batch["anchor_span_ids"] = []
            batch["pos_span_masks"] = []
            batch["neg_span_masks"] = []

        batch["span_ids"] = []
        batch["spans"] = []
        batch["span_masks"] = []
        batch["span_cpt_idx"] = []
        batch["span_cpt_masks"] = []
        batch["transition_masks"] = []

        if self.mode == 'Testing':
            batch["trigger_spans"] = []
            batch["trigger_span_cpts"] = []
            batch["triggers"] = []
            batch["et_id"] = []
            batch["is_positive"] = []

        if self.args.add_mlm_object:
            batch['target_idx'] = []
            batch["target_label"] = []
            batch['target_mask'] = []

        return batch

    def padding_target_mask_labels(self, data, batch):
        max_length = 0
        for exm in data:
            max_length = max(len(exm["anchor_idx"]), max_length)

        for exm in data:
            target_idx = copy.deepcopy(exm["anchor_idx"]) + [0] * max(0, max_length - len(exm["anchor_idx"]))
            target_masks = [1] * len(exm["anchor_idx"]) + [0] * max(0, max_length - len(exm["anchor_idx"]))
            target_labels = copy.deepcopy(exm["target_label"]) + [-1] * max(0, max_length - len(exm["target_label"]))
            batch["target_idx"].append(target_idx)
            batch["target_mask"].append(target_masks)
            batch["target_label"].append(target_labels)
        batch["target_idx"] = torch.LongTensor(batch["target_idx"])
        batch["target_mask"] = torch.LongTensor(batch["target_mask"])
        batch["target_label"] = torch.LongTensor(batch["target_label"])

    def span_with_anchor(self, data, batch):
        max_anchor_num = -1
        for exm in data:
            max_anchor_num = max(max_anchor_num, len(exm["anchor_idx"]))

        for exm in data:
            # exm["anchor_idx"] = [exm["anchor_idx"][0]]
            # exm["sent_pos_span"] = [exm["sent_pos_span"][0]]
            # exm["sent_neg_span"] = [exm["sent_neg_span"][0]]
            # sent_anchor_idx = [] + exm["anchor_idx"] + [0] * (max_anchor_num - len(exm["anchor_idx"]))
            sent_anchor_masks = [1] * len(exm["anchor_idx"]) + [0] * (max(max_anchor_num - len(exm["anchor_idx"]), 0))
            sent_anchor_spans = []
            sent_anchor_pos_masks = []
            sent_anchor_neg_masks = []
            for aps, ans in zip(exm["sent_pos_span"], exm["sent_neg_span"]):
                anchor_pos_span = [[tmp[0], tmp[-1] - tmp[0]] for tmp in aps]
                anchor_neg_span = [[tmp[0], tmp[-1] - tmp[0]] for tmp in ans]
                anchor_span = anchor_pos_span + anchor_neg_span
                anchor_span += [[0, 0] for _ in range(self.max_span_num - len(anchor_span))]
                anchor_pos_masks = [1] * len(anchor_pos_span) + [0] * len(anchor_neg_span)
                anchor_pos_masks += [0] * (self.max_span_num - len(anchor_pos_masks))
                anchor_neg_masks = [0] * len(anchor_pos_span) + [1] * len(anchor_neg_span)
                anchor_neg_masks += [0] * (self.max_span_num - len(anchor_neg_masks))
                assert len(anchor_pos_masks) == len(anchor_neg_masks) == self.max_span_num
                sent_anchor_spans.append(anchor_span)
                sent_anchor_pos_masks.append(anchor_pos_masks)
                sent_anchor_neg_masks.append(anchor_neg_masks)
            sent_anchor_spans += [[[0, 0] for _ in range(self.max_span_num)] for _ in range(max(0, max_anchor_num - len(exm["anchor_idx"])))]
            sent_anchor_pos_masks += [[0]*self.max_span_num] * max(0, max_anchor_num - len(exm["anchor_idx"]))
            sent_anchor_neg_masks += [[0]*self.max_span_num] * max(0, max_anchor_num - len(exm["anchor_idx"]))
            # batch["anchor_idx"].append(sent_anchor_idx) # B * max_anchor_num
            batch["anchor_masks"].append(sent_anchor_masks)
            batch["anchor_span_ids"].append(sent_anchor_spans)  # B * max_anchor_num * max_span_num * 2
            batch["pos_span_masks"].append(sent_anchor_pos_masks)   # B * max_anchor_num * max_span_num
            batch["neg_span_masks"].append(sent_anchor_neg_masks)   # B * max_anchor_num * max_span_num

        # batch["anchor_idx"] = torch.LongTensor(batch["anchor_idx"])
        batch["anchor_masks"] = torch.LongTensor([idx for idx in batch["anchor_masks"]])
        batch["anchor_span_ids"] = torch.LongTensor(batch["anchor_span_ids"])
        batch["pos_span_masks"] = torch.LongTensor(batch["pos_span_masks"])
        batch["neg_span_masks"] = torch.LongTensor(batch["neg_span_masks"])

    def padding_transfer_mask(self, data, batch):
        padding_mask = [0] * self.max_span_length
        for exm in data:
            cur_seq_len = len(exm["sent_str_token"]) + 2
            seq_dim = []
            span1_dim =[]
            for dd in range(self.max_span_length):
                if dd == 0:
                    span1_dim.append([1]*self.max_span_length)
                else:
                    span1_dim.append([0]*self.max_span_length)
            seq_dim.append(span1_dim)
            for ii in range(1, cur_seq_len):
                span1_dim = []
                for jj in range(self.max_span_length):
                    span2_dim = []
                    for kk in range(self.max_span_length):
                        span2_dim.append(1 if ii+jj+kk < cur_seq_len - 1 else 0)
                    span1_dim.append(span2_dim)
                seq_dim.append(span1_dim)
            seq_dim[cur_seq_len-1][0][0] = 1
            for ii in range(self.args.seq_max_length - cur_seq_len):
                seq_dim.append([padding_mask for _ in range(self.max_span_length)])
            batch["transition_masks"].append(seq_dim)
        batch["transition_masks"] = torch.LongTensor(batch["transition_masks"])


    def span_with_sample(self, data, batch):

        invalid_cpt_idx = [34442] * self.args.span_max_cpt_num
        invalid_cpt_idx_masks = [0] * self.args.span_max_cpt_num
        max_span_num = self.args.seq_max_length * self.max_span_length
        for exm in data:
            sent_span_cpt_ids = []
            sent_span_cpt_masks = []
            sent_span_cpt_ids.extend([invalid_cpt_idx for _ in range(self.max_span_length)])
            sent_span_cpt_masks.extend([invalid_cpt_idx_masks for _ in range(self.max_span_length)])
            for anchor_idx, (anchor_spans, anchor_cpt_ids) in enumerate(zip(exm["sent_spans"], exm["sent_cpt_ids"])):
                for offset in range(self.max_span_length):
                    cur_span = [anchor_idx + 1 + ii for ii in range(offset+1)]
                    if cur_span not in anchor_spans:
                        sent_span_cpt_ids.append(invalid_cpt_idx)
                        sent_span_cpt_masks.append(invalid_cpt_idx_masks)
                        continue
                    iidx = anchor_spans.index(cur_span)
                    tmp_cpt = copy.deepcopy(anchor_cpt_ids[iidx])
                    sent_span_cpt_ids.append(tmp_cpt[:self.args.span_max_cpt_num] + [0]*max(0, self.args.span_max_cpt_num - len(tmp_cpt)))
                    sent_span_cpt_masks.append([1] * min(self.args.span_max_cpt_num, len(tmp_cpt)) + [0] * max(0, self.args.span_max_cpt_num - len(tmp_cpt)))

            # print("max span cpt num: {}".format(max_span_cpt_num))
            batch["span_cpt_idx"].append(sent_span_cpt_ids + [invalid_cpt_idx for _ in range(max_span_num - len(sent_span_cpt_ids))])
            batch["span_cpt_masks"].append(sent_span_cpt_masks + [invalid_cpt_idx_masks for _ in range(max_span_num - len(sent_span_cpt_masks))])

        batch["span_cpt_idx"] = torch.LongTensor(batch["span_cpt_idx"])
        batch["span_cpt_masks"] = torch.LongTensor(batch["span_cpt_masks"])

    def new_dummy_batch(self):

        for step in range(self.batch_num):

            batch = self.empty_batch()
            start = step * self.batch_size
            end = min(step * self.batch_size + self.batch_size, self.sample_size)

            tmp_batch_data = self.samples[start: end]
            if self.shuffle:
                random.shuffle(tmp_batch_data)

            batch["input_mask"].extend([exm["input_mask"] for exm in tmp_batch_data])
            batch["input_mask"] = torch.LongTensor([ids for ids in batch["input_mask"]])
            batch["input_segment_ids"].extend([exm["segment_ids"] for exm in tmp_batch_data])
            batch["input_segment_ids"] = torch.LongTensor([ids for ids in batch["input_segment_ids"]])

            batch["sent_string"].extend([exm["sent_str_token"] for exm in tmp_batch_data])

            self.padding_transfer_mask(tmp_batch_data, batch)
            self.span_with_sample(tmp_batch_data, batch)

            if self.args.add_mlm_object and self.mode != 'Testing':
                self.padding_target_mask_labels(tmp_batch_data, batch)

            if self.mode == 'Testing':
                batch["input_ids"].extend([exm["input_ids"] for exm in tmp_batch_data])
                batch["input_ids"] = torch.LongTensor([ids for ids in batch["input_ids"]])
                batch["sent_ori_string"].extend([exm["sent_str"] for exm in tmp_batch_data])
                batch["trigger_spans"].extend([exm["trigger_spans"] for exm in tmp_batch_data])
                batch["trigger_span_cpts"].extend(exm["trigger_span_cpts"] for exm in tmp_batch_data)
                batch["triggers"].extend([exm["triggers"] for exm in tmp_batch_data])
                batch["et_id"].extend([exm["et_ids"] for exm in tmp_batch_data])
                batch["is_positive"].extend([exm["is_positive"] for exm in tmp_batch_data])

            if self.mode == "Training":
                if self.args.add_mlm_object:
                    batch["input_ids"].extend([exm["input_mask_ids"] for exm in tmp_batch_data])
                else:
                    batch["input_ids"].extend([exm["input_ids"] for exm in tmp_batch_data])
                batch["input_ids"] = torch.LongTensor([ids for ids in batch["input_ids"]])
                self.span_with_anchor(tmp_batch_data, batch)

            yield batch, step




