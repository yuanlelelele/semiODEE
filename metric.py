#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/10/20 13:32

import numpy as np


def strict_acc_old(preds, golds, sample_tag):
    '''
    :param preds: [batch_size, labels]
    :param golds: [batch_size, labels]
    :return:
    '''
    hit_count = 0
    total_count = 0
    for gold, pred, tag in zip(golds, preds, sample_tag):
        if not tag:
            continue
        total_count += 1
        if gold in pred:
            hit_count += 1
    return hit_count / total_count * 100.0


def metric_span(pred_spans, pred_et_ids, ground_spans, ground_et_ids, sample_tag):
    golden_num, pred_num, span_pred_num, right_num = 0, 0, 0, 0
    for sent_pred_spans, sent_pred_et_ids, gd_span, gd_et_id, tag in zip(pred_spans,
                                                                    pred_et_ids,
                                                                    ground_spans,
                                                                    ground_et_ids,
                                                                    sample_tag):
        for pred_et in sent_pred_et_ids:
            if 9 not in pred_et:
                pred_num += 1

        if tag:
            golden_num += 1
            if gd_span in sent_pred_spans:
                span_pred_num += 1
                idx = sent_pred_spans.index(gd_span)
                if gd_et_id in sent_pred_et_ids[idx]:
                    right_num += 1

    p = 100.0 * right_num / pred_num if pred_num > 0 else -1
    r = 100.0 * right_num / golden_num if golden_num > 0 else -1
    f = 2 * p * r / (p + r) if (p + r) > 0 else -1
    print("span pred: {} with golden num: {} and pred_span_num: {}".format(100.0 * span_pred_num / golden_num
                                                                           if golden_num > 0 else -1, golden_num, span_pred_num))

    return p, r, f


def metrics(preds, golds, sample_tag):
    acc = strict_acc_old(preds, golds, sample_tag)
    golden_num, pred_num = 0, 0
    right_num = 0
    for pp, gg, tag in zip(preds, golds, sample_tag):
        if tag:
            golden_num += 1
            pred_num += 1
            if gg in pp:
                right_num += 1
        else:
            if gg not in pp:
                pred_num += 1

    p = 100.0 * right_num / pred_num if pred_num > 0 else -1
    r = 100.0 * right_num / golden_num if golden_num > 0 else -1
    f = 2 * p * r / (p + r) if (p + r) > 0 else -1

    return acc, p, r, f


def f1_score(precision, recall):
    p, r = precision, recall
    if p or r:
        return 2 * p * r / (p + r + 1e-7)
    else:
        return 0


def microf1(preds, golds):
    assert len(preds) == len(golds)
    pred_total, gold_total, overlap = 0, 0, 0
    overlap += sum([i and j for i, j in zip(golds, preds)])
    gold_total += sum(golds)
    pred_total += sum(preds)
    p = 0 if pred_total == 0 else overlap / pred_total
    r = 0 if gold_total == 0 else overlap / gold_total
    return f1_score(p, r) * 100.0, p * 100.0, r * 100.0


def macrof1(preds, golds):
    assert len(preds) == len(golds)
    precision, recall, overlap = 0, 0, 0
    total_gold_num, total_pred_num = 0, 0
    overlap = sum([i and j for i, j in zip(golds, preds)])
    gold_num = sum(golds)
    pred_num = sum(preds)
    total_gold_num += (1 if gold_num > 0 else 0)
    total_pred_num += (1 if pred_num > 0 else 0)
    precision += 0 if pred_num == 0 else overlap/pred_num
    recall += 0 if gold_num == 0 else overlap/gold_num
    p = precision / total_pred_num if total_pred_num else 0
    r = recall / total_gold_num if total_gold_num else 0
    return f1_score(p, r) * 100.0, p * 100.0, r * 100.0


def transfer(preds, golds):
    new_preds = []
    new_golds = []
    neg_cnt = 0
    for pp, gg in zip(preds, golds):
        if gg == 9:
            neg_cnt += 1
            continue

        if len(pp) == 1:
            new_preds.append(pp[0])

        elif len(pp) > 1:
            if gg in pp:
                new_preds.append(gg)
            else:
                new_preds.append(pp[0])
        else:
            print("error pred length!\n")

        new_golds.append(gg)

    type_num = 34
    new_preds = np.eye(type_num)[new_preds].flatten()
    new_golds = np.eye(type_num)[new_golds].flatten()
    print("neg_cnt: {}".format(neg_cnt))

    return new_preds, new_golds


def strict_acc(preds, golds):
    '''
    :param preds: [batch_size, labels]
    :param golds: [batch_size, labels]
    :return:
    '''
    hit_count = 0
    for gold, pred in zip(golds, preds):
        if gold == pred:
            hit_count += 1
    return hit_count / len(golds) * 100.0


def count_TP(preds, golds, samples):
    hit_count = 0
    for pred, gold, tag in zip(preds, golds, samples):
        if tag:
            if gold in pred:
                hit_count += 1

    return hit_count


def count_FN(preds, golds, samples):
    hit_count = 0
    for pred, gold, tag in zip(preds, golds, samples):
        if tag:
            if gold not in pred:
                hit_count += 1

    return hit_count


def count_FP(preds, golds, samples):
    hit_count = 0
    for pred, gold, tag in zip(preds, golds, samples):
        if not tag:
            if gold not in pred:
                hit_count += 1

    return hit_count


if __name__ == '__main__':
    num = 34
    pred = [9,9,3,3,27,24,9]
    pred = np.eye(num)[pred]#.tolist()
    pred = pred.flatten()

    gold = [9,9,9,9,27,9,9]
    gold = np.eye(num)[gold]#.tolist()
    gold = gold.flatten()

    print(sum(pred))
    print(sum(gold))

    overlap = sum([i and j for i, j in zip(gold, pred)])

    print(overlap)

