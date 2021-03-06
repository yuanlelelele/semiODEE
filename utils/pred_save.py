#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/10/21 9:46

import json


def save_json(objs, out_file):
    with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
        for obj in objs:
            f.write('{}\n'.format(json.dumps(obj, ensure_ascii=False)))


def save_train_log(sent_str, labeled_index, random_index, neg_tags, outfile, anchor_idx, anchor_cpt_ids):
    results = []

    for ss, sent_labeled_index, sent_random_index, sent_neg_tags in zip(sent_str, labeled_index, random_index, neg_tags):

        results.append({
            'sent_str': ss,
            'anchor_idx': str(anchor_idx),
            'anchor_cpt_ids': str(anchor_cpt_ids),
            'labeled_idx': str(sent_labeled_index),
            'random_idx': str(sent_random_index),
            'neg_tags': str(sent_neg_tags)
        })

    save_json(results, outfile)


def save_preds(record, id2anno_cpt, id2et, outfile):

    tmp_result = []
    # print(record["sent_string"])
    # print(record["sent_string_token"])
    # print(record["pred_span"])
    # print(record["pred_cpt"])
    # print(record["pred_et_id"])
    # print(record["ground_trigger_span"])
    # print(record["ground_trigger"])
    # print(record["ground_trigger_cpts"])
    # print(record["ground_et"])
    # print(record["sample_tag"])

    id2et[9] = 'None'
    id2anno_cpt[34442] = 'None'

    sent_pred_span_cpt = []
    sent_pred_span_et = []
    for pred_span_cpt, pred_span_et in zip(record["pred_cpt"], record["pred_et_id"]):
        sent_pred_span_cpt.append([id2anno_cpt[ii] for ii in pred_span_cpt])
        sent_pred_span_et.append([[id2et[ii] for ii in iis] for iis in pred_span_et])

    ground_et = [[id2et[int(id)] for id in ids] for ids in record["ground_et"]]
    ground_cpt = [[id2anno_cpt[id] for id in cpt_ids] for cpt_ids in record["ground_trigger_cpts"]]

    for sample_list in zip(record["sent_string"], record["sent_string_token"], record["pred_span"], sent_pred_span_cpt, sent_pred_span_et, # 0-4
                           record["ground_trigger_span"], record["ground_trigger"], ground_cpt, ground_et, record["sample_tag"]):   # 5-9

        tmp_result.append({'sent_string': sample_list[0]})
        tmp_result.append({'sent_str_token': sample_list[1]})
        tmp_result.append({'pred_span': sample_list[2]})
        tmp_result.append({'pred_span_cpt': sample_list[3]})
        tmp_result.append({'pred_span_et': sample_list[4]})
        tmp_result.append({'ground_trigger_span': sample_list[5]})
        tmp_result.append({'ground_trigger': sample_list[6]})
        tmp_result.append({'ground_cpt': sample_list[7]})
        tmp_result.append({'ground_et': sample_list[8]})
        tmp_result.append({'sample_tag': sample_list[9]})

    # print(tmp_result)
    save_json(tmp_result, outfile)


def save_preds_for_cpts(record, outfile):
    sent_string, candidates, candi_idx, pred_cpt, ground_cpt = record["sent_string_token"], ["candidate"], \
                                                               record["candi_idx"], record["pred_cpt"], \
                                                               record["ground_cpt_for_char"]

    tmp_result = []
    cnt = 0

    for sent_str, candi, anchor_idx, pred_cc, ground_ccs in zip(sent_string,
                                                                candidates,
                                                                candi_idx,
                                                                pred_cpt,
                                                                ground_cpt):

        tmp_result.append({'sent_string': sent_str})

        for anchor, idx, pred_c, ground_c in zip(candi, anchor_idx[:len(candi)], pred_cc[:len(candi)], ground_ccs):

            tmp_result.append({'anchor': str(anchor)})
            tmp_result.append({'anchor_idx': str(idx)})
            tmp_result.append({'pred_cc': str(pred_c)})
            tmp_result.append({'ground_cc': str(ground_c)})

    # print(len(tmp_result))
    # print(tmp_result[0])
    save_json(tmp_result, outfile)


def record_dev():

    ...


def record_test():
    ...


if __name__ == '__main__':
    # x = [i for i in range(10)]
    # y = np.random.rand(10).tolist()
    # print_plot(x,y, "tmp")

    sent_string = [['???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???'], ['???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???'], ['???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???']]
    candidates = [['ss'], ['ss'], ['ss']]
    candi_idx = [22, 55, 8]
    pred_cpt = ['DEF={popular|??????}', 'DEF={popular|??????}', 'DEF={handle|??????:manner={permanent|?????????}}']
    ground_cpt = [['DEF={request|??????}'], ['DEF={expression|??????:{boast|??????:content={~}}}', 'DEF={boast|??????}'], ['DEF={shape|??????}']]
    save_preds_for_cpts(sent_string, candidates, candi_idx, pred_cpt, ground_cpt, './outfile.txt')
