#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/10/19 10:18

import sys
import os
import time
import logging
from tqdm import tqdm
import argparse
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from collections import defaultdict

import data_utils.arg_otions as arg_options
from DataSet.prep_cpt_data import InputAnchorExample_testing
from utils.pred_save import save_preds, save_preds_for_cpts
from metric import metric_span
from DataSet.cpt_file_load import load_main
from commonCptModel import commonCptODEE
from utils.utils_optimizer import adam_optimizer

logger = logging.getLogger()


def train(args):
    prep_datas = load_main(args)
    train_dataset = prep_datas["train_data"]
    # dev_dataset = prep_datas["dev_data"]
    test_dataset = prep_datas["test_data"]
    id2cpt = prep_datas["id2cpt"]
    id2et = prep_datas["id2et"]
    cpt_id2et_id = prep_datas["cpt_id2et_id"]
    print("Trainer init done!\n")

    if args.evaluate_during_training:
        print("\nWe do evaluate during training ...\n")

    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    n_gpu = len(args.device_id.split(','))
    model = commonCptODEE(args, prep_datas["cpt_vec"], len(prep_datas["cpt_vec"][0])).to(device)

    optim_mode = 'AdamW'
    training_total_batch_num = int(train_dataset.sample_size / (args.train_batch_size * args.gradient_accumulation_steps)) \
                               * args.train_epoch_num

    optimizer, scheduler = adam_optimizer(args, model, optim_mode,
                                                    t_total=training_total_batch_num,
                                                    warmup_steps=int(training_total_batch_num * args.warmup_ratio))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    global_step = 0
    print("Start training ... \n")

    ave_loss = 0.0
    # torch.cuda.synchronize()
    # start_tt = time.time()
    for epoch in range(args.train_epoch_num):
        data_generate = train_dataset.new_dummy_batch()
        for step in tqdm(range(train_dataset.batch_num), miniters=20, leave=False, file=sys.stdout):
            model.train()

            global_step += 1
            current_batch, tmp_step = data_generate.__next__()

            results = model(token_ids=current_batch["input_ids"].to(device),
                             type_ids=current_batch["input_segment_ids"].to(device),
                             mask_ids=current_batch["input_mask"].to(device),
                             anchor_target_idx= current_batch["target_idx"].to(device),    # current_batch["target_idx"].to(device)
                             anchor_target_labels= current_batch["target_mask"].to(device), # current_batch["target_mask"].to(device)
                             anchor_target_mask= current_batch["target_label"].to(device),   # current_batch["target_label"].to(device)
                             anchor_masks=current_batch["anchor_masks"].to(device),
                             anchor_span_ids=current_batch["anchor_span_ids"].to(device),
                             pos_span_masks=current_batch["pos_span_masks"].to(device),
                             neg_span_masks=current_batch["neg_span_masks"].to(device),
                             span_cpt_idx=current_batch["span_cpt_idx"].to(device),
                             span_cpt_masks=current_batch["span_cpt_masks"].to(device),
                             transition_masks = current_batch["transition_masks"].to(device))
            loss = results["loss"]

            if n_gpu > 1:
                loss = loss.sum()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            ave_loss += loss.item()

            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), 1.0, float('inf'))
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # ave_loss = float(ave_loss) / args.gradient_accumulation_steps
                current_print_loss = ave_loss
                ave_loss = 0

            if global_step % args.train_record_steps == 0:
                # ============= training record ================
                logger.info("step {} / {} in total {} of epoch {}, train/loss: {}\n".format(step, train_dataset.batch_num,
                                                                                            train_dataset.sample_size, epoch,
                                                                                            current_print_loss))

            if global_step % (args.dev_record_steps * args.gradient_accumulation_steps / args.gradient_average) != 0:
                continue

            if global_step % (args.test_record_steps * args.gradient_accumulation_steps / args.gradient_average) != 0:
                continue

            # ================== test step ===========================
            logger.info("************  pause in step : {}, start to {} evaluation.  *********\n".format(global_step, 'test'))
            output_dir = os.path.join(args.checkpoint_dir, "checkpoint-test-{}".format(args.test_logging_name))
            test_record_dict = defaultdict(list)

            test_data_generate = test_dataset.new_dummy_batch()
            for test_step in range(test_dataset.batch_num):
                current_batch, tmp_step = test_data_generate.__next__()

                # predict concept score and id for each sample
                results = model(token_ids=current_batch["input_ids"].to(device),
                                type_ids=current_batch["input_segment_ids"].to(device),
                                mask_ids=current_batch["input_mask"].to(device),
                                span_cpt_idx=current_batch["span_cpt_idx"].to(device),
                                span_cpt_masks=current_batch["span_cpt_masks"].to(device),
                                transition_masks=current_batch["transition_masks"].to(device),
                                evaluate=True)

                best_path_list = results["best_path_list"]
                path_cpt_idx_list = results["path_cpt_id_list"]
                path_et_id_list = []
                for pp in path_cpt_idx_list:
                    et_ids = []
                    for pred_id in pp:
                        if pred_id not in cpt_id2et_id:
                            et_ids.append([9])
                        else:
                            et_ids.append(cpt_id2et_id[pred_id])
                    path_et_id_list.append(et_ids)

                test_record_dict["pred_span"].extend(best_path_list)   # B * pred_span_num * 2
                test_record_dict["pred_cpt"].extend(path_cpt_idx_list)   # B * pred_span_num
                test_record_dict["pred_et_id"].extend(path_et_id_list)

                test_record_dict["sent_string_token"].extend(current_batch["sent_string"])
                test_record_dict["sent_string"].extend(current_batch["sent_ori_string"])
                test_record_dict["ground_trigger_span"].extend(current_batch["trigger_spans"])
                test_record_dict["ground_trigger_cpts"].extend(current_batch["trigger_span_cpts"])
                test_record_dict["ground_trigger"].extend(current_batch["triggers"])
                test_record_dict["ground_et"].extend(current_batch["et_id"])
                test_record_dict["sample_tag"].extend(current_batch["is_positive"])

            logger.info("\ntest data check! \n")
            save_preds(test_record_dict, id2cpt, id2et, output_dir)

            ground_event_type = [[int(et) for et in ets] for ets in test_record_dict["ground_et"]]

            P, R, F1 = metric_span(test_record_dict["pred_span"],
                                    test_record_dict["pred_et_id"],
                                    test_record_dict["ground_trigger_span"],
                                    ground_event_type,
                                    test_record_dict["sample_tag"])
            # logger.info('Strict accuracy: {:.2f}'.format(acc))
            logger.info('F1: {:.2f}, P: {:.2f}, R:{:.2f}'.format(F1, P, R))
    # torch.cuda.synchronize()
    # end_tt = time.time()
    # print("time cost: {}".format(end_tt - start_tt))

def test(args):
    pass
    # pred_data_samples = load_cached_predACE_samples()


def main():
    parser = argparse.ArgumentParser()
    arg_options.add_path_options(parser)
    arg_options.add_para_options(parser)
    args = parser.parse_args()

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging_fh = logging.FileHandler(os.path.join(args.log_output_dir, 'log-{}.txt'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    )))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    args.test_logging_name = logging_fh.baseFilename.split('/')[-1].split('.')[0].replace('log-', '')
    print(logging_fh.baseFilename.split('/')[-1].split('.')[0].replace('log-', ''))

    if args.test_only:
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    main()

