#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2021/1/19 21:02

import os
import time
import datetime
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.nn.utils import clip_grad_norm_
from torch.serialization import default_restore_location
from pkg_resources import parse_version
from apex import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils.utils_optimizer import adam_optimizer
from utils.metric import metrics
from utils.pred_save import save_preds, print_plot, save_train_log, save_preds_for_sems, save_preds_for_cpts

logger = logging.getLogger()


class Trainer():

    def __init__(self, args, train_samples, dev_samples, dev_ace_samples, test_ace_samples, cpt_model, id2cpt, id2et, cpt_id2et_id):
        self.id2cpt = id2cpt
        self.id2et = id2et
        self.cpt_id2et_id = cpt_id2et_id

        # init distributed
        self.device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")

        self.logger = logger.setLevel(logging.INFO if dist.get_rank() == 0 else logging.WARNING)
        # Setup logging
        # 同步start_time
        sync_time = torch.tensor(time.time(), dtype=torch.double).to(self.device)
        dist.broadcast(sync_time, src=0)
        # self.start_time = datetime.fromtimestamp(sync_time.item()).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.n_gpu = len(args.device_id.split(','))
        self.rank = args.local_rank
        self.world_size = dist.get_world_size()

        # self.model = cpt_model
        self.logger = logger
        self.logger.info("model name: {}".format("CptNllODEE"))
        self.logger.info("add_mlm_object tag: {}".format(args.add_mlm_object))
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # init data loader
        # ************** train data ************************
        # train_sampler = DistributedSampler(train_samples)
        self.train_loader = train_samples
        # ************** dev data ************************
        self.dev_loader = dev_samples
        self.dev_ace_loader = dev_ace_samples
        # ************** test data ************************
        # self.test_loader = DataLoader(test_ace_samples, batch_size=args.per_gpu_eval_batch_size,
        #                               collate_fn=test_ace_samples.collate_fn)

        cpt_model.to(self.device)

        self.optim_mode = 'AdamW'

        self.n_steps = len(self.train_loader) * args.train_epoch_num
        self.logger.info("dataloader length: {}".format(len(self.train_loader)))
        self.print_step = self.args.train_record_steps
        self.update_step = self.args.gradient_accumulation_steps
        self.dev_step = self.args.dev_record_steps * self.args.gradient_accumulation_steps / self.args.gradient_average
        self.test_step = self.args.test_record_steps * self.args.gradient_accumulation_steps / self.args.gradient_average

        self.optimizer, self.scheduler = adam_optimizer(args, cpt_model, self.optim_mode,
                                                        t_total=self.n_steps,
                                                        warmup_steps=int(self.n_steps * args.warmup_ratio))

        # init fp16, must before DataParallel init
        if len(args.fp16):
            assert isinstance(args.fp16,
                              str), "Please set Apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']"
            cpt_model, self.optimizer = amp.initialize(cpt_model, self.optimizer, opt_level=args.fp16)

        # init DataParallel
        self.ddp_model = DistributedDataParallel(cpt_model, device_ids=[args.local_rank],
                                                 output_device=args.local_rank, find_unused_parameters=True)

        self.model = self.ddp_model.module

        self.logger.info("Setup Distributed Trainer")
        self.logger.warning("Process pid: {}, rank: {}, local rank: {}, device: {}, fp16: {}".format(
            os.getpid(), self.rank, args.local_rank, self.device, args.fp16 if args.fp16 else False))
        self.logger.info("Num of processes: {}".format(self.world_size))
        self.logger.info("Use device: {}".format(self.device))
        self.logger.info("Training with fp16: {}, optimization level: {}".format(
            len(args.fp16) > 0, args.fp16 if args.fp16 else None))

    @property
    def is_master(self):
        return self.rank == 0

    def train(self):

        self.logger.info("Start training ... \n")
        self.tqdm_bar = tqdm(total=self.n_steps, miniters=20, leave=False, dynamic_ncols=True, disable=not self.is_master)
        tqdm_bar = self.tqdm_bar

        global_step = 0
        self.step = global_step
        ave_loss = 0.0

        self.ddp_model.zero_grad()
        for epoch in range(1, self.args.train_epoch_num+1):

            tqdm_bar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.train_epoch_num))

            for (batch_x, batch_y) in self.train_loader:

                self.ddp_model.train()
                self.step += 1

                results = self.train_with_anchor(global_step, batch_x, batch_y)
                loss, invalid_sum, labeled_index, random_index, neg_tags, needed_sum = results["loss"], \
                                                                                       results["score_tag_invalid_sum"], \
                                                                                       results["max_labeled_index"], \
                                                                                       results["max_random_index"], \
                                                                                       results["score_tag_invalid"], \
                                                                                       results["score_tag_needed_sum"]

                if self.args.add_mlm_object:
                    anchor_tag_sum, neg_tag_sum = results["pred_tag_sum"], results["pred_neg_tag_sum"]

                if self.n_gpu > 1:
                    loss = loss.sum()
                    invalid_sum = invalid_sum.sum()
                    needed_sum = needed_sum.sum()

                    if self.args.add_mlm_object:
                        anchor_tag_sum = anchor_tag_sum.sum()
                        neg_tag_sum = neg_tag_sum.sum()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                ave_loss += loss.item()

                if self.args.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    loss.backward()

                if self.step % self.update_step == 0:
                    clip_grad_norm_(self.model.parameters(), 5.0, float('inf'))
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                    ave_loss = float(ave_loss) / self.update_step
                    current_print_loss = ave_loss
                    ave_loss = 0

                if self.step % self.print_step == 0:

                    logger.info("invalid precent: {:.2f}. invalid value check num: {} with needed value: {} in total sample num".format(
                        torch.true_divide(invalid_sum, (invalid_sum + needed_sum)), invalid_sum, needed_sum))

                    if self.args.add_mlm_object:
                        logger.info("mask token predict correct num: {}, "
                                    "neg sum: {}, ratio: {}".format(anchor_tag_sum, neg_tag_sum, "{:.2f}".format(
                                    torch.true_divide(anchor_tag_sum, (anchor_tag_sum + neg_tag_sum)))))

                    print_out_info = (
                        "step {} / {} of epoch {}, train/loss: {}\n".format(self.step, self.n_steps, epoch,
                                                                            current_print_loss))
                    tqdm_bar.update(self.print_step)
                    tqdm_bar.set_postfix_str(print_out_info)

                if self.step % self.dev_step == 0:
                    self.test_with_anchor('dev')
                dist.barrier()

                if self.step % self.test_step == 0 and self.is_master:
                    acc, P, R, F1 = self.test_with_ACE('test')
                    self.logger.info('Strict accuracy: {:.2f}'.format(acc))
                    self.logger.info('F1: {:.2f}, P: {:.2f}, R:{:.2f}'.format(F1, P, R))
                dist.barrier()

        tqdm_bar.close()
        self.tqdm_bar = None

    def train_with_anchor(self, step, batch_x, batch_y):
        token_ids = batch_x["input_ids"].to(self.device)
        type_ids = batch_x["input_segment_ids"].to(self.device)
        mask_ids = batch_x["input_mask"].to(self.device)
        anchor_idx = batch_x["anchor_idx"].to(self.device)

        cpt_random_idx = batch_y["random_cpt_idx"].to(self.device)
        anchor_cpt_idx = batch_y["anchor_cpt_idx"].to(self.device)
        anchor_cpt_idx_masks = batch_y["anchor_cpt_idx_masks"].to(self.device)

        if self.args.add_mlm_object:
            anchor_target_idx = batch_x["target_idx"].to(self.device)
            anchor_target_labels = batch_y["target_label"].to(self.device)
            anchor_target_mask = batch_x["target_mask"].to(self.device)
        else:
            anchor_target_idx = 'none'
            anchor_target_labels = 'none'
            anchor_target_mask = 'none'

        results = self.ddp_model(token_ids,
                             type_ids,
                             mask_ids,
                             anchor_target_idx=anchor_target_idx,
                             anchor_target_labels=anchor_target_labels,
                             anchor_target_mask=anchor_target_mask,
                             cpt_random_idx=cpt_random_idx,
                             cpt_idx=anchor_cpt_idx,
                             cpt_idx_masks=anchor_cpt_idx_masks,
                             anchor_idx=anchor_idx,
                             anchor_cpt_masks=None,
                             span_tag=None,
                             is_eval=False)

        return results

    def test_with_anchor(self, eval_type):
        self.logger.info("************************pause in step : {}, start to {} evaluation.*************************\n".format(
            self.step, eval_type))

        # eval
        output_dir = os.path.join(self.args.checkpoint_dir, "checkpoint-dev-{}".format(self.args.test_logging_name))
        self.ddp_model.eval()
        pred_cpt = []
        ground_cpt_for_char = []
        sent_string_token = []
        candi_idx = []
        candidate = []

        cpt_invalid_sum_list = []
        cpt_needed_sum_list = []
        anchor_tag_sum_list = []
        anchor_neg_tag_sum_list = []

        with self.ddp_model.no_sync():
            for (batch_x, batch_y) in self.dev_loader:

                token_ids = batch_x["input_ids"].to(self.device)
                type_ids = batch_x["input_segment_ids"].to(self.device)
                mask_ids = batch_x["input_mask"].to(self.device)
                anchor_idx = batch_x["anchor_idx"].to(self.device)

                cpt_random_idx = batch_y["random_cpt_idx"].to(self.device)
                anchor_cpt_idx = batch_y["anchor_cpt_idx"].to(self.device)
                anchor_cpt_idx_masks = batch_y["anchor_cpt_idx_masks"].to(self.device)

                if self.args.add_mlm_object:
                    anchor_target_idx = batch_x["target_idx"].to(self.device)
                    anchor_target_labels = batch_y["target_label"].to(self.device)
                    anchor_target_mask = batch_x["target_mask"].to(self.device)
                else:
                    anchor_target_idx = 'none'
                    anchor_target_labels = 'none'
                    anchor_target_mask = 'none'

                # predict concept score and id for each sample
                results = self.ddp_model(token_ids,
                                      type_ids,
                                      mask_ids,
                                      anchor_target_idx=anchor_target_idx,
                                      anchor_target_labels=anchor_target_labels,
                                      anchor_target_mask=anchor_target_mask,
                                      cpt_random_idx=None,
                                      cpt_idx=anchor_cpt_idx,
                                      cpt_idx_masks=anchor_cpt_idx_masks,
                                      anchor_idx=anchor_idx,
                                      anchor_cpt_masks=None,
                                      span_tag=None,
                                      is_eval=eval_type)

                cpt_pred_idx, cpt_invalid_sum, cpt_needed_sum = results["max_pred_id"], \
                                                                results["score_tag_invalid_sum"], \
                                                                results["score_tag_needed_sum"]
                if self.args.add_mlm_object:
                    anchor_pred_idx, anchor_tag_sum, neg_tag_sum = results["anchor_pred_idx"], \
                                                                   results["pred_tag_sum"], \
                                                                   results["pred_neg_tag_sum"]

                cpt_invalid_sum_list.append(cpt_invalid_sum.sum().cpu().detach().numpy().tolist())
                cpt_needed_sum_list.append(cpt_needed_sum.sum().cpu().detach().numpy().tolist())
                if self.args.add_mlm_object:
                    anchor_tag_sum_list.append(anchor_tag_sum.sum().cpu().detach().numpy().tolist())
                    anchor_neg_tag_sum_list.append(neg_tag_sum.sum().cpu().detach().numpy().tolist())

                sent_string_token.extend(batch_x["sent_string"])
                candi_idx.extend(batch_x["anchor_idx"])
                candidate.extend(batch_x["anchor"])

                ground_cpt_for_char.extend([[[self.id2cpt[ii] for ii in tt] for tt in cpt_ids] for cpt_ids in batch_x["anchor_cpt_ids"]])
                tmp_pred_cpt = cpt_pred_idx.cpu().detach().numpy().tolist()

                pred_cpt.extend([[self.id2cpt[ii] for ii in tt] for tt in tmp_pred_cpt])

            save_preds_for_cpts(sent_string_token, candidate, candi_idx, pred_cpt, ground_cpt_for_char, output_dir)

            total_anchor_sum = sum(anchor_tag_sum_list) + sum(anchor_neg_tag_sum_list)
            self.logger.info("mask token predict correct num: {}, "
                                "neg sum: {}, ratio: {}".format(sum(anchor_tag_sum_list), sum(anchor_neg_tag_sum_list),
                                "{:.2f}".format(torch.true_divide(sum(anchor_tag_sum_list), total_anchor_sum))))

            total_sum = sum(cpt_needed_sum_list) + sum(cpt_invalid_sum_list)
            self.logger.info("total needed sum: {} and invalid sum: {}, accuracy rate: {}\n".format(
                sum(cpt_needed_sum_list), sum(cpt_invalid_sum_list),
                sum(cpt_needed_sum_list) / total_sum ))


    def test_with_ACE(self, eval_type):
        self.logger.info(
            "************************pause in step : {}, start to {} evaluation.*************************\n".format(
                self.step, eval_type))

        # eval
        output_dir = os.path.join(self.args.checkpoint_dir, "checkpoint-test-{}".format(self.args.test_logging_name))
        self.ddp_model.eval()
        pred_et_id = []
        pred_cpt = []
        ground_cpt_for_et = []
        ground_cpt_for_char = []
        DS_cpt_ids = []
        ground_et = []
        sent_string = []
        sent_string_token = []
        candi_idx = []
        candidate = []
        sample_tag = []

        with self.ddp_model.no_sync():
            for (batch_x, batch_y) in self.dev_ace_loader:
                token_ids = batch_x["input_ids"].to(self.device)
                type_ids = batch_x["input_segment_ids"].to(self.device)
                mask_ids = batch_x["input_mask"].to(self.device)
                anchor_idx = batch_x["anchor_idx"].to(self.device)

                anchor_cpt_idx = batch_y["anchor_cpt_idx"].to(self.device)
                anchor_cpt_idx_masks = batch_y["anchor_cpt_idx_masks"].to(self.device)

                # predict concept score and id for each sample
                results = self.model(token_ids,
                                     type_ids,
                                     mask_ids,
                                     anchor_target_idx=None,
                                     anchor_target_labels=None,
                                     anchor_target_mask=None,
                                     cpt_random_idx=None,
                                     cpt_idx=anchor_cpt_idx,
                                     cpt_idx_masks=anchor_cpt_idx_masks,
                                     anchor_idx=anchor_idx,
                                     anchor_cpt_masks=None,
                                     span_tag=None,
                                     is_eval=eval_type)

                pred_cpt_idx = results["pred_cpt_idx"].cpu().detach().numpy().tolist()

                sent_string_token.extend(batch_x["sent_string"])
                sent_string.extend(batch_x["sent_ori_string"])
                candi_idx.extend(batch_x["anchor_idx"])
                candidate.extend(batch_x["anchor"])

                ground_et.extend(batch_y["et_id"])
                ground_cpt_for_char.extend(batch_x["anchor_cpt_ids"])
                DS_cpt_ids.extend(batch_y["ds_cpt_ids"])
                sample_tag.extend(batch_y["is_positive"])

                # ground_cpt_for_char.extend(
                #     [[self.id2cpt[ii] for ii in cpt_ids] for cpt_ids in current_batch["anchor_cpt_ids"]])
                #
                tmp_pred_et_id = []
                for pred_cpt_id in pred_cpt_idx:
                    if pred_cpt_id not in self.cpt_id2et_id:
                        tmp_pred_et_id.append([9])
                    else:
                        tmp_pred_et_id.append(self.cpt_id2et_id[pred_cpt_id])

                pred_cpt.extend(pred_cpt_idx)
                pred_et_id.extend(tmp_pred_et_id)
            #
            self.logger.info("\ntest data check! \n")
            # print("ground et: ", ground_et[:20])
            # print("ground cpt for char: ", DS_cpt_ids[:20])
            # print("pred cpt :", pred_cpt[:20])
            # print("pred et: ", pred_et_id[:20])
            # print("is positive: ", sample_tag[:20])

            save_preds(sent_string, sent_string_token, candidate, candi_idx, pred_cpt, DS_cpt_ids, pred_et_id, ground_et,
                       sample_tag, self.id2cpt, self.id2et,
                       output_dir)

            ground_event_type = [int(et) for et in ground_et]

            acc, P, R, F1 = metrics(pred_et_id, ground_event_type, sample_tag)

        return acc, P, R, F1


