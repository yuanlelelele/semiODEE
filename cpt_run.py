#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2021/1/19 21:04

import sys
import os
import time
from collections import defaultdict
import argparse
# import logging
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from fastNLP import logger, init_logger_dist

import config
import data_utils.arg_otions as arg_options
from model.cpt_embeddings import CptEmbedding
from DataSet.prep_data import load_annotated_concepts_new, load_verb_concepts
from DataSet.my_dataset import MyData, MyTestData
from DataSet.prep_cpt_data import InputAnchorExample, InputAnchorExample_testing
from transformers import BertTokenizer, BertModel, BertConfig
from model.dist_model import commonCptODEE
from CptTrain import Trainer

# logger = logging.getLogger()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():

    parser = argparse.ArgumentParser()
    arg_options.add_path_options(parser)
    arg_options.add_para_options(parser)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    n_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    print("num gpus: {}".format(n_gpus))
    is_distributed = n_gpus > 1
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        args.world_size = dist.get_world_size()
        args.local_rank = int(args.local_rank)
        # synchronize()

    # Setup logging
    log_file_path = os.path.join(args.log_output_dir, 'log-{}.txt'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    init_logger_dist()
    logger.add_file(log_file_path, level='INFO')
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # logging_fh = logging.FileHandler(log_file_path)
    # logging_fh.setLevel(logging.DEBUG)
    # logger.addHandler(logging_fh)

    args.test_logging_name = log_file_path.split('/')[-1].split('.')[0].replace('log-', '')
    print(log_file_path.split('/')[-1].split('.')[0].replace('log-', ''))

    # cpt prep data load
    if args.local_rank == 0:
        print("Load prep data...\n")
    cpt2words, cpt2id, et2id, sememes, cpt_tree = load_annotated_concepts_new()
    verb_cpts = load_verb_concepts()
    CptEmb = CptEmbedding(sememes, cpt2words, cpt2id, et2id, cpt_tree, args.cpt_max_num, args.random_cpt_num)

    word2cpt_ids = CptEmb.word2cpt_idx
    verb_cpt_ids = [cpt2id[cc] for cc in verb_cpts]
    sememe2id = CptEmb.sememe2id

    cpt_vec = torch.load(CptEmb.cpt_vec_in_bert_file)[:34442]  # [:10907]   # [:34442]    # 34443 * 768, padding index = 34442
    logger.info("cpt embedding file: {}".format(CptEmb.cpt_vec_in_bert_file))
    logger.info("cpt vec length: {}".format(len(cpt_vec)))

    et2cpts = CptEmb.et2cpts
    cpt2center_sem = CptEmb.cpt2center_sem
    cpt_id2center_sem_id = {cpt2id[cc]: sememe2id[sem] for cc, sem in cpt2center_sem.items()}

    id2cpt = {idx: cc for cc, idx in cpt2id.items()}
    id2et = {id: et for et, id in et2id.items()}
    anno_cpt2et = defaultdict(list)
    et_id2cpt_ids = defaultdict(list)
    for et, cpts in et2cpts.items():
        # print(self.et_id2cpt_ids)
        for cc in cpts:
            anno_cpt2et[cc].append(et)
            et_id2cpt_ids[et2id[et]].append(cpt2id[cc])
    cpt_id2et_id = {cpt2id[cc]: [et2id[et] for et in ets] for cc, ets in anno_cpt2et.items()}

    args.cpt_num = CptEmb.cpt_num
    logger.info("cpt nums: {}\n".format(args.cpt_num))
    logger.info("HowNet words cnt: {}".format(len(word2cpt_ids)))

    # pred DataSet

    train_samples = MyData(args, 'train', args.world_size, args.local_rank)
    dev_samples = MyData(args, 'dev', args.world_size, args.local_rank)
    dev_ace_samples = MyTestData(args, os.path.join(config.cached_data_dir, "cached_devACE_fixed_samples"), args.local_rank)
    logger.info("rank {} / {} load dataset with length: {}.".format(args.local_rank, args.world_size, len(train_samples)))
    test_ace_samples = None
    # ************** train data ************************
    train_sampler = DistributedSampler(train_samples, rank=args.local_rank, num_replicas=args.world_size)
    train_loader = DataLoader(train_samples, batch_size=args.per_gpu_train_batch_size,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   num_workers=args.num_workers,
                                   collate_fn=train_samples.collate_fn)
    # ************** dev data ************************
    dev_loader = DataLoader(dev_samples, batch_size=args.per_gpu_dev_batch_size,
                                 collate_fn=dev_samples.collate_fn)
    dev_ace_loader = DataLoader(dev_ace_samples, batch_size=args.per_gpu_eval_batch_size,
                                     collate_fn=dev_ace_samples.collate_fn)
    # ************** test data ************************
    # self.test_loader = DataLoader(test_ace_samples, batch_size=args.per_gpu_eval_batch_size,
    #                               collate_fn=test_ace_samples.collate_fn)

    # ************** init model ***************************
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    bert_config = BertConfig.from_pretrained(args.bert_model_dir)
    bert_config.is_decoder = False
    cpt_model = commonCptODEE(args, bert_config, cpt_vec, len(cpt_vec[0]))

    # pred Trainer
    trainer = Trainer(args=args,
                      train_samples=train_loader,
                      dev_samples=dev_loader,
                      dev_ace_samples=dev_ace_loader,
                      test_ace_samples=None,
                      cpt_model=cpt_model,
                      id2cpt=id2cpt,
                      id2et=id2et,
                      cpt_id2et_id=cpt_id2et_id)
    trainer.train()


if __name__ == '__main__':
    main()