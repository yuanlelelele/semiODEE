#!/user/bin/env python
# -*- coding utf-8 -*-
# @Time    :2020/7/9 10:13
# @Author  :Yuanll
import config
import numpy as np

def add_path_options(parser):

    server_path = "/data/yjl/semi_crfODEE"
    parser.add_argument('-bert_model_dir', type=str, default='/data/yjl/cpt_prep/pretrain_model')
    parser.add_argument('-train_checkpoint_dir', default='%s/output' % server_path)
    parser.add_argument('-checkpoint_dir', default='%s/output' % server_path)
    parser.add_argument('-model_checkpoint_dir', default='%s/output/model_checkpoint' % server_path)
    parser.add_argument('-log_output_dir', default='%s/output/log' % server_path)
    parser.add_argument('-test_logging_name', type=str)
    #log_output_dir

def add_para_options(parser):

    # model training process
    parser.add_argument('-mask_ratio', type=int, default=0.15)
    parser.add_argument('-verb_mask_ratio', type=int, default=0.05)
    parser.add_argument('-add_mlm_object', type=bool, default=True)
    parser.add_argument('-evaluate_during_training', default=False)
    parser.add_argument('-et_num', type=int, default=2000,
                        help="The minimum number for each event type samples in total samples.")
    parser.add_argument('-test_only', type=bool, default=False)
    parser.add_argument('-is_master', type=bool, default=False,
                        help='If true, we store the model state dict rely on accuracy.')
    parser.add_argument('-span_detect', type=bool, default=False,
                        help='If True, we experiment sequence modeling for trigger span detecting. ')

    # data config parameters
    parser.add_argument('-cpt_emb_tag', type=bool, default=True)

    parser.add_argument('-seq_max_length', type=int, default=128)
    parser.add_argument('-trigger_max_length', type=int, default=5)
    parser.add_argument('-random_cpt_num', type=int, default=200)
    parser.add_argument('-cpt_num', type=int)
    parser.add_argument('-span_len', type=int, default=4)
    parser.add_argument('-cpt_max_num', type=int, default=5)
    parser.add_argument('-span_max_cpt_num', type=int, default=40)
    parser.add_argument('-neg_span_max_num', type=int, default=100)
    parser.add_argument('-test_cpt_max_num', type=int, default=100)
    parser.add_argument('-train_batch_size', type=int, default=16)
    parser.add_argument('-dev_batch_size', type=int, default=32)
    parser.add_argument('-test_batch_size', type=int, default=16)
    parser.add_argument('-per_gpu_train_batch_size', type=int, default=32)
    parser.add_argument('-neg_ratio', type=int, default=6)
    parser.add_argument('-neg_sample_seed', type=int, default=233)
    parser.add_argument('-labeled_sample_seed', type=int, default=477)

    parser.add_argument('-num_layers', type=int, default=1)
    parser.add_argument('-bidirect_flag', type=bool, default=True)

    parser.add_argument('-embedding_dim', type=int, default=768)
    parser.add_argument('-hidden_size', type=int, default=400)
    parser.add_argument('-span_hidden_size', type=int, default=128)
    parser.add_argument('-out_size', type=int, default=200)
    parser.add_argument('-token_size', type=int, default=21128)

    # model loss computed
    parser.add_argument('-temperature', type=int, default=0.02)
    parser.add_argument("--log_eps", default=1e-6, type=float, help="Epsilon for log computation.")
    parser.add_argument('-scale', type=int, default=np.sqrt(768))
    parser.add_argument('-hidden_dropout_rate', default=0.1)
    parser.add_argument('-margin_value', default=5)
    parser.add_argument('-cpt_margin_value', default=0.8)
    parser.add_argument('-seed', type=int, default=2020, help="random seed for initialization")
    parser.add_argument('-shuffle_seed_1', type=int, default=477)
    parser.add_argument('-shuffle_seed_2', type=int, default=233)

    # optimizer hyper parameter
    parser.add_argument('-weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('-lr', type=float, default=0.00001, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="BETA2 for Adam optimizer.")

    parser.add_argument("--warmup_ratio", default=0.05, type=int, help="Linear warmup ratio over warmup_steps.")

    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--train_epoch_num", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1,type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('-logging_steps', type=int, default=4,
                        help="Number of eval model and save model.")
    parser.add_argument('-gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('-gradient_average', type=int, default=4)
    parser.add_argument('-train_record_steps', type=int, default=64)
    parser.add_argument('-dev_record_steps', type=int, default=512)
    parser.add_argument('-test_record_steps', type=int, default=512)

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_dev_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for developing.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    # GPU envs
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_id', default='0')
    parser.add_argument('--local_rank', type=int, help='node rank for distributed training')
    parser.add_argument('-world_size', type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fp16', type=str, default='O1')  # O1

    parser.add_argument('-one_sample', type=bool, default=True, help="run the project with only one sample.")

