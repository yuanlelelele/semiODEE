#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/10/16 19:43


# **************************** File Path *****************************
# server_path = "/home/yuanjiale"
server_path = "/data/yjl"
#server_path = "/data/yuanllll"
# server_path = "/data/yuanjl"

concept_data_dir = "%s/cpt_prep" % server_path

train_raw_text_path = "/data/yjl/DS_EE/NFETC_EE/data/train_corpus"
train_raw_text_outpath = "%s/cpt_prep/train_data" % server_path

doc_labeled_corpus = "%s/cpt_prep/train_data/zbn_cmn_labeled_text.pkl" % server_path
doc_neg_corpus = "%s/cpt_prep/train_data/zbn_cmn_neg_text.pkl" % server_path

all_test_sent_dir = "/data/yjl/DS_ODEE/dataset/test_data/all_test_sent_file.txt"
all_test_golden_dir = "/data/yjl/DS_ODEE/dataset/test_data/all_test_golden_file.txt"
doc_labeled_testing_corpus = "%s/cpt_prep/test_data/labeled_testing_test.pkl" % server_path

eventType2id_file = "%s/cpt_prep/test_data/label2id.dat" % server_path
et_id2triggers_file = "%s/cpt_prep/et_id2triggers.pkl" % server_path

original_sememe_embedding = concept_data_dir + "/cpt_embeddings/sememe-vec.txt"

cached_data_dir = "%s/cpt_prep/train_data" % server_path
cached_test_dir = "%s/cpt_prep/test_data" % server_path
output_dir = "%s/CptODEE/output" % server_path

test_logging_name = ''

# ************************ PRE_TRAIN MODEL **************************
# bert_model_dir = "/data/yjl/DS_ODEE/dataset/pretrain_model"
bert_model_dir = "%s/cpt_prep/pretrain_model/" % server_path
seg_model_dir = "/data/yjl/DS_ODEE/dataset/segment_model"


# ************************* Pre-train Parameters *********************
seq_len = 126
invalid_seq_len = 20
sample_seed = 477

selected_max_cnt = 800
selected_droped_cnt = 5
stored_sample_num = 200000
dev_sample_length = 40
test_sample_length = 640

