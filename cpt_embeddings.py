#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/10/19 16:01

import numpy as np
import torch
import torch.nn as nn


class CptMap(nn.Module):

    def __init__(self, args, cpt_vec):
        super(CptMap, self).__init__()
        self.args = args
        self.cpt_emb = nn.Embedding.from_pretrained(cpt_vec, freeze=True)

    def forward(self, cpt_idx, generate_cpt_map=False):
        """

        :param cpt_idx: B * cpt_num
        :return: B * cpt_num * cpt_emb_size
        """
        #
        # if generate_cpt_map:
        #     return self._pre_constractive_map(cpt_idx)
        return self.cpt_emb(cpt_idx)


if __name__ == '__main__':
    ...
    # et2cpts, cpt2words, cpt2id, et2id, sememes, cpt_tree = load_annotated_concepts_new()
    # CptEmb = CptEmbedding(sememes, cpt2words, cpt2id, et2cpts, et2id, cpt_tree)
    # center_cpt2cpts = CptEmb.center_cpt2cpts
    #
    # cnt = 0
    # for center_cpt, cpts in center_cpt2cpts.items():
    #     cnt += len(cpts)
    # print(cnt)
    # print(CptEmb.ori_embedding[10])