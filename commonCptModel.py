#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/12/22 15:26
import time

import torch
import numpy as np
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_bert import BertLayerNorm, gelu

from cpt_embeddings import CptMap
from transformers import BertModel, BertForMaskedLM


class ClsHead(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout=0.15):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.layer_norm = BertLayerNorm(hidden_size)

        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class commonCptODEE(nn.Module):

    def __init__(self, args, cpt_vec, cpt_embedding_size):
        super(commonCptODEE, self).__init__()
        bert_config = BertConfig.from_pretrained(args.bert_model_dir)
        bert_config.is_decoder = False

        self.args = args
        self.cpt_num = args.cpt_num
        self.cpt_embedding_size = cpt_embedding_size

        self.cpt_emb_model = CptMap(args, cpt_vec)

        self.format_logits = ClsHead(self.args.embedding_dim, self.args.hidden_size, self.args.out_size)
        self.format_cpt_logits = ClsHead(self.args.embedding_dim, self.args.hidden_size, self.args.out_size)

        self.bert_mlm = BertForMaskedLM(bert_config)
        self.full_tensor = nn.Parameter(torch.Tensor([-10000.]), requires_grad=False)
        self.full_max_tensor = nn.Parameter(torch.Tensor([10000.]), requires_grad=False)
        self.one_tensor = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.zero_tensor = nn.Parameter(torch.Tensor([0]), requires_grad=False)

        self.beta = nn.Linear(self.args.embedding_dim * 2, 1)

        self.bert_model = BertModel.from_pretrained(args.bert_model_dir, config=bert_config)

        self.cos_similarity = nn.CosineSimilarity(dim=-1)  # [-1,1]

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.nllLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def _compute_span_max_pools(self, feas):
        batch_size, seq_length, emb_dim = feas.size()
        start_idx = torch.arange(seq_length).cuda().view(seq_length, 1).expand(seq_length, self.args.span_len)
        span_length = torch.arange(self.args.span_len).cuda().\
            view(1, self.args.span_len).expand(seq_length,  self.args.span_len).clone()
        last_span_length = torch.flip(torch.tril(torch.ones(self.args.span_len, self.args.span_len).cuda()).unsqueeze(0),
                                      [0, 1]).squeeze(0)
        span_length[- self.args.span_len:] = span_length[- self.args.span_len:] * last_span_length.long()

        span_idx = (start_idx + span_length).unsqueeze(0).expand(batch_size, seq_length, self.args.span_len).view(
            batch_size, -1)
        span_emb = torch.gather(feas, 1, span_idx.unsqueeze(2).expand(
            batch_size, seq_length * self.args.span_len, emb_dim)).view(
            batch_size, seq_length, self.args.span_len, emb_dim)
        span_emb = span_emb.unsqueeze(2).expand(batch_size, seq_length, self.args.span_len, self.args.span_len, emb_dim)
        span_mask = torch.tril(torch.ones(self.args.span_len, self.args.span_len).cuda()).unsqueeze(0).expand(seq_length,
                                                                                                       self.args.span_len,
                                                                                                       self.args.span_len)
        span_emb = span_emb + (span_mask.unsqueeze(0).unsqueeze(-1) == 0).float() * (
            -1e30)  # B * T * span_len * span_len * dim
        span_logits = span_emb.max(dim=3)[0]  # B * T * span_len * dim

        return span_logits

    def _compute_emission(self, feas, seq_mask, span_cpt_idx, span_cpt_masks):
        batch_size, seq_length, emb_dim = feas.size()
        span_logits = self._compute_span_max_pools(feas)    # B * T * span_len * dim

        sent_span_num, cpt_num = span_cpt_idx.size(1), span_cpt_idx.size(2)
        assert sent_span_num == seq_length * self.args.span_len
        span_cpt_emb = self.cpt_emb_model(span_cpt_idx.view(-1, self.args.span_max_cpt_num))  # (-1, span_max_cpt_num, emb_dim)
        # span_logits = torch.bmm(span_logits.view(-1, 1, emb_dim), span_cpt_emb.transpose(2, 1)).\
        #     squeeze(1).view(batch_size, seq_length, self.args.span_len, self.args.span_max_cpt_num)
        # span_logits = torch.div(span_logits, self.args.scale)
        span_logits = self.cos_similarity(span_logits.view(-1, 1, emb_dim), span_cpt_emb).view(batch_size, seq_length, self.args.span_len, self.args.span_max_cpt_num)
        # span_logits = self.sigmoid(span_logits)
        span_logits = span_logits.masked_fill(span_cpt_masks.view(batch_size, seq_length, self.args.span_len, cpt_num) == 0, -1e30)
        span_logits, cpt_max_idx = torch.max(span_logits, dim=-1)  # B * seq_len * span_len
        pred_cpt_idx = torch.gather(span_cpt_idx.view(batch_size, seq_length, self.args.span_len, -1),
                                    -1, cpt_max_idx.unsqueeze(3)).squeeze(3)

        seq_last_idx = torch.sum(seq_mask, dim=-1).view(-1) - 1  # B
        span_logits.scatter_(1, seq_last_idx.view(batch_size, 1, 1).expand(batch_size, seq_length, self.args.span_len),
                             src=torch.zeros(batch_size, seq_length, self.args.span_len).cuda())
        span_logits[:, 0, :] = 0
        e_scores = span_logits * seq_mask.unsqueeze(2).expand(batch_size, seq_length, self.args.span_len)

        return e_scores, pred_cpt_idx

    def _compute_transition(self, feas, seq_mask):
        batch_size, seq_length, emb_dim = feas.size()
        span_logits = self._compute_span_max_pools(feas)    # B * T * span_len * dim
        next_span_length = torch.arange(1, self.args.span_len + 1).cuda().view(1, self.args.span_len).expand(seq_length, self.args.span_len)
        next_span_start_idx = torch.arange(seq_length).cuda().view(seq_length, 1).expand(seq_length, self.args.span_len).clone()
        last_span_idx = torch.flip((torch.triu(torch.ones(self.args.span_len, self.args.span_len).cuda()) == 0).float().unsqueeze(0),
                                   [0,1]).squeeze(0)
        next_span_start_idx = (next_span_start_idx + next_span_length)
        next_span_start_idx[-self.args.span_len:] *= last_span_idx.long()
        next_span_start_idx = next_span_start_idx.view(1, seq_length, self.args.span_len, 1).\
            expand(batch_size, seq_length, self.args.span_len, emb_dim).view(batch_size, -1, emb_dim)

        next_span_logits = torch.gather(span_logits, 1, next_span_start_idx.
                                        unsqueeze(2).expand(batch_size, seq_length*self.args.span_len, self.args.span_len, emb_dim))
        next_span_logits = next_span_logits.view(batch_size, seq_length, self.args.span_len, self.args.span_len, emb_dim).contiguous()
        cur_span_logits = span_logits.unsqueeze(3).contiguous()
        relative_span_logits = torch.cat((cur_span_logits.view(-1, 1, emb_dim).expand(-1, self.args.span_len, emb_dim),
                                          next_span_logits.view(-1, self.args.span_len, emb_dim)), dim=-1).view(-1, emb_dim*2)
        t_scores = self.beta(relative_span_logits).view(batch_size, seq_length, self.args.span_len, self.args.span_len)
        # t_scores = torch.matmul(cur_span_logits.view(-1, 1, emb_dim), next_span_logits.view(-1, self.args.span_len, emb_dim).transpose(2,1))\
        #     .squeeze(1).view(batch_size, seq_length, self.args.span_len, self.args.span_len)
        t_scores = t_scores * seq_mask.view(batch_size, seq_length, 1, 1).expand(batch_size, seq_length, self.args.span_len, self.args.span_len)

        return t_scores

    def _forward_path(self, e_scores, t_scores, seq_mask, batch_size, seq_size):
        """
        :param e_scores: B * seq_size * 3
        :param t_scores: B * seq_size * 3 * 3
        """
        fp_scores = torch.zeros(batch_size, seq_size, self.args.span_len).cuda()
        for i in range(1, seq_size):
            for j in range(3):
                if i == 1:
                    fp_scores[:, i, j] = e_scores[:, i, j]
                    continue
                before = torch.empty((batch_size, 3)).cuda()
                before[:, 0] = (fp_scores[:, i-1, 0] + t_scores[:, i-1, 0, j] + e_scores[:, i, j]) if i-1 > 0 else torch.Tensor([[-1e30]*batch_size]).cuda().view(-1)
                before[:, 1] = (fp_scores[:, i-2, 1] + t_scores[:, i-2, 1, j] + e_scores[:, i, j]) if i-2 > 0 else torch.Tensor([[-1e30]*batch_size]).cuda().view(-1)
                before[:, 2] = (fp_scores[:, i-3, 2] + t_scores[:, i-3, 2, j] + e_scores[:, i, j]) if i-3 > 0 else torch.Tensor([[-1e30]*batch_size]).cuda().view(-1)
                fp_scores[:, i, j] = torch.max(before, dim=-1)[0]
        fp_scores = fp_scores * seq_mask.view(batch_size, seq_size, 1).expand(batch_size, seq_size, 3)

        return fp_scores

    def _backward_path(self, e_scores, t_scores, seq_mask, batch_size, seq_size):
        """
        :param e_scores: B * seq_size * 3
        :param t_scores: B * seq_size * 3 * 3
        """
        bp_scores = torch.zeros(batch_size, seq_size, self.args.span_len).cuda()
        for i in range(seq_size - 1, 0, -1):
            for j in range(3):
                if i == seq_size-1:
                    bp_scores[:, i, j] = e_scores[:, i, j]
                    continue
                before = torch.empty((batch_size, 3)).cuda()
                before[:, 0] = (bp_scores[:, i+j+1, 0] + t_scores[:, i, j, 0] + e_scores[:, i, j]) if i+j+1 < seq_size-1 else torch.Tensor([[-1e30]*batch_size]).cuda().view(-1)
                before[:, 1] = (bp_scores[:, i+j+1, 1] + t_scores[:, i, j, 1] + e_scores[:, i, j]) if i+j+2 < seq_size-1 else torch.Tensor([[-1e30]*batch_size]).cuda().view(-1)
                before[:, 2] = (bp_scores[:, i+j+1, 2] + t_scores[:, i, j, 2] + e_scores[:, i, j]) if i+j+3 < seq_size-1 else torch.Tensor([[-1e30]*batch_size]).cuda().view(-1)
                before = before * seq_mask[:, i].view(batch_size, 1).expand(batch_size, 3)
                bp_scores[:, i, j] = torch.max(before, dim=-1)[0]
                # bp_scores[:, i, j] = torch.max(torch.stack((before_1, before_2, before_3), dim=-1).cuda(), dim=-1)[0] + e_scores[:, i, j]
        # bp_scores = bp_scores * seq_mask.view(batch_size, seq_size, 1).expand(batch_size, seq_size, 3)
        return bp_scores

    def _compute_loss(self, anchor_idx_masks, anchor_span_idx, pos_span_masks, neg_span_masks, fp_scores, bp_scores, e_scores):
        """
        :param anchor_idx_masks:  B * max_anchor_num
        :param anchor_span_idx: B * max_anchor_num * max_span_num * 2
        :param pos_span_masks: B * max_anchor_num * max_span_num
        :param fp_scores: B * seq_size * 3
        :param bp_scores: B * seq_size * 3
        :param e_scores: B * seq_size * 3
        """
        batch_size = anchor_span_idx.size(0)
        anchor_num = anchor_span_idx.size(1)
        span_num = anchor_span_idx.size(2)

        anchor_start_idx = anchor_span_idx[:, :, :, 0].view(batch_size, anchor_num*span_num)
        anchor_span_length = anchor_span_idx[:, :, :, 1].view(batch_size, anchor_num*span_num)
        fp_selected_span = torch.gather(fp_scores, 1, anchor_start_idx.unsqueeze(2).expand(batch_size, anchor_num*span_num, self.args.span_len))
        fp_selected_scores = torch.gather(fp_selected_span, 2, anchor_span_length.unsqueeze(2)).squeeze(2).view(batch_size, anchor_num, span_num)

        bp_selected_span = torch.gather(bp_scores, 1, anchor_start_idx.unsqueeze(2).expand(batch_size, anchor_num*span_num, self.args.span_len))
        bp_selected_scores = torch.gather(bp_selected_span, 2, anchor_span_length.unsqueeze(2)).squeeze(2).view(batch_size, anchor_num, span_num)

        anchor_span_scores = fp_selected_scores + bp_selected_scores
        pos_spans = anchor_span_scores.masked_fill(pos_span_masks == 0, -10000.)
        # print("pos spans: {}".format(pos_spans))
        neg_spans = anchor_span_scores.masked_fill(neg_span_masks == 0, -10000.)
        # print("neg spans: {}".format(neg_spans))

        # anchor_span_idx = anchor_span_idx.view(batch_size, -1, 2)
        # anchor_span_scores = torch.zeros(batch_size, anchor_num * span_num).cuda() # B * anchor_num * span_num
        # for b in range(batch_size):
        #     for ii in range(anchor_num * span_num):
        #         start_idx = anchor_span_idx[b, ii, 0].item()
        #         span_len = anchor_span_idx[b, ii, 1].item()
        #         anchor_span_scores[b][ii] = fp_scores[b][start_idx][span_len] + bp_scores[b][start_idx][span_len] # - e_scores[b][start_idx][span_len]
        # # print("span scores: {}".format(anchor_span_scores[0]))
        # # anchor_span_scores = torch.log(anchor_span_scores.view(batch_size, anchor_num, span_num) + self.args.log_eps)
        # anchor_span_scores = anchor_span_scores.view(batch_size, anchor_num, span_num)
        # pos_spans = anchor_span_scores.masked_fill(pos_span_masks == 0, -10000.)
        # # print("pos spans: {}".format(pos_spans))
        # neg_spans = anchor_span_scores.masked_fill(neg_span_masks == 0, -10000.)
        # # print("neg spans: {}".format(neg_spans))

        max_pos_spans = torch.max(pos_spans, -1)[0] * anchor_idx_masks    # B * anchor_num
        max_neg_spans = torch.max(neg_spans, -1)[0] * anchor_idx_masks   # B * anchor_num

        loss = self.relu(self.args.margin_value - (max_pos_spans.view(-1) - max_neg_spans.view(-1)))
        # print("loss: {}".format(loss))
        loss = (loss * anchor_idx_masks.view(-1)).sum() / anchor_idx_masks.view(-1).sum()
        return loss

    def _forward_train(self, token_ids, type_ids, mask_ids,
                       anchor_target_idx, anchor_target_labels, anchor_target_mask,
                       anchor_masks, anchor_span_ids, pos_span_masks, neg_span_masks, span_cpt_idx, span_cpt_masks):
        bert_output = self.bert_model(input_ids=token_ids,
                                      attention_mask=mask_ids,
                                      token_type_ids=type_ids)
        bert_fea = bert_output[0]
        batch_size = bert_fea.size(0)
        seq_size = bert_fea.size(1)
        emb_size = bert_fea.size(2)
        bert_fea = self.format_logits(bert_fea.view(-1, emb_size)).view(batch_size, seq_size, emb_size)

        # torch.cuda.synchronize()
        # start_tt = time.time()
        e_scores, _ = self._compute_emission(bert_fea, mask_ids, span_cpt_idx, span_cpt_masks)
        # torch.cuda.synchronize()
        # end_tt = time.time()
        # print("time cost in emission: {}".format(end_tt - start_tt))
        # print("emissions: {}".format(e_scores[0][:5]))

        # torch.cuda.synchronize()
        # start_tt = time.time()
        t_scores = self._compute_transition(bert_fea, mask_ids)
        # torch.cuda.synchronize()
        # end_tt = time.time()
        # print("time cost in transition: {}".format(end_tt - start_tt))
        # print("transitions: {}".format(t_scores[0][:5]))

        # torch.cuda.synchronize()
        # start_tt = time.time()
        forward_path_scores = self._forward_path(e_scores, t_scores, mask_ids, batch_size, seq_size)
        # torch.cuda.synchronize()
        # end_tt = time.time()
        # print("time cost in fp: {}".format(end_tt - start_tt))
        # print("fp score: {}".format(forward_path_scores[0][:5]))

        # torch.cuda.synchronize()
        # start_tt = time.time()
        backward_path_scores = self._backward_path(e_scores, t_scores, mask_ids, batch_size, seq_size)
        # torch.cuda.synchronize()
        # end_tt = time.time()
        # print("time cost in bp: {}".format(end_tt - start_tt))
        # print("bp score: {}".format(backward_path_scores[0][:5]))

        # torch.cuda.synchronize()
        # start_tt = time.time()
        loss = self._compute_loss(anchor_masks, anchor_span_ids, pos_span_masks, neg_span_masks, forward_path_scores, backward_path_scores, e_scores)
        # torch.cuda.synchronize()
        # end_tt = time.time()
        # print("time cost in loss: {}".format(end_tt - start_tt))

        return {"loss": loss}

    def _viterbi_decode(self, e_scores, e_idx, t_scores, mask):
        batch_size, seq_size, _ = e_scores.size()
        seq_length = torch.sum(mask, dim=-1)    # B
        fp_scores = torch.empty(batch_size, seq_size, 3).cuda()
        history = []
        invalid_his = torch.zeros(batch_size, 2).cuda().long()
        for j in range(3):
            history.append(invalid_his)
        for i in range(1, seq_size):
            for j in range(3):
                if i == 1:
                    fp_scores[:, i, j] = e_scores[:, i, j]
                    history.append(invalid_his)
                    continue
                before = torch.empty((batch_size, 3)).cuda()
                before[:, 0] = (fp_scores[:, i - 1, 0] + t_scores[:, i - 1, 0, j] + e_scores[:, i, j]) if i - 1 > 0 else torch.Tensor([[-1e30] * batch_size]).cuda().view(-1)
                before[:, 1] = (fp_scores[:, i - 2, 1] + t_scores[:, i - 2, 1, j] + e_scores[:, i, j]) if i - 2 > 0 else torch.Tensor([[-1e30] * batch_size]).cuda().view(-1)
                before[:, 2] = (fp_scores[:, i - 3, 2] + t_scores[:, i - 3, 2, j] + e_scores[:, i, j]) if i - 3 > 0 else torch.Tensor([[-1e30] * batch_size]).cuda().view(-1)
                fp_scores[:, i, j], path_idx = torch.max(before, dim=-1)
                start_idx = - path_idx + i - 1
                max_span = torch.stack((start_idx, path_idx), dim=0).cuda().transpose(1, 0)    # B * 2
                history.append(max_span)

        history = torch.stack(history, dim=0).transpose(1, 0).contiguous().view(batch_size, seq_size, 3, 2)
        best_path_list = []
        path_cpt_id_list = []
        for b in range(batch_size):
            check_idx = seq_length[b].item()-1
            last_span = history[b][check_idx][0]
            best_path_span = [[last_span[0].item(), last_span[1].item()]]
            check_idx, check_length = last_span[0].item(), last_span[1].item()
            while check_idx > 1:
                last_span = history[b][check_idx][check_length]
                best_path_span.append([last_span[0].item(), last_span[1].item()])
                check_idx, check_length = last_span[0].item(), last_span[1].item()
            best_path_span.append([0, 0])
            best_path_span.reverse()
            best_path_list.append(best_path_span)
            path_cpt_idx = [0]
            for tmp_span in best_path_span[1:]:
                path_cpt_idx.append(e_idx[b][tmp_span[0]][tmp_span[1]].item())
            path_cpt_id_list.append(path_cpt_idx)

        return best_path_list, path_cpt_id_list

    def _forward_eval(self, token_ids, type_ids, mask_ids, span_cpt_idx, span_cpt_masks):
        bert_output = self.bert_model(input_ids=token_ids,
                                      attention_mask=mask_ids,
                                      token_type_ids=type_ids)
        bert_fea = bert_output[0]
        e_scores, e_idx = self._compute_emission(bert_fea, mask_ids, span_cpt_idx, span_cpt_masks)
        t_scores = self._compute_transition(bert_fea, mask_ids)
        best_path_list, path_cpt_id_list = self._viterbi_decode(e_scores, e_idx, t_scores, mask_ids)
        return {"best_path_list": best_path_list,
                "path_cpt_id_list": path_cpt_id_list}

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        return self._forward_eval(*args, **kwargs)

    def _forward_old(self, token_ids, type_ids, mask_ids, anchor_target_idx, anchor_target_labels, anchor_target_mask, cpt_random_idx, cpt_idx, cpt_idx_masks, anchor_idx, eval_type):

        bert_output = self.bert_model(input_ids=token_ids,
                                      attention_mask=mask_ids,
                                      token_type_ids=type_ids)
        bert_fea = bert_output[0]

        batch_size = bert_fea.size(0)
        emb_size = bert_fea.size(2)
        if len(anchor_idx.size()) == 2:
            anchor_num = anchor_idx.size(1)
        else:
            anchor_num = 1

        if self.args.add_mlm_object:
            if eval_type != 'test':
                target_size = anchor_target_idx.size(1)
                target_fea = torch.gather(bert_fea, 1,
                                          anchor_target_idx.view(batch_size, target_size, 1).expand(batch_size, target_size, emb_size)).view(-1, emb_size)
                # print("bert_fea after gather: {}".format(bert_fea.size()))

                anchor_logits = self.bert_mlm.cls(target_fea)  # (B * target_size) * vocab_size
                anchor_pred_idx = torch.max(anchor_logits, -1)[1]  # B * target_size

                pred_tag = torch.where(anchor_pred_idx.view(-1) == anchor_target_labels.view(-1), self.one_tensor,
                                       self.zero_tensor) * anchor_target_mask.view(-1)
                pred_neg_tag = torch.where(anchor_pred_idx.view(-1) != anchor_target_labels.view(-1), self.one_tensor,
                                           self.zero_tensor) * anchor_target_mask.view(-1)
                pred_tag_sum = pred_tag.sum()
                pred_neg_tag_sum = pred_neg_tag.sum()

                if eval_type == 'train':
                    mlm_loss = self.nllLoss(anchor_logits.view(-1, self.args.token_size), anchor_target_labels.view(-1))

        anchor_feature = torch.gather(bert_fea, 1, anchor_idx.view(batch_size, anchor_num, 1).
                                      expand(batch_size, anchor_num, emb_size)).view(-1, emb_size)
        anchor_feature = self.format_logits(anchor_feature)
        batch_size = anchor_feature.size(0)
        emb_size = anchor_feature.size(1)

        if eval_type == 'dev':
            anchor_max_num = cpt_idx.size(1)
            total_cpt_idx = torch.LongTensor([i for i in range(self.args.cpt_num)]).view(self.args.cpt_num, 1).cuda()
            cpt_embs = self.cpt_emb_model(total_cpt_idx)  # cpt_total_num * emb_size

            anchor_logits = torch.bmm(anchor_feature.view(batch_size, 1, emb_size),
                                      cpt_embs.view(1, self.args.cpt_num, emb_size).
                                      expand(batch_size, self.args.cpt_num, emb_size).
                                      transpose(2, 1)).squeeze(1)  # B * cpt_total_num
            anchor_logits = self.softmax(anchor_logits)
            max_pred_logits, max_pred_id = torch.max(anchor_logits, -1)
            score_tag = torch.where(
                max_pred_id.view(batch_size, 1).expand(batch_size, self.args.cpt_max_num) == cpt_idx,
                self.one_tensor, self.zero_tensor)
            score_tag = torch.max(score_tag, -1)[0].view(-1)
            score_neg_tag = (1 - score_tag).bool()

            score_tag_needed_sum = score_tag.sum()
            score_tag_invalid_sum = score_neg_tag.sum()

            if self.args.add_mlm_object:
                return {"pred_tag_sum": pred_tag_sum,
                        "pred_neg_tag_sum": pred_neg_tag_sum,
                        "max_pred_id": max_pred_id.view(-1, anchor_max_num),
                        "score_tag_invalid_sum": score_tag_invalid_sum,
                        "score_tag_needed_sum": score_tag_needed_sum}
            else:
                return {"max_pred_id": max_pred_id.view(-1, anchor_max_num),
                        "score_tag_invalid_sum": score_tag_invalid_sum,
                        "score_tag_needed_sum": score_tag_needed_sum}

        elif eval_type == 'test':
            labeled_cpt_embs = self.cpt_emb_model(cpt_idx)  # B * cpt_max_num * cpt_emb_size

            anchor_labeled_logits = torch.bmm(anchor_feature.view(batch_size, 1, emb_size),
                                              labeled_cpt_embs.transpose(2, 1)).squeeze(1)
            anchor_labeled_logits = torch.where(cpt_idx_masks == 1, anchor_labeled_logits, self.full_tensor)
            anchor_labeled_logits = self.softmax(anchor_labeled_logits)

            max_index = torch.argmax(anchor_labeled_logits, -1)
            pred_cpt_index = torch.gather(cpt_idx, 1, max_index.view(batch_size, 1)).squeeze()

            return {"pred_cpt_idx": pred_cpt_index}

        anchor_max_num = cpt_idx.size(1)

        cpt_idx = cpt_idx.view(-1, self.args.cpt_max_num)
        cpt_idx_masks = cpt_idx_masks.view(-1, self.args.cpt_max_num)
        cpt_random_idx = cpt_random_idx.view(-1, self.args.random_cpt_num)

        labeled_cpt_embs = self.cpt_emb_model(cpt_idx)  # -1 * cpt_max_num * cpt_emb_size
        random_cpt_embs = self.cpt_emb_model(cpt_random_idx)  # -1 * random_cpt_max_num * cpt_emb_size
        assert labeled_cpt_embs.size(2) == emb_size

        anchor_labeled_logits = self.cos_similarity(anchor_feature.view(batch_size, 1, emb_size), labeled_cpt_embs)
        # anchor_labeled_logits = torch.log(torch.sigmoid(anchor_labeled_logits))
        # anchor_labeled_logits = torch.bmm(anchor_feature.view(batch_size, 1, emb_size),
        #                                   labeled_cpt_embs.transpose(2, 1)).squeeze(1)  # B * cpt_max_num
        # print("bmm: {}".format(anchor_labeled_logits[0]))
        anchor_labeled_logits = torch.div(anchor_labeled_logits, self.args.temperature)
        anchor_labeled_logits = torch.exp(anchor_labeled_logits)
        # print("div & exp: {}".format(anchor_labeled_logits[0]))
        anchor_labeled_logits = torch.where(cpt_idx_masks == 1, anchor_labeled_logits, self.full_tensor)
        target_logits, target_index = torch.max(anchor_labeled_logits, dim=-1)
        # target_logits = target_logits * cpt_idx_masks[:, 0].view(-1)
        # print("positive after softmax: {}".format(target_logits))

        anchor_random_logits = self.cos_similarity(anchor_feature.view(batch_size, 1, emb_size), random_cpt_embs)
        # anchor_random_logits = torch.log(torch.sigmoid(anchor_random_logits))
        # anchor_random_logits = torch.bmm(anchor_feature.view(batch_size, 1, emb_size),
        #                                  random_cpt_embs.transpose(2, 1)).squeeze(1)  # B *
        # print("bmm: {}".format(anchor_random_logits[0]))
        anchor_random_logits = torch.div(anchor_random_logits, self.args.temperature)
        anchor_random_logits = torch.exp(anchor_random_logits)  # B
        # print("div & exp: {}".format(anchor_random_logits[0]))
        max_random_logits, target_random_index = torch.max(anchor_random_logits, -1)
        target_random_logits = max_random_logits * cpt_idx_masks[:, 0].view(-1)
        anchor_random_logits = torch.sum(anchor_random_logits, dim=-1)
        random_logits = anchor_random_logits * cpt_idx_masks[:, 0].view(-1)
        # print("negative after softmax: {}".format(random_logits))
        denominator = target_logits + random_logits
        # print("denominator: {}".format(denominator[0]))

        loss = torch.mean(-torch.log(torch.div(target_logits, denominator)))
        # loss = self.relu(self.args.margin_value - (target_logits - target_random_logits)).mean()  # + cpt_loss

        score_tag_needed = torch.where(target_logits > target_random_logits, self.one_tensor, self.zero_tensor)
        score_tag_invalid = torch.where(target_logits < target_random_logits, self.one_tensor, self.zero_tensor)
        score_tag_needed_sum = score_tag_needed.sum()
        score_tag_invalid_sum = score_tag_invalid.sum()

        if self.args.add_mlm_object:
            loss += mlm_loss

        return {"loss": loss,
                "score_tag_invalid_sum": score_tag_invalid_sum,
                "max_labeled_index": target_index.view(-1, anchor_max_num),
                "max_random_index": target_random_index.view(-1, anchor_max_num),
                "score_tag_invalid": score_tag_invalid,
                "score_tag_needed_sum": score_tag_needed_sum}

    # "pred_tag_sum": pred_tag_sum,
    # "pred_neg_tag_sum": pred_neg_tag_sum,

