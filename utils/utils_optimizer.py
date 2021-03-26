#!/user/bin/env python
# -*- coding utf-8 -*-
# @Time    :2020/7/22 17:27
# @Author  :Yuanll


from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup

def adam_optimizer(args, model, optim_mode, warmup_steps, t_total):

    if optim_mode == 'Adam':

        print('using optim Adam')
        parameters_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam(parameters_trainable, lr=args.lr,
                               eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return optimizer, scheduler

    elif optim_mode == 'AdamW':

        print('using optim AdamW')
        # Prepare optimizer and schedule (linear warmup and decay)
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", 'LayerNorm.bias', "LayerNorm.weight", 'layer_norm.bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                          eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return optimizer, scheduler