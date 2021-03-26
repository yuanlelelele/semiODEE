class SimpleTreeNode():

    def __init__(self, node, rel, candidates, et):
        self.node = node
        self.child = []
        self.p_rel = rel
        self.candidates = candidates
        self.eventType = et
        self.rel = []


class InputAnchorExample(object):
    """
    A single input with anchor index.
    Anchor_idx refer to index after adding special tokens.
    """
    def __init__(self, sent_str, sent_str_token, sent_seg_token,
                 anchor, anchor_idx, anchor_cpt_ids, mask_idx, mlm_label,
                 input_ids, input_mask, segment_ids):
        self.sent_str = sent_str
        self.sent_str_token = sent_str_token
        self.sent_seg_token = sent_seg_token

        self.anchor = anchor
        self.anchor_idx = anchor_idx
        self.anchor_cpt_ids = anchor_cpt_ids
        self.mask_idx = mask_idx
        self.mask_label = mlm_label

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class InputAnchorExample_testing(object):
    """A single test input with anchor index."""
    def __init__(self, sent_str, sent_str_token, is_positive,
                 anchor_pseudo_cpt_ids, anchor_cpt_ids, anchor_idx, anchor, trigger, et_id,
                 input_ids, input_mask, segment_ids):
        self.sent_str = sent_str
        self.sent_str_token = sent_str_token
        self.is_positive = is_positive

        self.anchor_idx = anchor_idx
        self.ds_cpt_ids = anchor_cpt_ids
        self.anchor_cpt_ids = anchor_pseudo_cpt_ids
        self.anchor = anchor
        self.trigger = trigger
        self.et_id = et_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids