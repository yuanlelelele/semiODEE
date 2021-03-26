#!/user/bin/env python
# -*- coding utf-8 -*-
# @Time    :2020/7/8 22:47
# @Author  :Yuanll

class LabelMap:
    def __init__(self, event_type_list):
        self.BIO = ['B-', 'I-']
        self.event_typpe = event_type_list

        self.BIO_label_list = []
        for t in self.event_typpe:
            self.BIO_label_list.extend([prefix+t for prefix in self.BIO])
        self.BIO_label_list.extend(['O'])
        self.BIO_label_list.extend([ 'START', 'END'])

        self.label_to_id = {label:idx for idx, label in enumerate(self.BIO_label_list)}
        self.id_to_label = {self.label_to_id[x]:x for x in self.label_to_id}

        self.label_size = len(self.label_to_id)
        self.START_TAG = self.label_to_id['START']
        self.END_TAG = self.label_to_id['END']
        self.NONE_TAG = self.label_to_id['O']

        self.span_types = self.get_span_group()

        self.TYPE_TAG = [id for id in self.id_to_label.keys()]
        # self.TYPE_TAG.remove(self.START_TAG)
        # self.TYPE_TAG.remove(self.END_TAG)
        self.TYPE_TAG.remove(self.NONE_TAG)

    def get_label_to_id(self):
        return self.label_to_id

    def get_id_to_label(self):
        return self.id_to_label

    def get_span_group(self):
        span_types = {}
        span_types[1] = " ".join(['B'])
        span_types[2] = " ".join(['B', 'I'])
        span_types[3] = " ".join(['B', 'I', 'I'])
        span_types[4] = " ".join(['B', 'I', 'I', 'I'])
        span_types[5] = " ".join(['B', 'I', 'I', 'I', 'I'])

        return span_types

    def get_label_group(self):
        pass

class SpanMap:
    def __init__(self):
        self.id2span = {
                        0: (0, 1),
                        1: (0, 2), 2: (1, 2),
                        3: (0, 3), 4: (1, 3), 5: (2, 3),
                        6: (0, 4), 7: (1, 4), 8: (2, 4), 9: (3, 4)}

        self.span2id = {self.id2span[x]: x for x in self.id2span.keys()}
        self.span_types = [x for x in self.span2id.keys()]  # (char_idx, token_length)


