#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by JaLcy on 2020/10/16 21:14


import pickle


def _save(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
