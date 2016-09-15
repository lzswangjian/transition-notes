#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from collections import defaultdict

logs = sys.stderr

LABEL_MAP = defaultdict(int)
WORD_MAP = defaultdict(int)


def read_raw_data(input_file):
    """record format: labelf1f2f3
    """
    YData = []
    XData = []
    for line in open(input_file):
        label, feats = line.strip().split('')
        if label not in LABEL_MAP:
            LABEL_MAP[label] = len(LABEL_MAP)
        YData.append(LABEL_MAP[label])

        xs = []
        for feat in feats.split(''):
            name, value = feat.split('=')
            if value not in WORD_MAP:
                WORD_MAP[value] = len(WORD_MAP)
            xs.append(WORD_MAP[value])
        XData.append(xs)
    print >> logs, 'read %d records' % len(XData)
    return np.array(XData), np.array(YData)


def read_data(input_file):
    YData = []
    XData = []
    for line in open(input_file):
        label, feats = line.strip().split('')
        YData.append(int(label))

        xs = []
        for feat in feats.split(''):
            if feat.strip() != "":
                xs.append(int(feat))
        XData.append(xs)
    print >> logs, 'read %d records' % len(XData)
    print >> logs, 'contains %d labels' % len(set(YData))
    return np.array(XData), np.array(YData)


