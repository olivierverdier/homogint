#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

def time_step(dt):
    def scale(vf):
        def scaled_vf(x):
            return dt*vf(x)
        return scaled_vf
    return scale
