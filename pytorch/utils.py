#!/usr/bin/python3
# coding=utf-8://github.com/pytorch/examples/tree/master/mnist

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




