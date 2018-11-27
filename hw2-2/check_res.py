#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:43:43 2018

@author: alpha
"""


x = open('test_input_without_space.txt')

x=x.readlines()

y=open('output_trial.txt')

y=y.readlines()

xy = zip(x, y)

for pair in xy:
    print(pair)
    input()