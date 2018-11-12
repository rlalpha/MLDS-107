#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:14:21 2018

@author: alphay
"""

import numpy as np
import itertools
import re


def data_generator(x_train_filename='./data/sel_conversation/question.txt', 
                   y_train_filename='./data/sel_conversation/answer.txt', 
                   thershold_of_occurences=2, 
                   replacement_of_special_character = ''):

    # Load data
    questions_file = open(x_train_filename, 'r')
    answers_file = open(y_train_filename, 'r')
    questions = [line.split(' ') for line in questions_file]
    answers = [line.split(' ') for line in answers_file]


    print('start encoding')
    # init dictonary
    dic = {}
    word_to_idx = {
        'BOS': 0,
        'EOS': 1,
        'UWK': 2,
        'PAD': 3
    }

    return questions, answers


def generate_batch(X, y_inputs, y_targets, caption_id_to_feature_id, word_idx, sequence_length, batch_size):

    idx = np.random.choice(len(X), batch_size)

    x_idx = [caption_id_to_feature_id[id] for id in idx]
    
    X_batch = X[x_idx]
    y_inputs_batch = y_inputs[idx]
    y_targets_batch = y_targets[idx]
    sequence_length_batch = sequence_length[idx]
    return X_batch, y_inputs_batch, y_targets_batch, sequence_length_batch


if __name__ == '__main__':
    questions, answers = data_generator()
#   X, y_inputs, y_targets, caption_id_to_feature_id, word_idx, idx_word, _, _, sequence_length, _ = data_generator(
#        './data/training_data', './data/training_label.json', 2)

#    generate_batch(X, y_inputs, y_targets, caption_id_to_feature_id, word_idx, sequence_length, 128)
