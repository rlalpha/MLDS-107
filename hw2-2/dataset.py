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
                   thershold_of_occurences=10,
                   replacement_of_special_character=''):

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
    idx_to_word = None

    for sentence in questions + answers:
        for word in sentence:
            dic[word] = dic.get(word, 0) + 1

    next_word_id = 4
    for item in dic.items():
        if item[1] > thershold_of_occurences:
            word_to_idx[item[0]] = next_word_id
            next_word_id += 1

    idx_to_word = dict((reversed(item) for item in word_to_idx.items()))

    questions = [[word_to_idx.get(word, 2) for word in question]
                for question in questions]
    answers = [[word_to_idx.get(word, 2) for word in answer]
                for answer in answers]

    print('start convert to npy')
    questions = np.array(questions)
    answers = np.array(answers)

    max_length = max([len(answer) for answer in answers]) + 1
    sequence_length = np.array([len(answer) + 1 for answer in answers])

    y_inputs = np.array([[word_to_idx['BOS']] + y + [word_to_idx['PAD']]
                         * (max_length - len(y) - 1) for y in answers])
    y_targets = np.array([y + [word_to_idx['EOS']] + [word_to_idx['PAD']]
                          * (max_length - len(y) - 1) for y in answers])

    # print('y_inputs: ', y_inputs)
    # print('y_targets: ', y_targets)
    print('Done data generation!')

    return questions, y_inputs, y_targets, word_to_idx, idx_to_word, next_word_id, max_length, sequence_length


    # return questions, answers


def generate_batch(x, y_inputs, y_targets, word_idx, sequence_length, batch_size):

    idx = np.random.choice(len(x), batch_size)

    x_batch = x[idx]
    y_inputs_batch = y_inputs[idx]
    y_targets_batch = y_targets[idx]
    sequence_length_batch = sequence_length[idx]
    return x_batch, y_inputs_batch, y_targets_batch, sequence_length_batch


if __name__ == '__main__':
    questions, y_inputs, y_targets, word_to_idx, idx_to_word, next_word_id, max_length, sequence_length = data_generator()

    x_batch, y_inputs_batch, y_targets_batch, sequence_length_batch = generate_batch(questions, y_inputs, y_targets, word_to_idx, sequence_length, batch_size=128)
