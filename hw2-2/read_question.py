#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import numpy as np

PATH_Q = "./question.txt"
PATH_A = "./answer.txt"
PATH_model = "./dictionary"
with open(PATH_model, 'rb') as file:
    dictionary = pickle.load(file)
      
        
def readfile():
    question = open(PATH_Q, 'r')
    answer = open(PATH_A, 'r')
    q = question.readlines()
    a = answer.readlines()
    return q, a

    
# def sent2vec(sent, sent_len):
#     sent_seg = sent[:-1].split(' ')
#     new_sent = np.zeros((sent_len, 250))
#     for i in range(sent_len):
#         if i <= len(sent_seg) - 1:
#             new_sent[i] = dictionary.get(sent_seg[i])
            
    
class DataLoader():
    def __init__(self, sent_len):
        q, a = readfile()
        self.question = q
        self.answer = a
        self.sent_len = sent_len
        self.dictionary = dictionary
        
    def load_on_batch(self, start, end):
        q = self.question[start:end]
        question = []
        mask = []
        for i in range(start, end):
            v, m = self.turn_sent_to_vec(q[i])
            question.append(v)
            mask.append(m)
        question = np.array(question)
        mask = np.array(mask)
#         question = torch.Tensor(question)
#         mask = torch.Tensor(mask)
        return question, mask
    
    def turn_sent_to_vec(self, sentence):
        vec = []
        mask = []
        sentence = sentence[:-1].split(' ')
        for i in range(self.sent_len):
            if i < len(sentence):
                mask.append(1)
                w = sentence[i]
                id = self.dictionary.get(w)
                if(id is not None):
                    vec.append(id)
                else:
                    vec.append(np.ones((250))) # unknown = 1
            elif i == len(sentence):
                mask.append(1)
                vec.append(np.zeros((250))) # end = 0
            else:
                mask.append(0)
                vec.append(np.ones((250)) * 2) # blank = 2
        vec = np.array(vec)
        mask = np.array(mask)
        return vec, mask
    
    
# for testing purpose
d = DataLoader(10)
q, a = d.load_on_batch(0, 1)
print(q.shape)