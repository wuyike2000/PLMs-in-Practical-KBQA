#!/usr/local/bin/python
# -*- coding: gbk -*-
from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import sys
import json
from datetime import datetime
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
from argparse import ArgumentParser
from transformers import BertTokenizer,RobertaTokenizer,AlbertTokenizer
import time

logger = logging.getLogger(__name__)

# personal package
from field import *
from bert_function import *
from model_my import *
from args import get_args
from data import Data

alpha = 0.1  

def occupied(seqa, seqb_list):
    # seqa:question
    # seqb:path
    # return value:[-1,1]
    scores = []
    for seqb in seqb_list:
        s = jaccard(seqa, seqb)
        scores.append(s)
    return scores


def jaccard(seqa, seqb):
    seqa = set(list(seqa.upper()))
    seqb = set(list(seqb.upper()))
    aa = seqa.intersection(seqb)
    bb = seqa.union(seqb)
    # return (len(aa)-1)/len(bb)
    return len(aa) / len(bb)


def hint(seqa, seqb):
    seqa = set(list(seqa.upper()))
    seqb = set(list(seqb.upper()))
    aa = seqa.intersection(seqb)
    return len(aa)


def data_batchlize(batch_size, data_tuple):
    '''
    give a tuple, return batches of data
    '''
    (subwords, mask) = data_tuple

    batches_subwords, batches_mask = [], []

    indexs = [i for i in range(len(subwords))]
    start = 0
    start_indexs = []
    while start <= len(indexs) - 1:
        start_indexs.append(start)
        start += batch_size

    start = 0
    for start in start_indexs:
        cur_indexs = indexs[start:start + batch_size]
        cur_subwords = [subwords[i] for i in cur_indexs]
        cur_mask = [mask[i] for i in cur_indexs]

        maxlen_i, maxlen_j = 0, 0
        for i, j in zip(cur_subwords, cur_mask):
            maxlen_i, maxlen_j = max(maxlen_i, len(i)), max(maxlen_j, len(j))
        batch_a, batch_b = [], []
        for a, b in zip(cur_subwords, cur_mask):
            batch_a.append([i for i in a] + [0] * (maxlen_i - len(a)))
            batch_b.append([i for i in b] + [0] * (maxlen_j - len(b)))

        batches_subwords.append(torch.LongTensor(batch_a))
        batches_mask.append(torch.LongTensor(np.array(batch_b)))

    return [item for item in zip(batches_subwords, batches_mask)]


def del_des(string):
    stack = []
    # if '_（' not in string and '）' not in string and '_(' not in string and ')' not in string:
    if '_' not in string:
        return string
    mystring = string[1:-1]
    if mystring[-1] != '）' and mystring[-1] != ')':
        return string
    for i in range(len(mystring) - 1, -1, -1):
        char = mystring[i]
        if char == '）':
            stack.append('）')
        elif char == ')':
            stack.append(')')
        elif char == '（':
            if stack[-1] == '）':
                stack = stack[:-1]
                if not stack:
                    break
        elif char == '(':
            if stack[-1] == ')':
                stack = stack[:-1]
                if not stack:
                    break
    if mystring[i - 1] == '_':
        i -= 1
    else:
        return string
    return '<' + mystring[:i] + '>'


def predict(args, model, field):
    Dataset = Data(args)

    fn_in = args.input_file
    if not args.output_file:
        f=open(args.output_file,'w',encoding='utf-8')
    else:
        f=open('./final_result/test.txt','w',encoding='utf-8')
    with open(fn_in, 'r', encoding='utf-8') as f1:
        raw_data = json.load(f1)

    topk = args.topk
    beta = args.weight

    pred_num=0
    for q,q1,entities,paths,scores in zip(raw_data['questions'],raw_data['origin_questions'],raw_data['entities'],raw_data['paths'],raw_data['scores']):
        
        scores1=[]
        for i in scores:
            scores1.append(float(i))
        scores1=np.array(scores1)
        one_question = Dataset.numericalize(field, [q]) 

        one_question = [t[0] for t in one_question]  

        one_question = (t for t in one_question)

        paths_input = [''.join([del_des(item) for item in path]) for path in paths]
        one_cands = Dataset.numericalize(field, paths_input)
        batches_cands = data_batchlize(args.test_batch_size, one_cands)
        
        char_scores = occupied(q, [''.join([del_des(i) for i in p]) for p in paths])
        char_scores = torch.Tensor(char_scores)
        
        model_scores = model.cal_score(one_question, batches_cands)
        all_scores = alpha * char_scores + (1 - alpha) * model_scores
        all_scores = beta * all_scores + (1 - beta) * scores1
        
        res=[]
        for i in range(0,len(all_scores)):
            res.append([all_scores[i],entities[i],paths[i]])
        res.sort(reverse=True)
        f.write(q1)
        if len(res)<topk:
            num=len(res)
        else:
            num=topk
        for k in range(0,num):
            f.write(' %%%% ')
            f.write(res[k][1])
            f.write('\t')
            f.write(res[k][2])
            f.write('\t')
            f.write(str(res[k][0]))
            f.write('\t')
        f.write('\n')
        pred_num+=1

    f.close()

def predict_one(args, model, tokenizer, text, rel_list):
    res = []
    output_data = {}

    bert_field = BertCharField('BERT', tokenizer=tokenizer)
    
    bos = '[CLS]'
    sequences = [bos] + tokenizer.tokenize(text)
    sequences = tokenizer.convert_tokens_to_ids(sequences)
    sequences = torch.tensor(sequences)
    mask = torch.ones(len(sequences))
    one_question = (sequences, mask)

    one_question = (t for t in one_question)
    
    paths_input = [''.join([del_des(item) for item in path]) for path in rel_list]

    rels = [[bos] + tokenizer.tokenize(t) for t in paths_input]
    rels = [tokenizer.convert_tokens_to_ids(t) for t in rels]
    rels = [torch.tensor(t) for t in rels]
    mask_rel = [torch.ones(len(t)) for t in rels]
    one_cands = (rels, mask_rel)

    batches_cands = data_batchlize(args.test_batch_size, one_cands)
  
    model_scores = model.cal_score(one_question, batches_cands)
    #all_scores = alpha * char_scores + (1 - alpha) * model_scores
    all_scores=model_scores
    if len(all_scores) > 0:
        res=all_scores
    else:
        res = []
        print(text, 'no path')
        print(text, 'no path')

    return res

if __name__ == "__main__":
    SCALE='large'
    MODEL='roberta-base'
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
    rescore=dict()
    pre_time=0
    #que_time=0
    beta=0.5
    #topk=5
    args = get_args(mode='predict')
    args.input_file='data/'+SCALE+'/'+MODEL+'/test.json'
    args.model_path=SCALE+'/'+MODEL+'/pytorch_model.bin'
    args.model=MODEL
    
    # tokenizer
    if 'uncased' in MODEL:
        tokenizer=BertTokenizer.from_pretrained("pretrain/"+MODEL)
    if 'roberta' in MODEL:
        tokenizer=RobertaTokenizer.from_pretrained("pretrain/"+MODEL)
    if 'albert' in MODEL:
        tokenizer=AlbertTokenizer.from_pretrained("pretrain/"+MODEL)
    bert_field = BertCharField('BERT', tokenizer=tokenizer)
    print("loaded tokenizer!")

    model_state_dict = torch.load(args.model_path)
    model = Bert_Comparing(args)
    print(model)
    model.load_state_dict(model_state_dict)
    
    if args.no_cuda == False:
        model.cuda()
    print('loaded model!')
    
    model.eval()
    
    pred_num=0
    data_re=[]
    data=json.load(open(args.input_file,'r',encoding='utf-8'))
    for ques,ques1,paths,scores,rels in zip(data['questions'],data['origin_questions'],data['paths'],data['scores'],data['relations']):
        start=time.time()
        res=predict_one(args,model,tokenizer,ques,rels)
        #end=time.time()
        #pre_time=pre_time+end-start
        #start=time.time()
        re_dict=dict(zip(rels,res))
        re_dict1=dict()
        for i in re_dict.items():
            re_dict1[i[0]]=float(i[1])
        rescore[ques1]=re_dict1
        en_re=dict()
        
        for index,i in enumerate(paths.items()):
            re_score=[]
            for j in i[1]:
               re_score.append([j,float(re_dict[j])*beta+float(scores[index])*(1-beta)])
            en_re[i[0]]=re_score
        data_re.append(en_re)
        end=time.time()
        pre_time=pre_time+end-start
        #que_time=que_time+end-start
        pred_num+=1
    data['re_scores']=data_re
    with open(SCALE+'/'+MODEL+'/result.json','w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False)
    with open(SCALE+'/'+MODEL+'/relation.json','w',encoding='utf-8') as f:
        json.dump(rescore, f,ensure_ascii=False)
    