import random
import json
import pandas as pd
import os
import math
import pickle
import time

from argparse import ArgumentParser
from collections import defaultdict
import time
import pandas as pd

SCALE='large'
time_ca=0
neg_num=10
train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
valid=pd.read_table('../mydata/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
def load_index(filename):
    print("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index

# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits):
    print("Entity Source : {}".format(filename))
    id2mids = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0]
        cand_mids = items[1:][:hits]
        for mid_entry in cand_mids:
            mid, mid_name, mid_type, score = mid_entry.split('\t')
            id2mids[lineid].append((mid, mid_name, mid_type, float(score)))
    return id2mids

def get_rels(filename, hits):
    print("Relation Source : {}".format(filename))
    id2rels = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0].strip()
        rel = www2fb(items[1].strip())
        label = items[2].strip()
        score = items[3].strip()
        if len(id2rels[lineid]) < hits:
            id2rels[lineid].append((rel, label, float(score)))
    return id2rels


def get_questions(filename):
    print("getting questions ...")
    id2questions = {}
    id2goldmids = {}
    fin =open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        lineid = items[0].strip()
        mid = items[1].strip()
        question = items[5].strip()
        rel = items[3].strip()
        id2questions[lineid] = (question, rel)
        id2goldmids[lineid] = mid
    return id2questions, id2goldmids

def get_mid2wiki(filename):
    print("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki

if SCALE=='small':
    index_reach = load_index('../indexes/reachability_2M.pkl')
if SCALE=='large':
    index_reach = load_index('../indexes/redict.pkl')

total_re=[]
if SCALE=='small':
    with open('./relation_small.txt','r',encoding='utf-8') as f:
        line=f.readline()
        while line!='':
            line=line.strip()
            total_re.append(line)
            line=f.readline()
if SCALE=='large':
    with open('./relation_large.txt','r',encoding='utf-8') as f:
        line=f.readline()
        while line!='':
            line=line.strip()
            total_re.append(line)
            line=f.readline()

question=[]
relation=[]
with open('./train_ques.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        question.append(line)
        line=f.readline()
        
with open('./train_re.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        relation.append(line)
        line=f.readline()
    
neg_sample=[]
train_mid=train['entity_mid']
s1=time.time()
for i in range(0,len(question)):
    re_list=[]
    if SCALE=='large':
        tempmid=train_mid[i][3:]
    else:
        tempmid=train_mid[i]
    for j in index_reach[tempmid]:
        j=j.replace('.',' ')
        j=j.replace('_',' ')
        if SCALE=='small':
            j=j[3:len(j)]
        if j!='type object name' and j!='common topic alias':
            re_list.append(j) 
    if relation[i] in re_list:
        re_list.remove(relation[i])
    temp=[]
    if len(re_list)<=neg_num:
        temp.extend(re_list)
        while len(temp)!=neg_num:
            neg = random.choice(total_re)
            if neg!=relation[i] and neg not in temp:
                temp.append(neg)
    else:
        while len(temp)!=neg_num:
            neg = random.choice(re_list)
            if neg!=relation[i] and neg not in temp:
                temp.append(neg)
    neg_sample.append(temp)
e1=time.time()
time_ca=time_ca+e1-s1

train_json=dict()
train_json["questions"]=question
train_json["golds"]=relation
train_json["negs"]=neg_sample
with open(SCALE+'/train.json', 'w') as f:
    json.dump(train_json, f,ensure_ascii=False)

question=[]
relation=[]
with open('./valid_ques.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        question.append(line)
        line=f.readline()
        
with open('./valid_re.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        relation.append(line)
        line=f.readline()
    
neg_sample=[]
valid_mid=valid['entity_mid']
s2=time.time()
for i in range(0,len(question)):
    re_list=[]
    if SCALE=='large':
        tempmid=valid_mid[i][3:]
    else:
        tempmid=valid_mid[i]
    for j in index_reach[tempmid]:
        j=j.replace('.',' ')
        j=j.replace('_',' ')
        if SCALE=='small':
            j=j[3:len(j)]
        if j!='type object name' and j!='common topic alias':
            re_list.append(j) 
    if relation[i] in re_list:
        re_list.remove(relation[i])
    temp=[]
    if len(re_list)<=neg_num:
        temp.extend(re_list)
        while len(temp)!=neg_num:
            neg = random.choice(total_re)
            if neg!=relation[i] and neg not in temp:
                temp.append(neg)
    else:
        while len(temp)!=neg_num:
            neg = random.choice(re_list)
            if neg!=relation[i] and neg not in temp:
                temp.append(neg)
    neg_sample.append(temp)
e2=time.time()
time_ca=time_ca+e2-s2

valid_json=dict()
valid_json["questions"]=question
valid_json["golds"]=relation
valid_json["negs"]=neg_sample
with open(SCALE+'/valid.json', 'w') as f:
    json.dump(valid_json, f,ensure_ascii=False)