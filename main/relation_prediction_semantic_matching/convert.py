#!/usr/local/bin/python
# -*- coding: gbk -*-
import json
import pickle
import time

MODEL='albert-base-v2'
SCALE='small'
topk=50
data=json.load(open(SCALE+'/'+MODEL+'/result.json','r',encoding='utf-8'))
f=open(SCALE+'/'+MODEL+'/result.txt','w',encoding='utf-8')
pred_num=0
sorttime=0

if SCALE=='small':
    de_dict=pickle.load(open("../indexes/degrees_2M.pkl",'rb'))
if SCALE=='large':
    de_dict=pickle.load(open("../indexes/degrees.pkl",'rb'))

for ques,ques1,paths,scores,re_scores in zip(data['questions'],data['origin_questions'],data['paths'],data['scores'],data['re_scores']):
    start=time.time()
    ques_result=[]
    for index,i in enumerate(paths.items()):
       res=re_scores[i[0]]
       for j in res:
           if i[0] in de_dict.keys():
               de_score=de_dict[i[0]][0]
           else:
               de_score=0
           ques_result.append([j[1],de_score,i[0],j[0]])
    ques_result.sort(key=lambda t: (t[0],t[1]), reverse=True)
    end=time.time()
    sorttime=sorttime+end-start
    f.write(ques1)
    if len(ques_result)<topk:
        num=len(ques_result)
    else:
        num=topk
    for k in range(0,num):
        f.write(' %%%% ')
        f.write(ques_result[k][2])
        f.write('\t')
        f.write(ques_result[k][3])
        f.write('\t')
        f.write(str(ques_result[k][0]))
        f.write('\t')
        f.write(str(ques_result[k][1]))
    f.write('\n')
    pred_num+=1

f.close()