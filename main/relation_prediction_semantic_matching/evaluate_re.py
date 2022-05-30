#!/usr/local/bin/python
# -*- coding: gbk -*-
import pandas as pd
import json

MODEL='bert-base-uncased'
SCALE='small'
truth=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
question=truth['question']
entity_mid=truth['entity_mid']
relation=[]
for i in truth['relation']:
    i=i.replace('.',' ')
    i=i.replace('_',' ')
    i=i[3:len(i)]
    relation.append(i) 
    
quessee=[]
quesunsee=[]
tra_re=set()
train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", 'question1',"tags"])
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", 'question1',"tags"])
for i in train['relation']:
    tra_re.add(i)
tra_re=list(tra_re)
for index,i in enumerate(test['relation']):
    if i in tra_re:
        quessee.append(test['question'][index])
    else:
        quesunsee.append(test['question'][index])

index=0
top1=0
top2=0
top3=0
top4=0
top5=0
seeac=0
unseeac=0
seere=0
unseere=0
num=0

data=json.load(open(SCALE+'/'+MODEL+'/relation.json','r',encoding='utf-8'))
for i in data.items():
    while i[0]!=question[index]:
        index+=1
    rescore=[]
    for j in i[1].items():
        rescore.append([j[0],j[1]])
    rescore.sort(key=lambda t: (t[1]), reverse=True)
    if i[0] in quessee:
        if len(rescore)>0 and rescore[0][0]==relation[index]:
            top1+=1
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            seeac+=1
            seere+=1
            continue
        if len(rescore)>1 and rescore[1][0]==relation[index]:
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            seere+=1
            continue
        if len(rescore)>2 and rescore[2][0]==relation[index]:
            top3+=1
            top4+=1
            top5+=1
            seere+=1
            continue
        if len(rescore)>3 and rescore[3][0]==relation[index]:
            top4+=1
            top5+=1
            seere+=1
            continue
        if len(rescore)>4 and rescore[4][0]==relation[index]:
            top5+=1
            seere+=1
            continue
    else:
        if len(rescore)>0 and rescore[0][0]==relation[index]:
            top1+=1
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            unseeac+=1
            unseere+=1
            continue
        if len(rescore)>1 and rescore[1][0]==relation[index]:
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            unseere+=1
            continue
        if len(rescore)>2 and rescore[2][0]==relation[index]:
            top3+=1
            top4+=1
            top5+=1
            unseere+=1
            continue
        if len(rescore)>3 and rescore[3][0]==relation[index]:
            top4+=1
            top5+=1
            unseere+=1
            continue
        if len(rescore)>4 and rescore[4][0]==relation[index]:
            top5+=1
            unseere+=1
            continue