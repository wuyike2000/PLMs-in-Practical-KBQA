#!/usr/local/bin/python
# -*- coding: gbk -*-
import pandas as pd

MODEL='albert-base-v2'
SCALE='large'
truth=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
question=truth['question']
entity_mid=truth['entity_mid']
relation=[]
for i in truth['relation']:
    i=i.replace('.',' ')
    i=i.replace('_',' ')
    i=i[3:len(i)]
    relation.append(i) 

index=0
top1=0
top2=0
top3=0
top4=0
top5=0
num=0
#f1=open('entity_miss.txt','w',encoding='utf-8')
with open(SCALE+'/'+MODEL+'/result.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        line=line.split(' %%%% ')
        ques=line[0].strip()
        while ques!=question[index]:
            index+=1
        if ques==question[index] and len(line)>1:
            for i in range(1,len(line)):
                if SCALE=='small':
                    entity=entity_mid[index]
                if SCALE=='large':
                    entity=entity_mid[index][3:]
                if (line[i].split('\t')[0].strip()==entity) and (line[i].split('\t')[1].strip()==relation[index]):
                    if i==1:
                        top1+=1
                        top2+=1
                        top3+=1
                        top4+=1
                        top5+=1
                        break
                    if i==2:
                        top2+=1
                        top3+=1
                        top4+=1
                        top5+=1
                        break
                    if i==3:
                        top3+=1
                        top4+=1
                        top5+=1
                        break
                    if i==4:
                        top4+=1
                        top5+=1
                        break
                    if i==5:
                        top5+=1
                        break
                    '''
                    if i>5:
                        break
                    
                    if  line[1].split('\t')[0].strip()!=entity_mid[index]:
                        f1.write(ques)
                        f1.write('\n')
                    '''
        line=f.readline()
        index+=1
