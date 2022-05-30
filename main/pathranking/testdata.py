import pandas as pd
import json
import time
import os

def convert(ques,label):
    if 'I' not in label or 'O' not in label:
        return ques
    result=''
    ques=ques.split(' ')
    label=label.split(' ')
    index=0
    while index!=len(ques):
        if label[index]=='O':
            result+=ques[index]
            index+=1
        else:
            result+='<e>'
            for i in range(index,len(ques)):
                if label[i]=='O':
                    index=i
                    break
            if label[index]=='I':
                return result
        result+=' '
    return result[:-1]

MODEL='roberta-base'
SCALE='large'

os.makedirs(SCALE+'/'+MODEL, exist_ok=True)
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
name_list=[]
with open('../entity_detection/'+MODEL+'_query/query.test','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.split(' %%%% ')
        name_list.append(line[1].strip())
        line=f.readline()

tagall=[]
with open('label/'+MODEL+'_label.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        tagall.append(line)
        line=f.readline()

f=open('test_ques.txt','w',encoding='utf-8')
total=0
total_dict=dict()
for i in range(0,len(test)):
    temp=dict()
    tempt=time.time()
    ques=convert(test["question"][i],tagall[i])
    total=total+time.time()-tempt
    f.write(ques)
    f.write('\n')
    temp['question']=ques
    temp['entity']=test['entity_mid'][i]
    temp['origin_question']=test['question'][i]
    relation=test['relation'][i]
    relation=relation.replace('.',' ')
    relation=relation.replace('_',' ')
    relation=relation[3:len(relation)]
    temp['relation']=relation
    total_dict[test['lineid'][i]]=temp
f.close()

with open(SCALE+'/'+MODEL+'/test_dict.json', 'w') as f:
    json.dump(total_dict, f,ensure_ascii=False)