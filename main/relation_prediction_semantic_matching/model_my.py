import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import unicodedata
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import torch.nn.functional as F
import random
from torch.nn import CosineSimilarity
from collections import Counter
from field import *
from transformers import BertModel,RobertaModel,AlbertModel

class BertCharEmbedding(nn.Module):
    def __init__(self, model, requires_grad=True):
        super(BertCharEmbedding, self).__init__()
        if 'uncased' in model:
            self.bert=BertModel.from_pretrained("pretrain/"+model)
        if 'roberta' in model:
            self.bert=RobertaModel.from_pretrained("pretrain/"+model)
        if 'albert' in model:
            self.bert=AlbertModel.from_pretrained("pretrain/"+model)
        self.requires_grad = requires_grad
    
    def forward(self, subwords, bert_mask):
        bert= self.bert(subwords, attention_mask=bert_mask).last_hidden_state
        return bert

class Bert_Comparing(nn.Module):
    def __init__(self, data):
        super(Bert_Comparing, self).__init__()

        self.question_bert_embedding = BertCharEmbedding(data.model, data.requires_grad)
        self.path_bert_embedding = BertCharEmbedding(data.model, data.requires_grad)
        self.args = data
        self.similarity = CosineSimilarity(dim=1)
    
    def question_encoder(self, input_idxs, bert_mask):
        bert_outs = self.question_bert_embedding(input_idxs, bert_mask)
        return bert_outs[:, 0]
    
    def path_encoder(self, input_idxs, bert_mask):
        bert_outs = self.path_bert_embedding(input_idxs, bert_mask)
        return bert_outs[:, 0]
        
    def forward(self, questions, pos, negs):
        '''
        questions: batch_size, max_seq_len

        pos_input_idxs: batch_size, max_seq_len
        pos_bert_lens: batch_size, max_seq_len
        pos_bert_mask: batch_size, max_seq_len

        neg_input_idxs: neg_size, batch_size, max_seq_len
        neg_bert_lens: neg_size, batch_size, max_seq_len
        neg_bert_mask: neg_size, batch_size, max_seq_len
        '''
        
        (q_input_idxs, q_bert_mask) = questions

        (pos_input_idxs, pos_bert_mask) = pos
        (neg_input_idxs, neg_bert_mask) = negs
        neg_size, batch_size, _ = neg_input_idxs.shape

        q_encoding = self.question_encoder(q_input_idxs, q_bert_mask) # (batch_size, hidden_dim)

        pos_encoding = self.path_encoder(pos_input_idxs, pos_bert_mask)

        neg_input_idxs = neg_input_idxs.reshape(neg_size*batch_size, -1) # (neg_size*batch_size, max_seq_len)
        neg_bert_mask = neg_bert_mask.reshape(neg_size*batch_size, -1) # (neg_size*batch_size, max_seq_len)

        neg_encoding = self.path_encoder(neg_input_idxs, neg_bert_mask) # (neg_size*batch_size, hidden_dim)
        # p_encoding = p_encoding.reshape(neg_size, batch_size, -1) # (neg_size, batch_size, hidden_dim)
        
        q_encoding_expand = q_encoding.unsqueeze(0).expand(neg_size, batch_size, q_encoding.shape[-1]).reshape(neg_size*batch_size, -1) # (neg_size*batch_size, hidden_dim)

        pos_score = self.similarity(q_encoding, pos_encoding)
        pos_score = pos_score.unsqueeze(1) # (batch_size, 1)
        neg_score = self.similarity(q_encoding_expand, neg_encoding)
        neg_score = neg_score.reshape(neg_size,-1).transpose(0,1) # (batch_size, neg_size)

        return (pos_score, neg_score)
    
    @torch.no_grad()
    def cal_score(self, question, cands, pos=None):
        '''
        one question, several candidate paths
        question: (max_seq_len), (max_seq_len), (max_seq_len)
        cands: (batch_size, max_seq_len), (batch_size, max_seq_len), (batch_size, max_seq_len)
        '''
        
        question = (t.unsqueeze(0) for t in question)

        if self.args.no_cuda == False:
            question = (t.cuda() for t in question)

        (q_input_idxs, q_bert_mask) = question
        
        
        q_encoding = self.question_encoder(q_input_idxs, q_bert_mask) # (batch_size=1, hidden_dim)
        
        if pos:
            pos = (t.unsqueeze(0) for t in pos)
            if self.args.no_cuda == False:
                pos = (t.cuda() for t in pos)
            
            (pos_input_idxs, pos_bert_mask) = pos
            pos_encoding = self.path_encoder(pos_input_idxs, pos_bert_mask) # (batch_size=1, hidden_dim)
            pos_score = self.similarity(q_encoding, pos_encoding) # (batch_size=1) 

        all_scores = []

        for (batch_input_idxs, batch_bert_mask) in cands:
            if self.args.no_cuda ==False:
                batch_input_idxs, batch_bert_mask = batch_input_idxs.cuda(), batch_bert_mask.cuda()
            path_encoding = self.path_encoder(batch_input_idxs, batch_bert_mask) #(batch_size, hidden_dim)
            q_encoding_expand = q_encoding.expand_as(path_encoding)
            scores = self.similarity(q_encoding_expand, path_encoding) # (batch_size)
            for score in scores:
                all_scores.append(score)
        all_scores = torch.Tensor(all_scores)

        if pos:
            return pos_score.cpu(), all_scores.cpu()
        else:
            return all_scores.cpu()
