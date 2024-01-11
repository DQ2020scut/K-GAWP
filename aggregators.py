import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import aggregators
import utils
import json
from importlib import import_module
from models.gnn_preprocess import *
#import transformers
import os
import sys
sys.path.append("..")

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, data, dim=1): # data: (bs, max_len, embed_dim)
        res = torch.mean(data, dim) #.unsqueeze(1)
        return res

class SUMAggregator(nn.Module):
    def __init__(self):
        super(SUMAggregator, self).__init__()
    def forward(self, data, dim=1): # data: (bs, max_len, embed_dim)
        res = torch.sum(data, dim) # .unsqueeze(1)
        return res
    
class MaxAggregator(nn.Module):
    def __init__(self):
        super(MaxAggregator, self).__init__()
    def forward(self, data, dim=1):
        res = torch.max(data, dim)[0]
        return res

class MLPAggregator(nn.Module):
    def __init__(self, **kwargs):
        super(MLPAggregator, self).__init__()
        self.input_dim = kwargs['input_dim']
        self.max_len = kwargs['max_len']
        self.layer_dims = kwargs['layer_dims']
        self.out_size = kwargs['out_size']
        self.task = kwargs['task']
        
        if self.task:  
            self.fc_1 = nn.Linear(self.input_dim, self.layer_dims[0])
        else:
            self.fc_1 = nn.Linear(self.input_dim*self.max_len, self.layer_dims[0])
        self.fc_2 = nn.Linear(self.layer_dims[0], self.layer_dims[1])
        self.fc_3 = nn.Linear(self.layer_dims[1], self.out_size)
        
    def forward(self, data, batch_masks=None):
        if isinstance(data, tuple):
            essence_data = data[0]
            gnn_spatial_data = data[1]
            data = torch.cat((essence_data, gnn_spatial_data), dim=-1)
        else:
            bs = data.shape[0]
            
        data = data.view(bs, -1)
        res = F.relu(self.fc_1(data))
        res = F.relu(self.fc_2(res))
        res = self.fc_3(res)
        return res, self
   
class LSTMAggregator(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMAggregator, self).__init__()
        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']
        self.out_size = kwargs['out_size']
        self.max_len = kwargs['max_len']
        self.num_layer = kwargs['num_layer']
        self.drop = kwargs['drop']
        self.bn = kwargs['bn']
        self.cuda = kwargs['cuda']
        self.bi = kwargs['direction']
        
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layer, bidirectional=self.bi, batch_first=True)
        self.batch_norm = torch.nn.BatchNorm1d(self.max_len)
        self.dropout = torch.nn.Dropout(self.drop)
        d_num = 1
        if self.bi:
            d_num = 2
        self.fc_in = nn.Linear(self.input_size, int(self.input_size/2))
        self.fc_out = nn.Linear(d_num * self.hidden_size * self.max_len, self.out_size)
        self.fc_out.requires_grad = True
        self.fc_in.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True
        
    def forward(self, data, batch_masks=None, cluster=False): #tasks_embeds, task_sp_embeds=None
        if isinstance(data, tuple) and cluster==False:
            essen_data = data[0]
            add_data = data[1]
            essen_data = self.fc_in(essen_data) # ensure dimension consistancy bs * max_len * input_size/2
            bs = essen_data.shape[0]
            data = torch.cat((essen_data, add_data), dim=-1)
            #if not task_sp_embeds==None:
            #    task_res = torch.cat((tasks_embeds, task_sp_embeds), dim=-1)   
            #dummy_data = data[-1].repeat((1,1,2))
            #data_ = torch.cat((dummy_data, data), dim=1)
        else:
            bs = data.shape[0]  
        data = data.view(self.max_len, bs, -1) # out: the output embedding at the last LSTM layer, each corresponds to a input word           
        out, (h_n, h_c) = self.lstm(data) # hidden_cell: the h and c for all LSTM layers
        
        out = self.dropout(out)
        if self.bn:
            out = out.reshape((data.shape[1], self.max_len, -1))
            out = self.batch_norm(out)  
        res = torch.tanh(self.fc_out(out.reshape((bs, -1)))) # batch_size, hidden_size * bi_dir * max_len
        return res, None
        
class CNNAggregator(nn.Module):
    def __init__(self, **kwargs):
        super(CNNAggregator, self).__init__()
        self.max_len = kwargs['max_len']
        self.stride = kwargs['stride']
        self.filters = kwargs['filters']
        self.in_channel = kwargs['in_channel']
        self.drop = kwargs['drop']
        self.bs = kwargs['bs']
        self.bn = kwargs['bn']
        self.input_dim = kwargs['input_dim']
        self.out_size = kwargs['out_size']
        self.cuda = kwargs['cuda']
        self.pool_ratio = 3
        self.conv_ops = []
        conv_dim = 0
        for f in self.filters:
            self.temp_conv = nn.Conv2d(f[0], f[1], (f[2], self.input_dim)) # # self.max_len f is for CNN operation: [in-channel, out-channel, filter-h, filter-w]
            self.hidden_w = int(np.floor((self.max_len - f[2])/self.stride) + 1)
            pooln = int(np.ceil(self.hidden_w/self.pool_ratio))

            self.temp_pool = nn.MaxPool2d((pooln, 1), stride=self.stride) #self.hidden_w
            conv_dim += f[1] * int(np.floor((self.hidden_w - pooln)/self.stride) + 1)            
            if self.cuda==True:
                self.temp_conv.cuda()
                self.temp_pool.cuda()
            self.conv_ops.append((self.temp_conv, self.temp_pool))
        
        self.conv_fc = nn.Linear(conv_dim, self.out_size)
        self.dropout = nn.Dropout(self.drop)
        
    def forward(self, data, args, batch_masks=None):
        if isinstance(data, tuple):
            essen_data = data[0]
            gnn_spatial_data = data[1]
            data = torch.cat((essen_data, gnn_spatial_data), dim=-1)
        else:
            bs = data.shape[0]
        data = data.view(bs, self.in_channel, self.max_len, -1) # input: [batch, in-channel, sentence_length, embed_dim]
        res_tensor = None
        for i, op in enumerate(self.conv_ops):
            conv_res = op[0](data) # op[0], CNN operation #conv_res
            conv_res = op[1](conv_res).view(bs, -1) # op[1], pool
            if i==0:
                res_tensor = conv_res
            else:
                res_tensor = torch.cat((res_tensor, conv_res), dim=-1)
        
        fc_res = self.conv_fc(res_tensor) # # op[2], FC, result shape: batch, in-channel, 50-dim
        if self.drop > 0:
            fc_res = self.dropout(fc_res) 
        return fc_res, self
    
class BertAggregator(nn.Module):
    def __init__(self, **kwargs):
        super(BertAggregator, self).__init__()
        args = kwargs['args']
        cuda = args.cuda
        self.bert_dic = {
            'attention_probs_dropout_prob': 0.1, 
            'directionality': "bidi", 
            'hidden_act': "relu", 
            'hidden_dropout_prob': 0.1, 
            'initializer_range': 0.02,
            'hidden_size': args.bert_hidden_dim, 
            'intermediate_size': args.bert_inter_dim, 
            'max_position_embeddings': args.bert_pos_dim, 
            'num_attention_heads': args.bert_heads,
            'num_hidden_layers': args.bert_layer_num, 
            'type_vocab_size': 2, 
            'max_len': args.max_len,
            'embedding_dim': args.embed_dim,
            'vocab_size': len(kwargs['corpus']),
            'out_size': args.out_size,
            'use_norm': args.batch_norm,
            'PAD_id': 0,
            # for DEBERTa
            'relative_attention':True,
            'att_type': ['p2c', 'c2p'], # ['p2c', 'c2p'] ['p2c'], ['c2p'] []
        }
        assert self.bert_dic['hidden_size'] % self.bert_dic['num_attention_heads'] == 0
        
        if args.use_trans==True: # use transformer toolkit
            if args.bert_name=='albert':
                x = import_module('models.albert.trainer.' + 'pretrain') # x is a python file .py
                self.model = x.PretrainTrainer(self.bert_dic).net
                self.config = self.bert_dic
            if args.bert_name=='deberta':
                x = import_module('models.transformers.models.deberta.modeling_deberta')
                self.config = import_module('models.transformers.models.deberta.configuration_deberta').DebertaConfig(bert_dic=self.bert_dic)
                self.model = x.DebertaForSequenceClassification(self.config)
        
        else: #use conventional separated BERT model
            with open('./models/bert_config.json', 'w') as f:
                json.dump(self.bert_dic, f)
            x = import_module('models.' + 'bert')
            self.config = x.Config(args) # this config is not the BERT config, it is the exogenous parameters, e.g. input/output size
            self.model = x.Model(self.config)
    
    def forward(self):
        return self.model, self.config
        
class GATAggregator(nn.Module):
    def __init__(self, **kwargs):
        super(GATAggregator, self).__init__()
        self.args = kwargs['args']
        self.initial_entity_embed = kwargs['ent_embeds']
        self.corpus = kwargs['corpus']
        self.all_products = kwargs['all_products']
        self.word2id = kwargs['word2id']
        self.id2word = kwargs['id2word']
        self.cluster_or_granu = kwargs['cluster_or_granu']
        self.embed_dim = self.args.embed_dim
        self.relation2id = self.args.rel_dic
        self.ent_out_dim = self.args.out_size # assume the two are the same
        self.rel_out_dim = self.args.out_size
        self.drop_gat = self.args.dropout
        self.nheads_gat = self.args.gnn_heads
        self.relu_alpha = self.args.gat_alpha
        self.cuda = self.args.cuda
        self.cnn_out_ch = self.args.gnn_conv_ch
        
        # generate relation embedding randomly
        self.id2relation = {v:k for k,v in self.relation2id.items()}
        rel_embed_dic = {}
        self.initial_relation_embed = torch.Tensor(len(list(self.relation2id.keys())), self.embed_dim)
        for rel in list(self.relation2id.keys()):
            rel_embed_dic.update({rel : np.random.uniform(low=-1.0, high=1.0, size=self.embed_dim)}) 
        for i, rel in enumerate(list(self.relation2id.keys())):
            rel_embed = utils.convert_data_to_tensor([rel], rel_embed_dic, pri=False)[0]
            self.initial_relation_embed[i] = rel_embed
        
        if self.cluster_or_granu==False: # if the model is trained
            self.all_term2id, self.id2all_term = generate_gnn_terms(self.all_products, self.word2id, self.relation2id)
        else: # during clustering or off-line testing, load all_term2id directly
            self.all_term2id = kwargs['all_term2id']
        gnn_pkg = import_module('models.' + 'gnn')
        self.model_gat = gnn_pkg.SpKBGATModified(self.word2id, self.id2word, self.relation2id, self.all_term2id, self.initial_entity_embed, self.initial_relation_embed, 
                        self.ent_out_dim, self.rel_out_dim, self.nheads_gat, cnn_out_ch=self.cnn_out_ch, drop=self.drop_gat, alpha=self.relu_alpha, cuda=self.cuda)
        
    def forward(self, batch_sp_triples, args, use_avg=False):
        init_adj_mt = generate_raw_adj(batch_sp_triples, self.all_term2id, args)
        self.batch_indices_nhop, self.batch_adj_matrix = constructNeighbourhood(batch_sp_triples, init_adj_mt, self.all_term2id, args.nhop)
        bacth_gnn_data = list(zip(self.batch_indices_nhop, self.batch_adj_matrix))
        
        gnn_all_entity = []
        gnn_all_rels = []
        for i, current_batch_gnn_data in enumerate(bacth_gnn_data):
            adj_mat = current_batch_gnn_data[1]
            indices_nhop = current_batch_gnn_data[0]
            out_entity, out_relation = self.model_gat(batch_sp_triples[i], adj_mat, indices_nhop, use_avg) # out_entity, term_num x embed_dim
            gnn_all_entity.append(out_entity)
            gnn_all_rels.append(out_relation)            
        
        gnn_all_entity = gnn_all_entity[-1].clone().detach().requires_grad_(True)
        gnn_all_rels = gnn_all_rels[-1].clone().detach().requires_grad_(True)
        return gnn_all_entity, gnn_all_rels 
        
        