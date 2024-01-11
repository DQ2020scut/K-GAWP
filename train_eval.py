import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import aggregators
import utils
import time
from aggregators import *
import random
import copy
import heapq
from heapq import nsmallest
import sys
import pandas as pd
from nltk.metrics.scores import precision
sys.path.append("..")
from models.gnn_preprocess import *
from cluster import *
from cm_build import Metrics
from drawing import *

def select_model(**kwargs):
    args = kwargs['args']
    model_name = args.model_name
    if args.use_gnn:
        max_len = args.max_len #+ 1
    else:
        max_len = args.max_len
    if model_name == 'mean':
        model = MeanAggregator()
    elif model_name == 'sum':
        model = aggregators.SUMAggregator()
    elif model_name == 'pool':
        model = MaxAggregator()
    elif model_name == 'lstm':
        model = LSTMAggregator(input_size=args.embed_dim, hidden_size=args.lstm_hs, max_len=max_len, num_layer=args.lstm_layer, 
        out_size=args.out_size, bs=args.batch_size, drop=args.dropout, cuda=args.cuda, bn=args.batch_norm, direction=args.direction)
    elif model_name == 'cnn':
        model = CNNAggregator(max_len=max_len, in_channel=args.cnn_inc, class_num=args.class_num, out_size=args.out_size, stride=args.cnn_st, 
        filters=args.cnn_filters, cuda=args.cuda, bs=args.batch_size, drop=args.dropout, bn=args.batch_norm, input_dim = args.embed_dim)
    elif model_name == 'mlp':
        model = MLPAggregator(input_dim=args.embed_dim, max_len=max_len, layer_dims=args.mlp_layer_dims, out_size=args.out_size, task=False)
    elif model_name == 'bert':
        bert_model = BertAggregator(args=args, corpus=kwargs['corpus'])
        model, _ = bert_model()
    if args.use_gnn:
        gnn_model = GATAggregator(args=args, corpus=kwargs['corpus'], all_products=kwargs['all_products'], word2id=kwargs['word2id'], id2word=kwargs['id2word'],
                ent_embeds=kwargs['ent_embeds'], cluster_or_granu=kwargs['cluster_or_granu'], all_term2id=kwargs['all_term2id'])
    else:
        gnn_model=None
    return model, gnn_model

def train_evaluate(args, model, optimizer, loader, embed_matrix, id2word, word2id, tasks_embeds, gnn=None, id2task=None, task2si=None, tid2tid=None, flag='train'):
    all_loss = []
    test_df = pd.DataFrame(columns=['hit@6', 'hit@3', 'hit@1']) #record test metrics
    train_df = pd.DataFrame(columns=['hit@6', 'hit@3', 'hit@1'])
    if args.use_trans:
        print_bert_name = args.bert_name
    else:
        print_bert_name = ''
    print('current mode:', flag, ' used model:', args.model_name, ' ', print_bert_name)
    
    if flag == 'evaluation' or args.full_mode=='simple':
        epoch_num = 1
    else:
        epoch_num = args.epoch_num
        
    all_preds = []
    all_labels = []
    all_scores = []
    all_pr_scores = []
    for epoch in range(epoch_num):
        print('current epoch: ', epoch)
        for batch_idx, item in enumerate(loader):            
            batch_products, batch_labels, batch_masks, sp_labels, abs_tlabels = item
            origin_len = len(batch_products) # due the different number of samples in the last batch, the length evaluation is critical
            origin_batch_labels = copy.deepcopy(batch_labels)
            
            if flag=='train': # and args.full_mode=='complex':
                batch_products, batch_labels, batch_masks, sp_labels, targets = generate_neg_data(batch_products, batch_labels, batch_masks, sp_labels, args)  
            if flag=='evaluation':
                batch_masks = batch_masks.reshape((origin_len, -1, batch_products.shape[1])).unsqueeze(1)
            
            batch_data = Variable(utils.generate_batch_data(batch_products, embed_matrix, id2word, word2id, args)) # convert batch data (indices) to numerical matrix         
            batch_labels = torch.stack([utils.process_task_ids(args, bl, word2id, id2task) for bl in batch_labels]).unsqueeze(1)
            batch_labels = Variable(utils.generate_batch_data(batch_labels, embed_matrix, id2word, word2id, args)).squeeze(1)
            
            sp_labels_ = []
            if args.use_sp_data==True:
                for sl in sp_labels:
                    try:
                        sl = [int(s) for s in sl.split('+')]
                        sp_labels_.append(torch.tensor(sl))
                    except:
                        sl = [-1]
                        sp_labels_.append(torch.tensor(sl))
                         
            if args.cuda:
                batch_products = batch_products.cuda()
                embed_matrix = embed_matrix.cuda()
                batch_data = batch_data.cuda()
                batch_labels = batch_labels.cuda()
                tasks_embeds = tasks_embeds.cuda()
                batch_masks = batch_masks.cuda()
                if args.use_sp_data==True:
                    sp_labels_ = [sl.cuda() for sl in sp_labels_]
                abs_tlabels = abs_tlabels.cuda()
                model = model.to(torch.device('cuda:0'))
                if flag=='train':
                    targets = targets.cuda()
            else:
                model = model.to('cpu')
     
            if not gnn==None: # sp_batch_inputs ===> the AS-IS length of sptial triples in each input 
                sp_batch_inputs = reconstruct_terms_form_ids(batch_products, gnn.all_term2id, word2id, id2word, args.rel_dic) # join the words into terms and label them using all_term2id
                if args.cuda:
                    gnn = gnn.to(torch.device('cuda:0'))
                if flag=='train':
                    gnn_all_entity, gnn_all_rels = gnn(sp_batch_inputs, args) # input sp_triples (e.g., x constrains y & x contains y..) x,y = terms
                else:
                    with torch.no_grad():
                        gnn_all_entity, gnn_all_rels = gnn(sp_batch_inputs, args)
                
                batch_gnn_add = []
                for sp_triples in sp_batch_inputs:
                    sp_triples = torch.tensor(sp_triples)
                    gnn_conv_input = torch.cat((gnn_all_entity[sp_triples[:,0]].unsqueeze(1), gnn_all_rels[sp_triples[:,1]].unsqueeze(1), 
                                                gnn_all_entity[sp_triples[:,2]].unsqueeze(1)), dim=1)
                    gnn_conv_out = gnn.model_gat.convKB(gnn_conv_input, args)
                    batch_gnn_add.append(gnn_conv_out)
                batch_gnn_add = torch.stack(batch_gnn_add)
                dummy_task = gnn_all_entity[0] # gnn_all_entity is the embedding matrix of terms
                dummy_task = dummy_task.repeat((batch_data.shape[0], 1)).unsqueeze(1)
                
            if flag=='train' and args.full_mode=='complex':
                if not gnn==None:
                    outputs, _ = model.forward((batch_data, batch_gnn_add, dummy_task), batch_masks=batch_masks)
                else:
                    outputs, _ = model.forward(batch_data, batch_masks=batch_masks)  
                loss, loss_func = get_batch_loss(outputs, batch_labels, targets=targets, loss_name=args.loss_func, origin_len=origin_len)
                all_loss.append(loss.data)
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                
                flops, params = utils.complexity_analyze(model, (batch_data, args, batch_masks))
                train_hit6, _, _ = evaluate_complex_agg(tasks_embeds, outputs, origin_batch_labels.data, origin_len, args.cuda, topk=6, sp_labels=sp_labels_,
                                                  abs_tlabels=abs_tlabels, task2si=task2si, tid2tid=tid2tid, id2task=id2task, flag=flag)
                train_hit3, _, _ = evaluate_complex_agg(tasks_embeds, outputs, origin_batch_labels.data, origin_len, args.cuda, topk=3, sp_labels=sp_labels_,
                                                  abs_tlabels=abs_tlabels, task2si=task2si, tid2tid=tid2tid, id2task=id2task, flag=flag)
                train_hit1, _, _ = evaluate_complex_agg(tasks_embeds, outputs, origin_batch_labels.data, origin_len, args.cuda, topk=1, sp_labels=sp_labels_, 
                                                  abs_tlabels=abs_tlabels, task2si=task2si, tid2tid=tid2tid, id2task=id2task, flag=flag)
                train_df = train_df.append({'hit@6':train_hit6, 'hit@3':train_hit3, 'hit@1':train_hit1}, ignore_index=True)
                print('\ttraining performance, hit@6:{}, hit@3:{}, hit@1:{}'.format(train_hit6, train_hit3, train_hit1))
            
            else: # evaluation stage
                time_s = time.time() # recording time
                
                if args.full_mode=='complex': # complex deep learning model evaluation
                    if not gnn==None:
                        with torch.no_grad():
                            outputs, _ = model.forward((batch_data, batch_gnn_add, dummy_task), batch_masks=batch_masks)
                    else:
                        with torch.no_grad():
                            outputs, _ = model.forward(batch_data, batch_masks=batch_masks)
                else: # simple model evaluation
                    outputs = model(batch_data)
                    
                test_hit6, _, _ = evaluate_complex_agg(tasks_embeds, outputs, origin_batch_labels.data, origin_len, args.cuda, topk=6, sp_labels=sp_labels_, 
                                                 abs_tlabels=abs_tlabels, task2si=task2si, tid2tid=tid2tid, id2task=id2task, flag=flag, full_mode=args.full_mode)
                test_hit3, _, _ = evaluate_complex_agg(tasks_embeds, outputs, origin_batch_labels.data, origin_len, args.cuda, topk=3, sp_labels=sp_labels_, 
                                                 abs_tlabels=abs_tlabels, task2si=task2si, tid2tid=tid2tid, id2task=id2task, flag=flag, full_mode=args.full_mode)
                test_hit1, preds, scores = evaluate_complex_agg(tasks_embeds, outputs, origin_batch_labels.data, origin_len, args.cuda, topk=1, sp_labels=sp_labels_, 
                                                 abs_tlabels=abs_tlabels, task2si=task2si, tid2tid=tid2tid, id2task=id2task, flag=flag, full_mode=args.full_mode)
  
                # complexity analysis --> time, FLOPs, and parameter size
                flops, params = utils.complexity_analyze(model, (batch_data, args, batch_masks))
                time_lag = time.time() - time_s #compute time
                
                # t-SNE analysis
                num_class = tasks_embeds.shape[0]
                #utils.tsne_analyze(batch_data, outputs, num_class, os.path.join(os.getcwd(), 'pretrain_info/py_color.txt'), threed=True)
                          
                print('\ttesting performance, testing data size {}, hit@6:{}, hit@3:{}, hit@1:{}'.format(origin_len, test_hit6, test_hit3, test_hit1))
                test_df = test_df.append({'hit@6':test_hit6, 
                                          'hit@3':test_hit3, 
                                          'hit@1':test_hit1,
                                          'compute_time': time_lag,
                                          'gflops': flops,
                                          'param_size': params}, ignore_index=True)
                
                all_preds.append(preds)
                all_scores.append(scores[0])
                all_pr_scores.append(scores[1])
                all_labels.append(list(origin_batch_labels.cpu().numpy()))
    
    if flag=='evaluation':
        ''''roc analysis and drawing'''
        res_df = pd.DataFrame({'ytest':utils.flatten(all_labels),'scores':utils.flatten(all_scores), 
                               'preds':utils.flatten(all_preds), 'prscores':utils.flatten(all_pr_scores)})
        n_class = 28 if args.fineg else 23
        compute_roc_and_prs(res_df, n_class, os.path.join(os.getcwd(), 'roc'), 
                    os.path.join(os.getcwd(), 'results/roc_df.csv'),
                    os.path.join(os.getcwd(), 'results/tfr_df.csv'),
                    os.path.join(os.getcwd(), 'results/pr_df.csv'))
        
        ''''confusion matrix analysis and drawing'''
        cm_metric = Metrics(all_labels, all_preds, id2task, args)
        precision_scores, recall_scores, f1_scores = cm_metric.return_metrics()
        #metric_df = pd.DataFrame({'P':list(precision_scores.values()), 'R':list(recall_scores.values()), 'F1':list(f1_scores.values())})
        #metric_df.to_csv('./results/metrics_prf_df.csv', mode='a')
        print('precision: ', np.mean(list(precision_scores.values())), 
              'recall: ', np.mean(list(recall_scores.values())), 
              'f1: ', np.mean(list(f1_scores.values())))
        #cm_metric.plot_confusion_matrix(all_labels, all_preds, os.path.join(os.getcwd(), 'cm'), normalize=True)
        
    train_df = train_df.append(train_df.mean(axis=0).rename('means'))
    train_df = train_df.append(train_df.var(axis=0).rename('vars'))
    test_df = test_df.append(test_df.mean(axis=0).rename('means'))
    test_df = test_df.append(test_df.var(axis=0).rename('vars'))
    
    if not gnn==None:
        return all_loss, (gnn_all_entity, gnn_all_rels), test_df, train_df
    else:
        return all_loss, None, test_df, train_df

                                
def evaluate_complex_agg(tasks_embeds, outputs, orignal_labels, origin_len, cuda, topk=None, id2task=None, sp_labels=None, abs_tlabels=None, task2si=None, tid2tid=None, flag='train', full_mode='complex'): #id2task is for demonstration
    all_preds = []
    all_scores = []
    all_pr_scores = []
    hit = 0
    
    if flag=='evaluation' or full_mode=='simple': # the range searching is only applicable for testing
        task2id = {v:k for k, v in id2task.items()}
        for i, label in enumerate(orignal_labels): # here label is the current task label, the golden class = current batch_size
            task_ranges_sp = []
            if not task2si==None and len(sp_labels)>0:
                sp_label = [int(n) for n in sp_labels[i].data.cpu().detach().numpy()] # as there can be multiple sp id for one data
                for t, sp in task2si.items():
                    if set(sp_label).issubset(set(sp)):
                        task_ranges_sp.append(task2id[t])           
            
            task_ranges_abt = []
            ab_tlabel = abs_tlabels[i] # there is only one abstract task given a detailed task
            if not tid2tid==None:    
                for did, aid in tid2tid.items(): # task2abd {detail task id : abs task id}
                    if aid==ab_tlabel:
                        task_ranges_abt.append(did)
                task_ranges = list(set(task_ranges_sp) and set(task_ranges_abt)) # ***overlap...
            else:
                task_ranges = task_ranges_sp
                
            if cuda:
                task_ranges = torch.tensor(task_ranges).cuda()
            else:
                task_ranges =  torch.tensor(task_ranges)
            
            current_outputs = outputs[i]
            
            preds_full = torch.matmul(current_outputs, tasks_embeds.t())
            if len(task_ranges)>0:
                task_ranges_embeds = torch.index_select(tasks_embeds, 0, task_ranges)
                preds = torch.matmul(current_outputs, task_ranges_embeds.t())
            else:
                preds=preds_full
            
            num_list = [n for n in preds.cpu().detach().numpy().tolist()]
            num_list_full = [n for n in preds_full.cpu().detach().numpy().tolist()]
            
            top_id = num_list_full.index(nsmallest(1, num_list_full, key=lambda x: abs(x-np.max(num_list)))[0])
            top_ids_full = list(map(num_list_full.index, heapq.nlargest(topk, num_list_full)))
            
            try:
                top_ids_full.pop(top_ids_full.index(top_id))
                top_ids_full.insert(0, top_id)
            except:
                top_ids_full.insert(0, top_id)
                top_ids_full.pop()           
            
            all_preds.append(top_ids_full[0]) # regardless of K in top-K, only consider the first item top_ids_full for CM building
            
            scores = np.array(num_list_full)
            pr_scores = utils.softmax(num_list_full)
            all_scores.append(scores)
            all_pr_scores.append(pr_scores)
            
            if label in top_ids_full: 
                hit += 1
        hit_acc = round(hit/origin_len, 3)        
        return hit_acc, all_preds, (all_scores, all_pr_scores)
    
    else:
        outputs = outputs[:origin_len]
        if full_mode=='simple':
            outputs = torch.mean(outputs, 1)
        preds = torch.matmul(outputs, tasks_embeds.t())
        current_size = preds.shape[0] # batch size considering the last batch
        res_preds = []
        for i in range(current_size):
            num_list = preds.cpu().data[i].detach().numpy().tolist()
            top_ids = list(map(num_list.index, heapq.nlargest(topk, num_list)))
            res_preds.append(top_ids)
            all_preds.append(top_ids[0]) # regardless of K in top-K, only consider the first item top_ids_full for CM building
            
        for i, label in enumerate(orignal_labels):
            if label in res_preds[i]:
                hit += 1
                
        hit_acc = round(hit/current_size, 3)
        return hit_acc, all_preds, None


def get_batch_loss(out, labels, loss_name='cross_entropy', targets=None, origin_len=None):
    loss = None
    if loss_name == 'cross_entropy':
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(out, labels)
    elif loss_name == 'margin_loss':
        loss_func = torch.nn.MarginRankingLoss(margin=0)
        loss = loss_func(out, labels, targets)
    elif loss_name == 'triple_margin':
        loss_func = torch.nn.TripletMarginLoss(margin=0)
        anchors = labels[:origin_len]
        pos = out[:origin_len]
        neg = out[origin_len:]
        loss = loss_func(anchors, pos, neg)
    elif loss_name == 'soft_margin':
        loss_func = torch.nn.SoftMarginLoss()
        loss = loss_func(out, labels)
    return loss, loss_func

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if name.startswith('weight'):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    

     
