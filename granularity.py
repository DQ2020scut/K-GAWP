# -*- coding: UTF-8 -*-
import torch
import os
import numpy as np
import torch.utils.data.dataloader as DataLoader
from utils import *
import train_eval
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.gnn_preprocess import *
from sklearn.metrics.pairwise import pairwise_distances

GENERALIZATION_TEST = True
if __name__ == '__main__':
    train_eval.setup_seed(20)
    '''step 1: load pre-trained info'''
    args = load_args('pretrain_info/args_set.txt')
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    corpus, word2id, id2word, task2id, id2task, all_term2id = load_x2id_dic('pretrain_info/dics.json')
    sp_dic = {'none':0, 'plane':1, 'room space':2, 'door interface':3, 'window interface':4, 'wall plane':5, 'mep interface':6, 'ceiling plane':7,
              'floor plane':8}
  
    embed_matrix = load_pretrain_mat('pretrain_info/embed_mat.txt', 'float32').to(device)
    #tasks_embeds = load_pretrain_mat('pretrain_info/task_embed.txt', 'float32').to(device)
    gnn_all_entity = load_pretrain_mat('pretrain_info/gnn_ent_embeds.txt', 'float32').to(device) # the entity for terms 
    gnn_all_rels = load_pretrain_mat('pretrain_info/gnn_rel_embeds.txt', 'float32').to(device)
    
    '''step 2: load model and data'''
    model, gnn_model = train_eval.select_model(args=args, corpus=corpus, all_products=None, word2id=word2id, id2word=id2word, 
                                               ent_embeds=embed_matrix, cluster_or_granu=True, all_term2id=all_term2id)
    
    if args.use_gnn == True:
        model_state = torch.load(os.path.join(os.getcwd(), 'pretrain_info/agg_gnn.pkl'), map_location=device)
        gnn_model_state = torch.load(os.path.join(os. getcwd(), 'pretrain_info/gnn.pkl'), map_location=device)
        gnn_model.load_state_dict(gnn_model_state['model'], strict=False)
        gnn_model = gnn_model.to(device)
    else:
        model_state = torch.load(os.path.join(os.getcwd(), 'pretrain_info/agg.pkl'), map_location=device)
    
    model.load_state_dict(model_state['model'], strict=False)
    model = model.to(device)
    
    '''testing model with different granularities'''
    if GENERALIZATION_TEST or args.fineg:
        test_pth = 'data/test_spatial_data_detail.txt'
        sp_dic_pth = 'pretrain_info/sp_dir_detail.txt'
    else:
        test_pth = 'data/test_spatial_data.txt'
        sp_dic_pth = 'pretrain_info/sp_dir.txt' # different granularities have different sp_dics for generating task2sp
    task2task_pth = 'pretrain_info/task2task.txt'
    print('current data path: ', test_pth)
    
    # below, the test_tid and test_dit are generated from the specific testing data, as the task names in datasets of different granularities can be different
    test_task_labels, test_products, test_corpus, test_tid, test_dit = generate_entities(os.path.join(os.getcwd(), test_pth))
    task2sp = process_sp_info(sp_dic_pth) 
    task2si = process_task_si(test_task_labels, task2sp, sp_dic)
    task2task, tid2tid = process_task_hier(task2task_pth, task2id, test_tid) # task2id is abs_tid, return detail_id2abs_id
    
    test_data = generate_data(test_products, test_task_labels, word2id, test_tid, args.max_len, id2word=id2word, tid2tid=tid2tid, sp_dic=sp_dic)
    test_dataset = Dst(test_data, word2id, args.tensor_max_len)
    test_bs = int(np.max((args.batch_size*2, 1))) # len(test_data)
    test_loader = DataLoader.DataLoader(test_dataset, test_bs, shuffle=True, num_workers=args.num_worker)
    
    # the test_task_ids and test_task_embeds also depend on the task names in the testing data
    test_task_ids = torch.stack([process_task_ids(args, tid, word2id, test_dit) for tid in list(test_dit.keys())]).unsqueeze(1)
    test_tasks_embeds = generate_batch_data(test_task_ids, embed_matrix, id2word, word2id, args).squeeze(1)
    
    _, res, test_df, _ = train_eval.train_evaluate(args, model, None, test_loader, embed_matrix, id2word, word2id, test_tasks_embeds, gnn=gnn_model, 
                                             id2task=test_dit, task2si=None, tid2tid=None, flag='evaluation')
    
    test_df.to_csv('./results/granu_test_df.csv', mode='a')
    print('finished...')   

       
