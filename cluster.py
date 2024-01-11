# -*- coding: UTF-8 -*-
import torch
import os
import argparse
import numpy as np
import torch.utils.data.dataloader as DataLoader
from utils import *
import train_eval
import copy
from torch.autograd import Variable
from utils import *
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from models.gnn_preprocess import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import json
from typing import Optional

def process_par_task(origin_task_ids, par_dic):
    task_ids = []
    for loc in origin_task_ids:
        par_t = par_dic[str(loc)]
        if -1 in par_t:
            task_ids.append(loc)
        else:
            replace = np.min(par_t)
            task_ids.append(replace)    
    return task_ids

def generate_dis_mat(dim, deps): # deps is a list of list [pre, suc]
    temp_mat = np.zeros((dim, dim), dtype=int)
    par_dic = {}
    for t in list(range(dim)):
        par_dic.update({str(t) : [-1, t]})
    par_dic.update({'1':[2,1], '2':[1,2]})
    
    origin_task_ids = list(range(dim))
    task_ids = process_par_task(origin_task_ids, par_dic)
    for i in origin_task_ids: # i for lines
        for j in origin_task_ids: # i for columns
            if i>j:
                temp_mat[i][j] = 0
            elif j==i:
                temp_mat[i][j] = 0
            else: # i<j
                temp_mat[i][j] = task_ids[j] - task_ids[i]
    return temp_mat

def cat_deps_vec_old(vec_, deps, task_embeds): # deps is a list of list [pre, suc]
    deps_embeds = []
    for i, de in enumerate(deps):
        temp_dep_embeds = []
        deps[i] = [int(d) for d in de]
        temp_dep_embeds.append(task_embeds[deps[i][0]])
        temp_dep_embeds.append(task_embeds[deps[i][1]])
        deps_embeds.append(np.hstack(temp_dep_embeds))
    deps = np.array(deps)
    deps_embeds = np.vstack(deps_embeds)
    vec_ = np.hstack((vec_, deps_embeds))
    return vec_

def process_final_lables(raw_labels):    
    res_l = []
    temp_l = 0
    for i, pl in enumerate(raw_labels):
        if i==0:
            res_l.append(0)
            continue
        test = raw_labels[:i]
        if not pl in test:
            temp_l += 1
            res_l.append(temp_l)                
        else:
            res_l.append(res_l[list(test).index(pl)])
    return res_l

def cluster_dist(vec_, dis_mat, avg_task=None, lamb=10):
    res_mat = np.zeros((vec_.shape[0], vec_.shape[0]))
    for i, line in enumerate(res_mat):  
        for j, _ in enumerate(line):
            t_dis = dis_mat[i][j] * avg_task
            adj = t_dis/lamb
            temp = eucli_dist(vec_[i], (vec_[j] + adj))
            line[j] = temp
    return res_mat

def eucli_dist(x, y):
    return np.sqrt(sum(np.power((x - y), 2)))

def cos_dist(x, y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

def mat_turn(matrix):
    n = len(matrix)
    m = 1
    for j in range(n):
        for i in range(m, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        m += 1
    return matrix

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    sch.set_link_color_palette(["#B061FF","#61ffff"])
    sch.dendrogram(linkage_matrix, color_threshold=None, leaf_font_size=12, leaf_rotation=45, above_threshold_color='grey')
    #sns.clustermap(linkage_matrix, linewidths=.5)

'''=========================================================Main drawing code=========================================================='''
    
if __name__ == '__main__':
    train_eval.setup_seed(20)        
    '''step 1: load cluster data and pre-trained info''' 
    args = load_args('pretrain_info/args_set.txt')
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    if args.fineg:
        task_len = 28
        data_pth = 'data/cluster_detail.txt'
        manual_packages = [0,1,1,2,2,3,3,4,4,4,4,5,5,6,6,6,7,7,7,8,9,10,10,11,11,12,12,12]
    else:
        task_len = 23
        data_pth = 'data/cluster.txt'        
        manual_packages = [0,1,1,1,2,3,3,4,4,5,5,5,5,6,6,7,8,8,9,9,9,10,10]
        
    task_labels_, products_, deps_ = generate_entities(os.path.join(os.getcwd(), data_pth), cluster=True)
    corpus, word2id, id2word, task2id, id2task, all_term2id = load_x2id_dic('pretrain_info/dics.json')
    
    embed_matrix = load_pretrain_mat('pretrain_info/embed_mat.txt', 'float32').to(device)
    tasks_embeds = load_pretrain_mat('pretrain_info/task_embed.txt', 'float32').to(device)
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
      
    '''run clustering'''
    cluster_data = generate_data(products_, task_labels_, word2id, task2id, args.max_len) #args.data_split
    cluster_dst = Dst(cluster_data, word2id, args.tensor_max_len)
    cluster_loader = DataLoader.DataLoader(cluster_dst, len(cluster_data), shuffle=False, num_workers=args.num_worker)
        
    norm = False # True False
    n_clusters = 11 # the expected number of packages
    theta = 1.5 # weight of task name, we do not change this, as we at least know the task name
    lamb = 0.8 # weight of work dependencie the small, the more important
    gamma = 1.0 # weight of prodcuts
    sigma = 1.5#1.5 # weight of spatial info
    
    for _, item in enumerate(cluster_loader):
        batch_products, batch_labels, batch_masks, _, _ = item
        batch_products.to(device)
        batch_labels.to(device)
        batch_masks.to(device)
        batch_data = Variable(utils.generate_batch_data(batch_products, embed_matrix, id2word, word2id, args))
         
        products_vec, _ = model.forward(batch_data, batch_masks=batch_masks, cluster=True) # product_vec t_num*embed_dim
        if args.use_gnn:
            sp_batch_inputs = reconstruct_terms_form_ids(batch_products, all_term2id, word2id, id2word, args.rel_dic)
            conv_sp_vec = []
            for sp_triples in sp_batch_inputs:
                sp_triples = torch.tensor(sp_triples)
                gnn_conv_input = torch.cat((gnn_all_entity[sp_triples[:,0]].unsqueeze(1), gnn_all_rels[sp_triples[:,1]].unsqueeze(1), 
                                            gnn_all_entity[sp_triples[:,2]].unsqueeze(1)), dim=1)
                gnn_conv_out = gnn_model.model_gat.convKB(gnn_conv_input, args)            
                conv_sp_vec.append(gnn_conv_out)
            conv_sp_vec = torch.mean(torch.stack(conv_sp_vec), dim=1)           
            cluster_vec = torch.cat((theta * tasks_embeds + gamma * products_vec, sigma * conv_sp_vec), dim=1) # task_name + products used + sptial info after conv
        
        if norm==True:
            cluster_vec = normalization(cluster_vec.cpu().detach().numpy(), mode=1)
        
        dis_mat = generate_dis_mat(cluster_vec.shape[0], deps_) # tasks_embeds.cpu().detach().numpy()        
        dis_mat_origin = copy.deepcopy(dis_mat)
        dis_mat = dis_mat_origin + mat_turn(dis_mat)
        #print('\n', dis_mat)
        cluster_vec = cluster_vec.cpu().detach().numpy()
        tasks_embeds = tasks_embeds.cpu().detach().numpy()
        m2 = cluster_dist(cluster_vec, dis_mat, avg_task=np.mean(cluster_vec, axis=0), lamb=lamb)
        
        h_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='precomputed', compute_distances=True) #euclidean cosine precomputed
        h_model = h_model.fit(m2) #m1
        s_score = silhouette_score(m2, h_model.labels_)
        pre_labels = process_final_lables(h_model.labels_)
        df = pd.DataFrame({'task_id':list(range(0, task_len)), 'pre_package':h_model.labels_, 'process_labels':pre_labels, 'manual_res':manual_packages})
        print(df)
        print('Silhouette Coefficient: ', s_score) 
    
    print('clustering finished...')
    
    #plot_dendrogram(h_model)
    #ax_h = sns.clustermap(m2, method='average', linewidths=.5)
    #plt.show()



