import numpy as np
import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import torch.nn as nn
import re
import os
from thop import profile # for evaluating model complexity
import json
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from sklearn import datasets
    
class Dst(Dataset.Dataset):
    def __init__(self, data, word2id, tensor_max_len):
        self.data = [d[0][0] for d in data]
        self.masks = [d[0][1] for d in data]
        self.labels = [d[1] for d in data]
        self.sp_labels = [d[2] for d in data]
        self.abs_tlabels = [d[3] for d in data]
        self.word2id = word2id       
        self.tensor_max_len = tensor_max_len 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.stack([ padding_single_tensor(d, self.tensor_max_len, self.word2id) for d in self.data[index] ]) # data is triples
        mask = torch.tensor(self.masks[index])
        label = torch.tensor(self.labels[index])
        sp_labels = self.sp_labels[index]
        abs_tlabels = self.abs_tlabels[index]
        return data, label, mask, sp_labels, abs_tlabels

'''perform complexity analysis, flops and paramter size'''
def complexity_analyze(model, data):
    '''compute FLOPs per unit'''
    input_size = list(data[0].size())[1:]
    input_size.insert(0, 1)
    inputs = torch.rand(input_size) # only consider 1 piece of data
    if data[1].cuda: # data[1]=args
        inputs = inputs.cuda()

    flops, params = profile(model, inputs, verbose=False)
    gflops_ = flops/1e+9
    params_ = params/1e+6
    return gflops_, params_

'''visualize features in the last hidden layer'''
def tsne_analyze(raw_data, hidden_states, num_class, col_path, tsne=None, threed=True):
    def get_color(col_path, num_class):
        pre_colors = dict()
        with open(col_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = [re.sub(r"[\', ]", '', l) for l in line.strip().split(':')]
                pre_colors.update({line[0] : line[1]})
        
        col_candiates = list(pre_colors.values())
        colors = []        
        for _ in range(num_class):
            randi = random.randint(0, num_class-1)
            tempc = col_candiates.pop(randi)
            colors.append(tempc)
        return colors
    
    try:
        hidden_states = np.array(hidden_states.cpu().numpy())
        raw_data = np.mean(np.array(raw_data.cpu().numpy()), axis=1)
    except:
        hidden_states = np.array(hidden_states.numpy())
        raw_data = np.mean(np.array(raw_data.numpy()), axis=1)
    
    if tsne==None:
        if threed:
            tsne = TSNE(n_components=3, init='pca')
        else:
            tsne = TSNE(n_components=2, init='pca')
    res = tsne.fit_transform(hidden_states)
    raw = tsne.fit_transform(raw_data)
    #res = (res - res.min()) / (res.max() - res.min()) # data normalization
    
    #colors = get_color(col_path, num_class)
    _, colors = datasets.make_s_curve(res.shape[0], random_state=0)
    fig = plt.figure(figsize=(16, 8))
    if threed:
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(raw[:,0], raw[:,1], raw[:,2], c=colors, linewidths=0.5, marker='o', cmap=plt.cm.Spectral)
        ax.set_title('raw inputs')
        ax.view_init(4, -72)
        ax = fig.add_subplot(212, projection='3d')
        ax.scatter(res[:,0], res[:,1], res[:,2], c=colors, linewidths=0.5, marker='o', cmap=plt.cm.Spectral)
        ax.set_title('last hidden')
        ax.view_init(4, -72)
    else:
        ax = fig.add_subplot(2,1,1)
        ax.scatter(raw[:,0], raw[:,1], c=colors, linewidths=0.5, marker='o', edgecolors='k', cmap=plt.cm.Spectral)
        ax.set_title('raw inputs')
        ax = fig.add_subplot(2,1,2)
        ax.scatter(res[:,0], res[:,1], c=colors, linewidths=0.5, marker='o', edgecolors='k', cmap=plt.cm.Spectral)
        ax.set_title('last hidden')
    fig.tight_layout()
    #plt.show()
    

'''load pre-trained embedding matrix'''
def load_pretrained_embedding(corpus, embedding_file, split_char, embedding_dim, add_words=None, threshold=5):
    embeddings_dict = {}
    with open(embedding_file, 'r', encoding='UTF-8') as f: 
        for line in f:
            values = line.strip().split(split_char)
            if len(values) < threshold: # handling some special lines, e.g. tells us the number and dimension of the words in the file
                continue
            word = values[0] # values is a long list, the first element is the character/word, the others are embedding values
            embedding = np.asarray(values[1:], dtype='float32') # convert the embedding values
            embeddings_dict[word] = embedding # add above information to the dictionary
        if not add_words==None: # handle 'PAD' & 'DUMMY', randomly generate embeddings for them
            for sw in add_words:
                embeddings_dict.update({sw : np.random.uniform(low=-1.0, high=1.0, size=embedding_dim)}) 
        print('found {} word vectors in the entire pre-trained embeddings\n'.format(len(embeddings_dict)))
    
    word2id = {}
    corpus = add_words + corpus # special tokens are put at the beginning, PAD=0, DUMMY_TASK=1
    corpus_embedding_matrix = torch.Tensor(len(corpus), embedding_dim)
    for i, word in enumerate(corpus): # corpus: a list of all unique words in the training dataset
        word_embed = convert_data_to_tensor([word], embeddings_dict, dim=embedding_dim, pri=False)[0] # word is a single word
        corpus_embedding_matrix[i] = word_embed
        word2id.update({word:i})
    id2word = {v:k for k,v in word2id.items()}
    return corpus_embedding_matrix, word2id, id2word    

'''convert txt to embedding'''
def convert_data_to_tensor(txts, embedding_dic, mu=0, sigma=0.5, dim=50, pri=True):
    data_lst = []
    miss_count = 0
    for txt in txts:
        if len(txt.split(' '))>1: # in case txt is a phrase rather than a single word
            words = [w.strip() for w in txt.split(' ')]
            temp_tensors = [convert_data_to_tensor([w.strip()], embedding_dic, mu, sigma, dim)[0] for w in words]
            total_tesnor = temp_tensors[0]
            for i, t in enumerate(temp_tensors):
                if i==0:
                    continue
                total_tesnor = torch.cat((total_tesnor, t), 0)
            data_lst.append(torch.mean(total_tesnor, 0).numpy())
        else:
            if txt in list(embedding_dic.keys()):
                data_lst.append(embedding_dic[txt])
            else:
                data_lst.append(np.random.normal(mu, sigma, dim))
                miss_count += 1
    tensor = torch.FloatTensor(np.array(data_lst))
    if pri:
        print('found {} words without pre-trained embeddings'.format(miss_count))
    return tensor, miss_count

def padding_sequence(data, max_len, word2id, lst=False): # seq max_len is temporarily set as 15
    current_len = len(data)
    pad_res = data
    if lst:
        pad = [word2id['PAD']]
    else:
        pad = word2id['PAD']
    if current_len < max_len:
        while(len(pad_res) < max_len):
            pad_res.append(torch.tensor(pad))
    else:
        pad_res = pad_res[ : max_len]
    return pad_res

def padding_single_tensor(data, tensor_max_len, word2id): # tensor_max_len is temporarily set as 10
    try:
        current_len = len(data)
    except:
        current_len = 1
        data = torch.tensor([data])
    if len(data) > tensor_max_len:
        data = data[:tensor_max_len]
    pad = word2id['PAD']
    need_len = tensor_max_len - current_len
    pad_tensor = torch.tensor([pad] * need_len)
    pad_res = torch.cat((data, pad_tensor))
    return pad_res

def padding_tensors(batch_tensors, max_len, tensor_max_len, word2id):
    pad_res = []
    for tensors in batch_tensors:
        lag = max_len - tensors.shape[0]
        empty_tensor = torch.zeros((lag, tensor_max_len))
        empty_tensor.fill_(word2id['PAD']) 
        tensors = torch.cat((tensors, empty_tensor))
        pad_res.append(tensors)
    pad_res = torch.stack(pad_res)
    return pad_res

'''process corpus and product/task entities'''
def generate_entities(data_path, rels=None, cluster=False):
    task_labels = []
    all_products = []
    all_terms = []
    deps = []
    with open(data_path, mode='r', encoding='utf-8') as f:
        for line in f:
            temp_task = line.strip().split('--')[0].strip()
            temp_task = re.sub(r'[^ a-zA-Z0-9]', '', temp_task) # remove characters that are not numbers/alphbets/spaces
            temp_task = re.sub(r'\s+', ' ', temp_task) # remove multiple spaces
            task_labels.append(temp_task.lower())
            temp_products = [p.strip().lower() for p in line.strip().split('--')[1].split(',') if not p=='']
            if cluster==True:
                temp_deps = temp_products[-3:]
                temp_products=temp_products[:-3]
                deps.append(temp_deps)
            for i, _ in enumerate(temp_products):
                temp_products[i] = re.sub(r'[^ a-zA-Z0-9]', '', temp_products[i])
                temp_products[i] = re.sub(r'\s+', ' ', temp_products[i])
            all_products.append(temp_products)
        all_terms.extend(task_labels) # terms are phrases & corpus are individual words
        all_terms.extend(all_products)
        all_terms = list(set(list(flat(all_terms))))
    f.close()
    
    if cluster==True:
        return task_labels, all_products, deps
    
    corpus = []
    for term in all_terms:
        corpus.extend(term.strip().split(' '))
    corpus = list(set(corpus))
    if not rels==None:
        for rel in list(rels.keys()):
            if not rel in corpus:
                corpus.append(rel)
    
    task2id = {}
    for i, t in enumerate(sorted(set(task_labels), key=task_labels.index)): 
        task2id.update({t:i})
    id2task = {v:k for k,v in task2id.items()}
    return task_labels, all_products, corpus, task2id, id2task

def split_dataset(ratio, input_data):
    all_idx = list(range(0, len(input_data))) # all indices: 0, 1, 2... len(all_data)
    res_size = int(np.floor(len(all_idx) * ratio))
    count = 0
    select_idx = []
    assert len(all_idx) == len(input_data)
    
    while (count < res_size):
        temp_idx = random.randint(0, len(all_idx)-1)
        if not temp_idx in select_idx:
            select_idx.append(temp_idx)
            count += 1
    
    res_data = [input_data[i] for i in select_idx]
    remain_idx = [i for i in all_idx if not i in select_idx]
    remain_data = [input_data[i] for i in remain_idx]
    return res_data, remain_data

def generate_data(all_products, task_labels, word2id, task2id, max_len, sp_dic=None, id2word=None, data_split=None, tid2tid=None, rel_name='contains'): # generate all data with ids
    data = []
    labels = []
    sp_labels = []
    abt_labels = []
    try:
        rel_id = word2id[rel_name]
    except:
        rel_id = -1
    for i, products in enumerate(all_products): # each products is a list of single products
        temp_ids = []
        temp_spatials = []
        task_label = task2id[task_labels[i]] # note the task2id is the detail_tid, and the task_label is the detailed tid
        for product in products: # each product contains several words (including spatial relations)
            product_words_ids = torch.tensor([word2id[p] for p in product.strip().split(' ')])
            temp_ids.append(product_words_ids)
        temp_ids.insert(0, torch.tensor(word2id['DUMMY_TASK']))
        temp_ids = padding_sequence(temp_ids, max_len, word2id) # including padding or truncating
        masks = [0 if word2id['PAD'] in tid.numpy() else 1 for tid in temp_ids] # if the product is a PAD
        
        if not sp_dic==None:
            # if the product implies spatial info          
            sp_ids = [list(tid.numpy())[0:list(tid.numpy()).index(rel_id)] if rel_id in tid.numpy() else -1 for tid in temp_ids]
            
            current_sp_ids = []
            for sp_id in sp_ids:
                if not sp_id==-1:   
                    sp_label = sp_dic[(' '.join([id2word[id] for id in sp_id])).strip()]
                    current_sp_ids.append(sp_label)
                    current_sp_ids = list(np.unique(current_sp_ids))
                else:
                    continue
            sp_labels.append('+'.join([str(i) for i in current_sp_ids])) 
        else:
            sp_labels.append(-1)
            
        if not tid2tid==None:
            abt_labels.append(tid2tid[task_label]) # tid2tid {detail_tid : abstract_tid}
        else:
            abt_labels.append(-2)
        
        data.append((temp_ids, masks))
        labels.append(task_label)
        
    all_data = list(zip(data, labels, sp_labels, abt_labels))
    if not data_split==None:
        train_r, dev_r = data_split
        train_data, dev_data = split_dataset(train_r, all_data) # dev_data, test_data = split_data(dev_r, dev_test_data)
        return train_data, dev_data
    else:
        return all_data

def generate_batch_data(batch_rows, embed_matrix, id2word, word2id, args, sp_dir=None): # generate batch (embeddings) for both products and task labels
    batch_res = []
    for i, row in enumerate(batch_rows): # batch_products: batch_size, max_len (e.g., 10 or 15), tensor_max_len (e.g., 5 or 10)
        temp_res_embeds = []
        for k, words in enumerate(row): # row=products for one task; words=product, which is a product name containing certain words          
            init_word = id2word[int(words[0].numpy())]
            if not init_word == 'PAD':
                temp_ids = torch.tensor([int(word_id.numpy()) for word_id in words if not id2word[int(word_id.numpy())]=='PAD'])
            else:
                temp_ids = torch.tensor([word2id['PAD']]) # here only one element is ok
            if args.cuda:
                temp_ids = temp_ids.cuda()
                embed_matrix = embed_matrix.cuda()
            temp_embds = torch.index_select(embed_matrix, 0, temp_ids) # embedding for the current product/term
            temp_embds = torch.mean(temp_embds, 0) # get average of the terms
            temp_res_embeds.append(temp_embds) # products_embeds: list, each element is max_len (e.g., 10) * (embed_dim, )
        batch_res.append(torch.stack(temp_res_embeds)) # batch_es: list, each element is a tensor obtained by stacking products_embeds
    res = torch.stack(batch_res) # res: batch_size, max_len, embed_dim
    return res

def generate_neg_data(products, labels, batch_masks, sp_labels, args):
    '''
        a key different between this and transE negative sampling is that this contaminate the task laebls, rather than the data
    '''
    neg_pos_ratio = args.neg_ratio
    if args.loss_func == 'soft_margin':
        neg_pos_ratio = 1
    last_idx = len(products)
    unique_labels = list(range(0, args.class_num))
    
    products = products.repeat((neg_pos_ratio, 1, 1))
    batch_masks = batch_masks.repeat((neg_pos_ratio, 1, 1))
    batch_masks = batch_masks.reshape((last_idx * neg_pos_ratio, -1, products.shape[1])).unsqueeze(1)
    sp_labels = sp_labels * neg_pos_ratio #repeat((neg_pos_ratio, 1, 1))
    #sp_labels = sp_labels.reshape((last_idx * neg_pos_ratio, -1, products.shape[1])).unsqueeze(1)    
    
    labels = labels.repeat((neg_pos_ratio))
    for i, l in enumerate(labels):
        if i<last_idx: # ensure the previous 'last_indx' data are true data
            continue
        else: # else, replace the taks label to generate negative data
            while(True):
                temp_label = random.choice(unique_labels)
                if not temp_label == l: # select a task label other than the original one
                    labels[i] = temp_label
                    break
    targets = torch.tensor([int(1)] * last_idx + [int(-1)] * (neg_pos_ratio-1) * last_idx) # -1 refers to the original data
    return products, labels, batch_masks, sp_labels, targets.unsqueeze(-1)

def generate_task_sp_embeds(sp_dir, id2task, all_term2id, gnn_out_entity, cuda): # generate spatial embeddings (based on spatial terms) for tasks
    task_sp_embeds = []
    for tid, sp_terms in sp_dir.items(): # a term is a phrase indicating the spatial space, a task can involve 2 or more spaces
        t_name = id2task[tid]
        temp_sp_ids = []
        for term in sp_terms:
            temp_sp_ids.append(all_term2id[term])
        
        temp_sp_ids = torch.tensor(temp_sp_ids)
        if cuda==True:
            temp_sp_ids = temp_sp_ids.cuda()
        temp_sp_embeds = torch.mean(torch.index_select(gnn_out_entity, 0, temp_sp_ids), dim=0)
        task_sp_embeds.append(temp_sp_embeds)
    task_sp_embeds = torch.stack(task_sp_embeds)
    return task_sp_embeds

def process_sp_info(sp_pth): # build spatial information directory i.e., {task_id:sp_name}
    task_sp_dir = {}
    with open(sp_pth, encoding='utf-8') as f:
        for line in f:
            t_name = line.strip().split('--')[0]
            t_name = re.sub(r'[^ a-zA-Z0-9]', '', t_name) # remove characters that are not numbers/alphbets/spaces
            t_name = re.sub(r'\s+', ' ', t_name).lower().strip()
            sp_info = [sp.lower() for sp in line.strip().split('--')[1].split(';')]
            task_sp_dir.update({t_name : sp_info})
    return task_sp_dir

def process_task_hier(t2t_pth, abs_tid, detail_tid, none_tag='NA'): # note the abs_tid and detail_tid can be the same (same granularity)
    task2task = {}
    tid2tid = {}
    if len(abs_tid)==len(detail_tid):
        tid_detial = {v:k for k,v in detail_tid.items()}
        task2task = {k:tid_detial[v] for k,v in abs_tid.items()}
        tid2tid = {v:detail_tid[k] for k,v in abs_tid.items()}
    else: 
        with open(t2t_pth) as f:
            for line in f:
                t_abs = line.strip().split('--')[0].strip() # in the txt file, the left is the abstract task, the right is the fine-grained task
                t_det = [t.strip() for t in line.strip().split('--')[1].split(';')]
                if not none_tag in t_det:
                    for t in t_det:
                        task2task.update({t:t_abs})
                        tid2tid.update({detail_tid[t] : abs_tid[t_abs]})
                else:
                    task2task.update({t_abs : t_abs}) # if there is no detailed task, then use t_abs to replace it 'the first t_abs'
                    tid2tid.update({detail_tid[t_abs] : abs_tid[t_abs]}) # detailed task id : abstract task id
    return task2task, tid2tid

def process_task_si(task_labels, task2sp, sp_dic):
    task_names = list(np.unique(task_labels))
    task2si = {}
    for tn in task_names:
        temp_sp_names = task2sp[tn]
        temp_sid = [ sp_dic[n] for n in temp_sp_names ]
        task2si.update({tn : temp_sid})
    return task2si

def process_task_ids(args, task_id, word2id, id2task): # convert textual task names to ids using word2id
    if type(task_id) == torch.Tensor:
        idx = task_id.item()
    else:
        idx = task_id   
    task_name = id2task[idx].strip()
    task_name_ids = torch.tensor([word2id[w] for w in task_name.split(' ')])
    task_name_ids = padding_single_tensor(task_name_ids, args.tensor_max_len, word2id) 
    return task_name_ids

def process_str_dics(dic_):
    res_dic = {}
    res_dic_reverse = {}
    for k, v in dic_.items():
        res_dic.update({k : int(v)})
        res_dic_reverse.update({int(v) : k})
    return res_dic, res_dic_reverse

def load_pretrain_mat(pth, dp): #'float32'
    embedding_file = os.path.join(os.getcwd(), pth)
    res_embed_mat = []
    with open(embedding_file, 'r', encoding='UTF-8') as f:
        for line in f:
            values = line.strip().split(' ')
            res_embed_mat.append(torch.tensor(np.asarray(values, dtype=dp)))
    res_embed_mat = torch.stack(res_embed_mat)
    return res_embed_mat

def load_x2id_dic(x2id_pth): # this dic record xx2id
    with open(os.path.join(os.getcwd(), x2id_pth), mode='r') as jf: 
        dics = json.load(jf)
        str_word2id = dics['word2id']
        word2id, id2word = process_str_dics(str_word2id)
        str_task2id = dics['task2id']
        task2id, id2task = process_str_dics(str_task2id)
        try:
            str_term2id = dics['term2id']
            all_term2id, _ = process_str_dics(str_term2id)
        except:
            all_term2id = None
        corpus = dics['corpus']
    jf.close()
    return corpus, word2id, id2word, task2id, id2task, all_term2id

def load_args(pth):
    args = argparse.ArgumentParser()
    args.add_argument("-ld", "--mlp_layer_dims", default=[256, 256], help="the dims of different layers for the MLP model")
    args.add_argument("-cf", "--cnn_filters", default=[(1, 8, 5), (1, 4, 25), (1, 2, 45)], help="CNN kenerls")
    args.add_argument("-rid", "--rel_dic", default={'contains':0, 'constrains':1}, help='relation2id')
    
    with open(os.path.join(os.getcwd(), pth), 'r') as af:
        for line in af:
            k = line.strip().split(':')[0].strip()
            v = line.strip().split(':')[1].strip()
            tp = line.strip().split(':')[-1].strip()
            if 'bool' in tp:
                v = True if v.lower() == 'true' else False
            elif 'int' in tp:
                v = int(v)
            elif 'float' in tp:
                v = float(v)
            elif 'str' in tp:
                pass
            else:
                continue
            args.add_argument('--'+k, default=v)
            #args_dic.update({k : v})
    args = args.parse_args() 
    return args

def one_hot(labels, batch_size, class_num):
    labels = labels.view(batch_size, 1)
    m_zeros = torch.zeros(batch_size, class_num)
    one_hot = m_zeros.scatter_(1, labels, 1)
    one_hot = one_hot.long() #print(one_hot.type(), ' ',one_hot[1:10])
    return one_hot

def softmax(x):
    row_max = np.max(x)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    res = x_exp / x_sum
    return res

def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

def flat(input_lst):
    lst= []
    for i in input_lst:
        if type(i) is list:
            for j in i:
                lst.append(j)
        else:
            lst.append(i)
    return(lst)

def normalization(data, mode=1):
    if mode==1:
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma
    else:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
def write_tensor(tensor_, path):
    with open(os.path.join(os.getcwd(), path), mode='w') as f:
        np.savetxt(f, tensor_.cpu().detach().numpy())
    f.close()

def write_dev_data(dev_data, id2task, id2word, out_pth):
    with open(out_pth, mode='w') as f:
        for i, dev in enumerate(dev_data):
            task = id2task[dev[1]]
            f.write(task + '--')      
            tensor_products = dev[0][0]
            str_products = []
            for p in tensor_products:
                try:
                    p_name = ' '.join([id2word[s] for s in p.tolist()]).strip()
                except:
                    p_name = id2word[p.numpy().tolist()]
                    if p_name == 'PAD':
                        continue
                #print(p_name)
                f.write(p_name + ', ')
                str_products.append(p_name)
            f.write('\n')
    f.close() 
    
            
            
