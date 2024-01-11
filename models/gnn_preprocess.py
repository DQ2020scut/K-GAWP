import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import queue
import sys
sys.path.append("..")
import utils

def generate_gnn_terms(all_products, word2id, relation2id, split_ch=' ', end_char='PAD', dummy_t='DUMMY_TASK'):
    unique_rels = list(relation2id.keys())
    all_term2id = { dummy_t:0 } # for easy extracting embedding of DUMMY-TASK
    for products in all_products:
        for words in products:
            if len(list(set(unique_rels).intersection(set(words.split(split_ch))))) == 0:
                current_term = (' '.join([w.strip() for w in words.split()])).strip()
                if not current_term in list(all_term2id.keys()):
                    all_term2id.update({current_term : len(all_term2id)})
            else:
                relation = list(set(unique_rels).intersection(set(words.split(split_ch))))[0]
                e1, relation, e2, all_term2id = parse_rel(words, relation, all_term2id)
                
    for k in list(relation2id.keys()):
        all_term2id.update( {k:len(all_term2id)} )
    all_term2id.update({ end_char : len(all_term2id) })
    id2all_term = {v:k for k,v in all_term2id.items()}    
    return all_term2id, id2all_term

def find_gnn_terms(batch_sp_triples, all_term2id):
    batch_gnn_terms = []
    dummy_id = all_term2id['DUMMY_TASK']
    for sp_triples in batch_sp_triples:
        current_terms = []
        for spt in sp_triples:
            if spt[-1] == dummy_id: # if the sp triple is 'x constrains dummy task'
                current_terms.append(spt[0]) # then add the head term, ignore the dummy task
            else: # if the sp triple is 'x contains y', x is a spatial term
                current_terms.append(spt[0]) # then we add both the head and tail term
                current_terms.append(spt[-1])
        batch_gnn_terms.append(current_terms)
    return batch_gnn_terms

def look_up_gnn_embedings(batch_gnn_terms, gnn_out_entity, gnn_out_relation, args):
    res = []
    for ents in batch_gnn_terms:
        ents = torch.tensor([e for e in ents if not e==-1])
        lag = args.max_len - len(ents)
        if args.cuda:
            ents = ents.cuda()
        temp_res = torch.index_select(gnn_out_entity, 0, ents)
        
        if lag<=0: # the number of gnn terms exceeds args.max_len
            res.append(temp_res[:args.max_len,:])
        else:
            zeros =  torch.zeros((lag, int(args.out_size/2)))
            if args.cuda:
                zeros = zeros.cuda()
            temp_res = torch.cat((temp_res, zeros), dim=0)
            res.append(temp_res)
    return res

def parse_rel(words, rel, all_term2id): # in this function, rel is not updated in all_term2id
    temps = [w.strip() for w in words.split()]
    rel_loc = temps.index(rel)
    e1 = (' '.join(temps[0:rel_loc])).strip()
    rel = temps[rel_loc]
    e2 = (' '.join(temps[rel_loc+1:])).strip()
    if not e1 in list(all_term2id.keys()):
        all_term2id.update({e1:len(all_term2id)})
    if not e2 in list(all_term2id.keys()):
        all_term2id.update({e2:len(all_term2id)})   
    return e1, rel, e2, all_term2id

def generate_raw_adj(batch_sp_triples, all_term2id, directed=True, is_weigted=True):
    batch_init_adj_mat = []
    for sp_triples in batch_sp_triples:
        rows, cols, data = [], [], []
        for sp_triple in sp_triples:
            if True: #entity2id[e1]==0 or entity2id[e2]==0: #***通过调整这行代码可以改变图谱范围 目前来说只考虑了0***
                #rel = all_term2id[id2relation[sp_triple[1]]]
                rel = sp_triple[1]
                e1 = sp_triple[0] # head
                e2 = sp_triple[-1] # tail
                if not directed: # if not directed, then we should add both direction, i.e., e1 and e2 should be both tail and source
                    rows.append(e1)
                    cols.append(e2)
                    if is_weigted:
                        data.append(rel)
                    else:
                        data.append(1)
                # if the graph is directed, then, e1 and e2 can only be source or tail, so we only add the row/col lists once
                rows.append(e2) # e2 is the tail entity (the one that the arrow towards)
                cols.append(e1)
                if is_weigted:
                    data.append(rel)
                else:
                    data.append(1)
        batch_init_adj_mat.append(tuple((rows, cols, data)))
    return batch_init_adj_mat

def reconstruct_terms_form_ids(batch_products, all_term2id, word2id, id2word, rel2id):
    pad_id = word2id['PAD']
    rel_ids = [word2id[rn] for rn in list(rel2id.keys())]
    batch_sp_triples =[]
    
    for products in batch_products: # sp_batch_products: (batch_size,15,10); products: (15,10)
        current_products_triples =[]
        for product_data in products: # 1x10
            # find the rel location if any
            current_rel_id = None
            for rel_id in rel_ids: # note the rel_ids is the ids in the word-dict, not the relation dict
                try:
                    rel_loc = product_data.tolist().index(rel_id)
                    if not rel_loc==None:
                        current_rel_id = rel_id
                        break # one data (1x10) only has one rel
                except: # the whole row (1x10) is a term #print(product_data)
                    continue    
            if current_rel_id==None:
                term_lst = [ t for t in product_data.tolist() if not t==pad_id ]
                if len(term_lst)>0:
                    term_1 = ( ' '.join([id2word[id] for id in term_lst]) ).strip() # join the words into terms and label them using all_term2id
                    term_1_id = all_term2id[term_1]
                    term_2_id = all_term2id['DUMMY_TASK']
                    sp_rel = rel2id['constrains']
                else:
                    continue  
            else:
                term_1_lst = product_data.tolist()[ : rel_loc]
                term_1 = ( ' '.join([id2word[id] for id in term_1_lst]) ).strip() # join the words into terms and label them using all_term2id
                term_1_id = all_term2id[term_1]
                term_2_lst = [ t for t in product_data.tolist()[rel_loc+1 : ] if not t==pad_id]
                term_2 = ( ' '.join([id2word[id] for id in term_2_lst]) ).strip()
                term_2_id = all_term2id[term_2]
                sp_rel = rel2id[id2word[current_rel_id]]
            
            sp_triple = tuple((term_1_id, sp_rel, term_2_id))
            current_products_triples.append(sp_triple)
        batch_sp_triples.append(current_products_triples)  
    return batch_sp_triples
    
def constructNeighbourhood(train_data, init_adj_mt, all_term2id, nbd_size):
    train_adj_matrix = []
    train_indices_nhop = []
    for current_init_adj in init_adj_mt:
        adj_indices = torch.LongTensor([current_init_adj[0], current_init_adj[1]])
        adj_values = torch.LongTensor(current_init_adj[2])
        
        current_train_adj_matrix = (adj_indices, adj_values) # adj matrix中第一行是原triples中的tail 第二行是原triples的head
        current_graph = create_graph(current_train_adj_matrix, all_term2id)
        current_neighbours = source_get_further_neighbors(current_graph, all_term2id, nbd_size=nbd_size)
        unique_source_nodes=list(all_term2id.values())
        current_train_indices_nhop = source_get_batch_nhop_neighbors_all(unique_source_nodes, current_neighbours, nbd_size)
        current_train_indices_nhop = Variable(torch.LongTensor(current_train_indices_nhop))
        train_adj_matrix.append(current_train_adj_matrix)
        train_indices_nhop.append(current_train_indices_nhop)
    #print('train_adj_matrix (sparse tensor) shape: ', train_adj_matrix[0].shape)
    return train_indices_nhop, train_adj_matrix

def create_graph(train_adj_matrix, all_term2id): # add the 'all_term2id' for debugging
    graph = {}
    all_triples = torch.cat([train_adj_matrix[0].transpose(0,1), train_adj_matrix[1].unsqueeze(1)], dim=1)
    for data in all_triples: # tai head rel
        source = data[1].item() # head entity 注意这里根据process.py里的设计 head entity放在了[1]位置
        target = data[0].item() # source is data[1]
        value = data[2].item() # value is equal to relation
        if(source not in graph.keys()):
            graph[source] = {}
            graph[source][target] = value
        else:
            graph[source][target] = value
    return graph

def bfs(graph, source, all_term2id, nbd_size=2):
    visit = {}
    distance = {}
    parent = {}
    visit[source] = 1
    distance[source] = 0
    parent[source] = (-1, -1)
    
    q = queue.Queue()
    q.put((source, -1)) 

    while(not q.empty()):
        top = q.get() # q是先进先出的 初始化时q里只有source 所以合理top就是(0, -1)
        # 但是注意 后面每次迭代 这里的 top就会变 只有第一次是source node 第二次应该就是第一次迭代时put进去的那个些 target node中的一个
        # 这个过程会一直持续 每个 target node 作为top node时又可能关联更多邻域点
        # 相当于从 source点一直向外拓展点一个网状图
        if top[0] in graph.keys(): # top[0]代表的是这个source 点的id
            # 注意graph实际定义了每个点的一阶领域点以及与这些领域点之间点关系
            for target in graph[top[0]].keys(): #这个source点对应点所有相关联的点 也就是其1-hop邻域点
                if target in visit.keys():
                    continue #如果当前的top[0] 已经访问过这个点就继续跳到下一个
                else: #否则：把这个target node 以及 central node与之相连的关系都添加到queue里
                    q.put((target, graph[top[0]][target])) # a 2-tuple: (target node, the relation between target and the node in the last step)
                    distance[target] = distance[top[0]] + 1 # 代表这个target点 和source点的距离(i.e. 第几阶领域)
                    visit[target] = 1
                    if distance[target]>2:
                        continue
                    # parent这个字典定义了每个点的parent node以及和parent node之间的关系
                    parent[target] = (top[0], graph[top[0]][target])
                    #if distance[target] not in distance_length.keys():
                    #    distance_length[distance[target]] = 1
    neighbors = {}
    for target in visit.keys(): # 这里其实是一个反向追溯的过程 从所有和source点关联的点反推回去
        if distance[target] != nbd_size: # 假设处理的是source点的2阶邻域 那就把除了2阶邻域之外的所有邻域信息都去掉
            continue
        #print('source entity:', source, 'target entity:', target)
        relations = []
        entities = [target]
        temp = target
    
        while(parent[temp]!=(-1,-1)):
            relations.append(parent[temp][1])
            entities.append(parent[temp][0]) 
            #print('\tcurrent entity:',temp, 'parent entity:',parent[temp][0], 'relation with parent:',parent[temp][1])
            temp = parent[temp][0]
        
        # 这里一定注意neighbours点结构 {distance(几阶领域): (rels, entities)}
        if distance[target] in neighbors.keys():
            neighbors[distance[target]].append((tuple(relations), tuple(entities[:-1]))) # [:-1]是把最后一个点 i.e., source node 给去掉了
        else:
            neighbors[distance[target]] = [(tuple(relations), tuple(entities[:-1]))]
    return neighbors

# bfs是针对每个点构建的 这里是把所有点的领域结构整合在一起 返回一个大字典
def source_get_further_neighbors(graph, all_term2id, nbd_size=2):
    neighbors = {}
    #print('length of graph keys is ', len(graph.keys()))
    
    for source in graph.keys(): # 对graph内的每个点调用 bfs()
        temp_neighbors = bfs(graph, source, all_term2id, nbd_size)
        # neighbours 字典结构是 第一层keys是source id 第二层keys是每个source node的第几阶邻域(distance)
        for distance in temp_neighbors.keys():
            if(source in neighbors.keys()): # 如果当前source已经在 graph-neighbours里考虑到了
                if(distance in neighbors[source].keys()):
                    neighbors[source][distance].append(temp_neighbors[distance])
                else:
                    neighbors[source][distance] = temp_neighbors[distance]
            else:
                neighbors[source] = {}
                neighbors[source][distance] = temp_neighbors[distance]
    #print('length of neighbors dict is ', len(neighbors))
    return neighbors

# 这个函数其实是把整合好打邻域结构转换为triples
def source_get_batch_nhop_neighbors_all(unique_source_nodes, node_neighbors, nbd_size, partial_2hop=False):
    graph_path = []
    #print('length of nodes for GNN ', len(unique_source_nodes))
    count = 0
    for source in unique_source_nodes: # source node in unique entities for train
        # randomly select from the list of neighbors
        if source in node_neighbors.keys():
            nhop_list = node_neighbors[source][nbd_size] # nbd_size = 几阶领域

            for i, tup in enumerate(nhop_list): # tuo here: ((rel, rel...), (ent, ent, ...))
                if(partial_2hop and i >= 2):
                    break
                count += 1
                # tup or nhop_list[i]会是1个tuple 里面又包含两个tuple
                # tup or nhop_list[i][0][-1] 是source node到1阶领域点的rel nhop_list[i][0][0] 是1阶领域点到2阶领域点的rel
                # tup or nhop_list[i][1][0] 是2阶领域点
                # graph_path append的就是 (source node, rel from source to 1st neighbor, rel from 1st to 2nd neighbour, 2nd neighbor node)
                # graph_path.append([source, nhop_list[i][0][-1], nhop_list[i][0][0], nhop_list[i][1][0]])
                graph_path.append([source, tup[0][-1], tup[0][0], tup[1][0]])
    return np.array(graph_path).astype(np.int32)

def select_sp_data(batch_products, sp_masks, max_len, tensor_max_len, word2id):
    batch_sp_products = []
    non_batch_sp_products = []
    for i, products in enumerate(batch_products): # products: 15x10, a batch has many 'products' each for a task
        sp_products = []
        non_sp_products = []
        temp_mask = sp_masks[i].reshape((sp_masks.shape[-1], -1)).squeeze(-1).tolist()
        for k, product in enumerate(products): # product: 1x10, k from 0-14
            if temp_mask[k] == 1:
                sp_products.append(products[k])
            else:
                if not product[0]==word2id['PAD']:
                    non_sp_products.append(products[k])
        batch_sp_products.append(torch.stack(sp_products))        
        non_batch_sp_products.append(torch.stack(non_sp_products))
    
    non_batch_sp_products = utils.padding_tensors(non_batch_sp_products, max_len, tensor_max_len, word2id)
    return batch_sp_products, non_batch_sp_products

    for i, current_op in enumerate(gnn_outputs):
        len_sp = len(current_op)
        current_sp_mask = batch_sp_masks[i].view(-1).tolist()
        seg_id = [int(t) for t in current_sp_mask].index(1)
        if seg_id+len_sp >= limit:
            #print(i, seg_id, len_sp, non_sp_batch_data[i][ seg_id :  ].shape, gnn_outputs[i].shape)
            non_sp_batch_data[i][ seg_id: ] = gnn_outputs[i]
        else:
            #print(i, seg_id, len_sp, non_sp_batch_data[i][ seg_id : seg_id+len_sp ].shape, gnn_outputs[i].shape)
            non_sp_batch_data[i][ seg_id: seg_id+len_sp ] = gnn_outputs[i]           
    return non_sp_batch_data
    
    




