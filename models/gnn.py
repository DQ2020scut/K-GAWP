import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import sys
sys.path.append("..")
from models.gnn_preprocess import *
import utils

class SpKBGATModified(nn.Module):
    def __init__(self, word2id, id2word, rel2id, all_term2id, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim, nheads, **kwargs):
        super().__init__()
        self.CUDA = kwargs['cuda']
        self.word2id = word2id
        self.id2word = id2word
        self.rel2id = rel2id
        self.all_term2id = all_term2id
        self.num_nodes = len(all_term2id)
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.relation_dim = initial_relation_emb.shape[1]
        self.entity_out_dim = entity_out_dim

        self.ent_embedding = nn.Embedding(initial_entity_emb.shape[0], self.entity_in_dim)
        self.ent_embedding.weight.data.copy_(initial_entity_emb)
        term_embedding_mat = torch.zeros((self.num_nodes, self.entity_in_dim))
        for i, term in enumerate(all_term2id):
            words = torch.LongTensor([self.word2id[w.strip()] for w in term.split(' ')])
            temp_embedding = self.ent_embedding(words)
            temp_embedding = torch.mean(temp_embedding, dim=0)
            term_embedding_mat[i] = temp_embedding # term_embedding[0] is dummy task
            
        self.term_embedding = nn.Parameter(term_embedding_mat) # important nn.embedding to be fed into GAT
        self.relation_embeddings = nn.Parameter(initial_relation_emb) # relation embeddings are randomly generated in the GNN class initiation
        
        self.original_term_embeddings = deepcopy(term_embedding_mat.data)
        
        self.sparse_gat = GAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim, self.relation_dim, nheads, kwargs['alpha'], kwargs['drop'])
        self.convKB = ConvKB(int(self.entity_out_dim/2), 3, 1, kwargs['cnn_out_ch'], self.entity_out_dim, kwargs['drop'], kwargs['alpha'])        
        
        self.W_E = nn.Parameter(torch.zeros(size=(self.entity_in_dim, entity_out_dim * nheads)))
        self.OUT_E = nn.Parameter(torch.zeros(size=( entity_out_dim*nheads, int(entity_out_dim*1/2) )))
        nn.init.xavier_uniform_(self.W_E.data, gain=1.414)
        nn.init.xavier_uniform_(self.OUT_E.data, gain=1.414)
        #self.init_W_E = deepcopy(self.W_E) # 记录初始化参数

    def forward(self, batch_sp_triples, train_adj_matrix, train_indices_nhop, use_avg=False):
        edge_list = train_adj_matrix[0] # rows and cols (a pair of nodes in the 1st neighbourhood)
        edge_type = train_adj_matrix[1] # rel_data, i.e., rel_ids 这个记录了各点1阶邻域内相关联点的关系 对应edge_list中的每一组点
        
        edge_embed = self.relation_embeddings[edge_type] 
        # 是2阶邻域 所以[:,3]代表获取外层终点[:,0]代表获取中心点
        edge_list_nhop = torch.cat([train_indices_nhop[:,3].unsqueeze(-1), train_indices_nhop[:,0].unsqueeze(-1)], dim=1).t()
        # 这个edge type 是获取的边的类型
        edge_type_nhop = torch.cat([train_indices_nhop[:,1].unsqueeze(-1), train_indices_nhop[:,2].unsqueeze(-1)], dim=1)   
        
        # 把 entity_embedding的数据沿着列的方向进行L2归一化
        #self.term_embedding.weight = F.normalize(self.term_embedding.data, p=2, dim=1).detach()
        #self.relation_embeddings.data = F.normalize(self.relation_embeddings.data, p=2, dim=1).detach()
        
        if self.CUDA:
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_embed = edge_embed.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()
            self.W_E = nn.Parameter(self.W_E.cuda())
            self.OUT_E = nn.Parameter(self.OUT_E.cuda())
            self.original_term_embeddings = self.original_term_embeddings.cuda()
            self.term_embedding = self.term_embedding.cuda()
            self.relation_embeddings = self.relation_embeddings.cuda()
            self.cuda()
            
        # out_entity和out_relation就是最后一层GAT之后的embeddings 后面就是加上origional embedding了
        self.term_embedding.requires_grad = False
        out_entity, out_relation = self.sparse_gat(self.term_embedding, self.relation_embeddings, edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop, self.CUDA, use_avg=False)
    
        # mask的作用是有些没有出现在当前training batch的entity我们不会选择更新 于是通过mask将其zeroing out    
        mask_indices = []
        for sp_triples in batch_sp_triples:
            mask_indices.append(sp_triples[0])
            mask_indices.append(sp_triples[2])    
        mask_indices = torch.unique(torch.LongTensor(mask_indices))
        mask = torch.zeros(self.term_embedding.shape[0])
        mask[mask_indices] = 1.0 # 只有在当前batch 中被用到的node的idx被设为1 其他是0
        entity_restore = self.original_term_embeddings.mm(self.W_E)
        if self.CUDA:
            entity_restore = entity_restore.cuda()
            mask = mask.cuda()
       
        out_entity = entity_restore + (mask.unsqueeze(-1).expand_as(out_entity)) * out_entity  # 考虑batch inputs的mask 只更新部分nodes
        #out_entity = mask.unsqueeze(-1).expand_as(out_entity)*out_entity
        out_entity = out_entity.mm(self.OUT_E)
        out_relation = out_relation.mm(self.OUT_E)
        #out_entity = F.normalize(out_entity, p=2, dim=1)
        #out_relation = F.normalize(out_relation, p=2, dim=1)
        return out_entity, out_relation
    
class GAT(nn.Module):
    def __init__(self, num_nodes, entity_in_dim, nhid, relation_dim, nheads, alpha, dropout):
        # nhid <- entity_out_dim 
        # relation_dim <- relation_in_dim
        super(GAT, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        
        #这里定义一个attention层 包含multi-head attention
        self.attentions = []
        for i in range(nheads):
            temp_att = GATlayer(num_nodes, entity_in_dim, nhid, relation_dim, dropout, alpha)
            self.attentions.append(temp_att)
            setattr(self, 'att_' + str(i), temp_att) # 如果不想最底层的att接收反向传播这里可以不用setattr

        self.WR = nn.Parameter(torch.zeros((relation_dim, nheads*nhid)))
        nn.init.xavier_uniform_(self.WR.data, gain=1.414)
        # 注意out_att和att构建对维度不同
        self.out_att = GATlayer(num_nodes, nhid*nheads, nheads*nhid, nheads*nhid, dropout, alpha, out_layer=True)
         # 记录初始化参数 self.init_WR = deepcopy(self.WR)

    def forward(self, term_embeddings, relation_embeddings, edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop, CUDA, use_avg=False):
        # edge_list edge_list_nhop, edge_type, edge_type_nhop说明见上面一个cell
        x = term_embeddings
        # relation embedding的加合 文章提到的 sum of the path edge embeddings
        edge_embed_nhop = relation_embeddings[edge_type_nhop[:,0]] + relation_embeddings[edge_type_nhop[:,1]]
        
        # 下面这行代码又会调用attention layer这个类里的forward()
        temp_res_list = [] # recall edge_lst 1阶邻域node(source-target) edge_lst_nhop(多阶邻域nodes)
        for i, att in enumerate(self.attentions):  # edge_embed是1阶邻域relation的embedding 
            temp_res = att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, CUDA)
            temp_res_list.append(temp_res)

        layer_x = torch.cat(temp_res_list, dim=1)
        #layer_x = self.dropout_layer(layer_x)
        # 一个attention layer完成后对 relation embedding也进行变换
        out_relation = relation_embeddings.mm(self.WR)
        
        # 在前一层新得到的relation embedding中重新进行一次lookup和长路径的embedding求和
        edge_embed = out_relation[edge_type]
        edge_embed_nhop = out_relation[edge_type_nhop[:,0]] + out_relation[edge_type_nhop[:,1]]
        
        # 注意原模型里只做了一次attention layer 所以下面就直接到final layer了 即第1层attention layer也是最后一层
        # 进行最后一次attention layer 下面是在out_att 返回平均值的代码
        layer_x = self.out_att(layer_x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, CUDA)
        
        if use_avg==True:
            layer_x_for_avg = torch.split(layer_x, self.nihd, 1) # 在维度1 每隔nhid个拆分 split()返回一个tuple
            x_total = torch.stack(layer_x_for_avg)
            layer_x_avg = torch.mean(x_total, dim=0) 
            layer_x_avg = F.elu(layer_x_avg) # 普通layer是在每个att里面做了relu out_att是在这里做relu 但是都做了relu
            return layer_x_avg, out_relation
        else:
            out_entity = F.elu(layer_x)
            return out_entity, out_relation
        
class GATlayer(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, rela_dim, dropout, alpha, out_layer=False):
        # in_features <- entity_in_dim <- entity_in_dim
        # out_features <- nhid <- entity_out_dim
        super(GATlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.out_layer = out_layer
        # 下面是2个entity1个relation拼接成长向量
        self.w1 = nn.Parameter(torch.zeros((out_features, 2*in_features + rela_dim)))
        self.w2 = nn.Parameter(torch.zeros(1, out_features))
        nn.init.xavier_normal_(self.w1.data, gain=1.414)        
        nn.init.xavier_normal_(self.w2.data, gain=1.414)

        self.dropout_layer = nn.Dropout(dropout)
        self.leakyRelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()
        # 记录初始化参数 self.init_w1 = deepcopy(self.w1) self.init_w2 = deepcopy(self.w2)
            
    def forward(self, x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, CUDA):
        # x <- entity_embeddings
        # edge_embed <- edge_embed (已经对nhop path进行了加合操作的 edge embedding)
        # edge_list_nhop, edge_embed_nhop 就是对应前面得到的值
        N = x.shape[0] # 就是graph里unique nodes数量 是所有entity的embedding 用在了e_row_sum
        
        # 下面的edge是把 edge_list和edge_list_nhop 拼接在一起 得到的是一个(2, xxx)的tensor 这个tensor上下点在n阶邻域内相连
        edge = torch.cat((edge_list[:, :], edge_list_nhop[:,:]), dim=1)
        # 同理把所有的edge (relation) 的embedding拼接在一起 (num_1_hop+...+num_n_hop, relation_embed_dim)
        edge_embed_total = torch.cat((edge_embed[:, :], edge_embed_nhop[:,:]), dim=0)
       
        # ***attention计算的第一步 把source entity, target entity和 relation的embedding都拼接起来 这里是包括了1，2..n阶邻域的triple
        # 超过1阶邻域的triple之间的关系用多次关系之和代替(体现在edge_embed_nhop里)
        # 注意如果这里是最后一层att out_att的话 x的维度会变大 因为是拼接了multi-heads之后的layer_x
        edge_h = torch.cat([x[edge[0,:], :], x[edge[1,:], :], edge_embed_total[: , :]], dim=1).t()
        edge_m = self.w1.mm(edge_h) # 这个就是cijk
        
        # ***attention计算的第二步 计算绝对 attention values
        powers = -self.leakyRelu(self.w2.mm(edge_m).squeeze()) # 这个就是 bijk 使用squeeze()降低维度 不给出dim即把所有维度=1则的维度去掉
        store_alpha = F.softmax(powers, dim=-1)
        
        # ***attention计算的第三步 计算相对 attention values
        edge_e = torch.exp(powers).unsqueeze(1) # edge_e对应 exp(bijk) unsqueeze即在指定的index位置插入一个额外维度
        assert not torch.isnan(edge_e).any() # 检查 不能有na值

        e_row_sum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1, CUDA) #从代码来看经过这一步之后就已经得到了相对attention值
        e_row_sum[e_row_sum<1e-12]=1e-12 # 消除0的影响
    
        edge_e = edge_e.squeeze(1) # 使用squeeze()降低维度 给定dim即在指定的维度上如果这个维度=1则把该维度去掉
        #edge_e = self.dropout_layer(edge_e)

        # ***attention计算的第四步 计算h
        edge_w = (edge_e * edge_m).t() #对应exp(bijk) * cijk
        
        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features, CUDA)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_row_sum)
        assert not torch.isnan(h_prime).any()
                
        if self.out_layer:
            layer_x = F.elu(h_prime)
        else:
            layer_x = h_prime
        return layer_x

class SpecialSpmmFunctionFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features, CUDA):
        if CUDA:
            a = torch.sparse_coo_tensor(edge, edge_w, torch.Size([N, N, out_features])).cuda()
        else:
            a = torch.sparse_coo_tensor(edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]
        return b.to_dense()
    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
        #print(grad_values.device, ctx.device, edge_sources.device)
        return None, grad_values, None, None, None, None

class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features, CUDA):
        # 因为定义了staticmethod所以可以直接通过类名称调用
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features, CUDA)

class ConvKB(nn.Module):    
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, entity_out_dim, drop_prob, alpha_leaky):
        super().__init__()

        self.bn_1 = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len)) # for a triple, in_channel=1, seq_len=3 by default
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.LeakyReLU(alpha_leaky)
        self.fc_layer = nn.Linear((input_dim) * out_channels, int(entity_out_dim/2))
        
        nn.init.xavier_uniform_(self.fc_layer.weight) 
        nn.init.xavier_uniform_(self.conv_layer.weight)

    def forward(self, conv_input, args):
        size = conv_input.shape[0]
        dim = conv_input.shape[-1]
        conv_input = conv_input.reshape((size, 1, dim, 3))
        #print('\tconv_input after transpose in convKB:', conv_input.shape) # (batch_size, 1, n_heads*out_embed, 3)
        
        #out_conv = self.bn_1(conv_input)
        out_conv = self.conv_layer(conv_input)
        #out_conv = self.bn_2(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.squeeze(-1).view(size, -1)
        #out_conv = self.dropout(out_conv)
        
        cnn_outputs = self.fc_layer(out_conv)
        lag = args.max_len - size
        add_tensor = torch.zeros((lag, dim))
        if args.cuda:
            add_tensor = add_tensor.cuda()
        cnn_outputs = torch.cat((cnn_outputs, add_tensor))
        return cnn_outputs
    
 
        