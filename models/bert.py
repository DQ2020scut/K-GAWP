# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertConfig
import os
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    
class Config(object):
    def __init__(self, args):
        self.model_name = 'bert'
        self.require_improvement = 1000                                 # terminate the training
        self.bert_path = 'bert_pretrain'
        self.out_size = args.out_size
        self.input_dim = args.embed_dim
        self.use_gnn = args.use_gnn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # BertModel是继承了modelling.py中的PreTrainedBertModel
        # 后者中定义了一个 from_trained()函数 是classmethod 因此可以直接用类名访问
        config_file = os.path.join(os.getcwd(), './models/bert_config.json')
        config_instance = BertConfig.from_json_file(config_file) # bert model configuration
        self.bert = BertModel(config_instance)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        if config.use_gnn==True:
            self.fc_in = nn.Linear(config.input_dim + int(config.input_dim/2), config_instance.hidden_size)
        else:
            self.fc_in = nn.Linear(config.input_dim, config_instance.hidden_size)
        self.fc_out = nn.Linear(config_instance.hidden_size, config.out_size)
        self.fc_in.requires_grad = True
        self.fc_out.requires_grad = True
       
    def forward(self, data, batch_masks=None):
        if isinstance(data, tuple):
            essen_data = data[0]
            gnn_spatial_data = data[1]
            data = torch.cat((essen_data, gnn_spatial_data), dim=-1)
        if batch_masks==None:
            batch_masks = torch.ones(data.shape[0], 1, 1, data.shape[1], device=device)
        data = self.fc_in(data) # keep dimension consistency for bert QKV
        _, pooled = self.bert(data, attention_mask=batch_masks, output_all_encoded_layers=False)
        out = self.fc_out(pooled)
        return out, self
