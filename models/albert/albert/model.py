import torch
import torch.nn as nn
import torch.nn.functional as F

from .nystrom import NystromAttention
from .attention import SoftmaxAttention
from .helper import HelperModule

"""
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEmbedding(HelperModule):
    def build(self, nb_in:int, dropout: float = 0.0, max_length: int = 5000):
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, nb_in)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-torch.log(torch.tensor(10000.0)) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.FloatTensor):
        x = x.view((self.pe.shape[0], -1, self.pe.shape[2]))
        temp = self.pe[:x.size(0), :]
        x = x + temp
        return self.dropout(x)

"""
    Implement O(VE + EH) token embeddings from ALBERT paper
    Better than casting directly to hidden space. 
"""
class FactorizedEmbedding(HelperModule):
    def build(self,
            vocab_size:     int,
            embedding_dim:  int,
            hidden_size:    int,
            max_len:        int = 5000,
        ):
        self.ve = nn.Embedding(vocab_size, embedding_dim) # this ve is the conventional embedding looking-up
        self.eh = nn.Linear(embedding_dim, hidden_size) # here assume hidden size=intermediate size

    def forward(self, x: torch.LongTensor):
        #x = self.ve(x) # this returns bs*max_len*embedding
        res = self.eh(x) # project bs*max_len*embedding to bs*max_len*hidden_size
        return res

"""
    One single Transformer encoder layer
    Options for softmax or nystromformer attention
"""
class TransformerLayer(HelperModule):
    def build(self,
            nb_in:      int,
            mlp_dim:    int,
            seq_len:    int,
            nb_heads:   int = 8,
            head_dim:   int = 64,
            norm:       bool = True,
            dropout:    float = 0.0,
            attention_type: str = 'softmax',
        ):
        if attention_type in ['nystrom', 'nystromformer']:
            self.attn = NystromAttention(dim=nb_in, dim_head=head_dim, heads=nb_heads, out_dim=out_dim, dropout=dropout)
        elif attention_type in ['default', 'softmax']:
            self.attn = SoftmaxAttention(mlp_dim, dim_head=head_dim, heads=nb_heads, dropout=dropout) #mlp_dim=nb_in??
        else:
            raise ValueError("invalid attention type selected.")

        if norm:
            self.norm1 = nn.LayerNorm([seq_len, nb_in])
            self.norm2 = nn.LayerNorm([seq_len, nb_in])
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, nb_in),
            nn.GELU(),
            nn.Linear(nb_in, mlp_dim)
        )

    def forward(self, x, mask: torch.BoolTensor = None):
        att_res = self.attn(x, mask = mask)
        x = self.norm1(x + att_res)
        x = self.norm2(x + self.mlp(x))
        return x

"""
    Transformer class with cross-layer parameter sharing -- ie. ALBERT
    TODO: Add learned positional embeddings like huggingface version?
"""
class ALBERT(HelperModule):
    def build(self, 
            nb_in:      int, # this is actually not used
            seq_len:    int,
            mlp_dim:    int = 768,
            emb_dim:    int = 128,
            nb_heads:   int = 16,
            nb_layers:  int = 12,
            nb_seg:     int = 2,
            dropout:    float = 0.1, # in albert, dropout can potentially hurt performance at large sizes.
            norm:       bool = True,
            attention_type: str = 'softmax',
            out_dim:    int = 200
        ):
        self.nb_in = nb_in # hidden_dim
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim # the intermediate dim
        self.out_dim = out_dim
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.norm = norm
        self.head_dim = int(self.mlp_dim / self.nb_heads)
        
        # note the key of ALBERT, instead of directly using Vocab_size * hidden_size, which can be very large
        # Ablert projects Vocab_size * Embedding_size to Embedding_size*hidden_size
        self.in_emb = FactorizedEmbedding(self.nb_in, self.emb_dim, self.mlp_dim) # mlp_dim is the intermediate/ hidden dim
        self.seg_emb = nn.Embedding(nb_seg, self.mlp_dim)
        self.pos_emb = PositionalEmbedding(self.mlp_dim, max_length=self.seq_len, dropout=dropout) 

        self.drop_emb = nn.Dropout(dropout)
        if self.norm:
            self.norm_emb = nn.LayerNorm(self.mlp_dim)
        else:
            self.norm_emb = nn.Identity()
        
        self.transformer = TransformerLayer(
            nb_in=self.nb_in, 
            mlp_dim=self.mlp_dim,
            seq_len=self.seq_len, 
            nb_heads=self.nb_heads, 
            head_dim=self.head_dim, 
            norm=self.norm, 
            dropout=dropout,
            attention_type=attention_type,
        )
        self.fc_out = nn.Linear(self.mlp_dim, self.out_dim)
        self.fc_out.requires_grad = True
        self.activation = nn.Tanh()
        
    def forward(self, x:torch.LongTensor, batch_masks=None): #seg: torch.LongTensor
        bs = x.shape[0]
        x = self.in_emb(x) # + self.seg_emb(seg) # bs*max_len*embed_dim + bs*max_len*hidden_dim
        #x = self.pos_emb(x)
        #x = x.view(bs, self.seq_len, -1) # reshape x back if the pos_embed is applied
        x = self.drop_emb(self.norm_emb(x))
        for _ in range(self.nb_layers):
            x = self.transformer(x, mask=batch_masks) #batch_masks
        out = x[:, 0] # POOL get the first token tensor for [DUMMY_TASK], similar to [CLS]
        
        out = self.fc_out(out)
        out = self.activation(out)
        return out, self
