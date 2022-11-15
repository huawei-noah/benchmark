

from torch import nn as nn
import torch.nn.functional as F
import torch
import math
from .utils import SublayerConnection, BehaviorSpecificPFF
from .relative_position import RelativePositionBias

class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, b_mat=None, rpb=None, W1=None, alpha1=None, W2=None, alpha2=None, mask=None):
        # 1. Calculate Q-K similarity. w. / w.o. multi-behavior dependencies
        if b_mat is not None:
            W1_ = torch.einsum('Bhmn,CBh->Chmn', W1, F.softmax(alpha1, 1))
            att_all = torch.einsum('bhim,Chmn,bhjn->bhijC', query, W1_, key)
            h=W1.size(1)
            scores = att_all.gather(4, b_mat[:,None,:,:,None].repeat(1,h,1,1,1)).squeeze(4) \
                / math.sqrt(query.size(-1)) + rpb
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(query.size(-1)) + rpb

        # 2. dealing with padding and softmax.
        if mask is not None:
            assert len(mask.shape) == 2
            mask = (mask[:,:,None] & mask[:,None,:]).unsqueeze(1)
            if scores.dtype == torch.float16:
                scores = scores.masked_fill(mask == 0, -65500)
            else:
                scores = scores.masked_fill(mask == 0, -1e30)
        p_attn = self.dropout(nn.functional.softmax(scores, dim=-1))

        # 3. information agregation. w./w.o. multi-behavior dependencies
        if b_mat is not None:
            h=W2.size(1)
            one_hot_b_mat = F.one_hot(b_mat[:,None,:,:], num_classes=alpha2.size(0)).repeat(1,h,1,1,1)
            W2_ = torch.einsum('BhdD,CBh->ChdD', W2, F.softmax(alpha2, 1))
            return torch.einsum('bhij, bhijC, ChdD, bhjd -> bhiD', p_attn, one_hot_b_mat, W2_, value)
            # return torch.matmul(p_attn, value)
        else:
            return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, n_b, battn, brpb,  d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.n_b = n_b
        self.battn = battn
        self.brpb = brpb
        
        if battn and n_b > 1: # behavior-specific mutual attention
            self.W1 = nn.Parameter(torch.randn(self.n_b, self.h, self.d_k, self.d_k))
            self.alpha1 = nn.Parameter(torch.randn(self.n_b * self.n_b + 1, self.n_b, self.h))
            self.W2 = nn.Parameter(torch.randn(self.n_b, self.h, self.d_k, self.d_k))
            self.alpha2 = nn.Parameter(torch.randn(self.n_b * self.n_b + 1, self.n_b, self.h))
            self.linear_layers = nn.Parameter(torch.randn(3, self.n_b+1, d_model, self.h, self.d_k))
        else:
            self.W1 = None
            self.W2 = None
            self.alpha1, self.alpha2 = None, None
            self.linear_layers = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        self.linear_layers.data.normal_(mean=0.0, std=0.02)

        if self.brpb:
            self.rpb = nn.ModuleList([RelativePositionBias(32,40,self.h) for i in range(self.n_b * self.n_b + 1)])
        self.attention = Attention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, b_seq=None, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        b_mat = ((b_seq[:,:,None]-1)*self.n_b + b_seq[:,None,:]) * (b_seq[:,:,None]*b_seq[:,None,:]!=0)
        # 0. rel pos bias
        if self.brpb:
            rel_pos_bias = torch.stack([layer(seq_len, seq_len) for layer in self.rpb], -1).repeat(batch_size,1,1,1,1)
            rel_pos_bias = rel_pos_bias.gather(4, b_mat[:,None,:,:,None].repeat(1,self.h,1,1,1)).squeeze(4)
        else:
            rel_pos_bias = 0
        
        if self.battn and self.n_b>1: # behavior-specific mutual attention
            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [torch.einsum("bnd, Bdhk, bnB->bhnk", x, self.linear_layers[l], F.one_hot(b_seq,num_classes=self.n_b+1).float())
                             for l, x in zip(range(3), (query, key, value))]
        else:
            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers[l])
                             for l, x in zip(range(3), (query, key, value))]
            b_mat = None

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, b_mat=b_mat, rpb=rel_pos_bias, W1=self.W1, alpha1=self.alpha1, W2=self.W2, alpha2=self.alpha2, mask=mask)

        # 3) "Concat" using a view.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, n_b, battn, bpff, brpb, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param n_b: number of behaviors
        :param battn: use multi-behavior cross attention
        :param bpff: use behavior-specific multi-gated mixture of experts
        :param brpb: use behavior-specific relative position bias
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, n_b=n_b, battn=battn, brpb=brpb, d_model=hidden, dropout=dropout)
        self.feed_forward = BehaviorSpecificPFF(d_model=hidden, d_ff=feed_forward_hidden, n_b=n_b, bpff=bpff, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b_seq, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, b_seq, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, b_seq))
        return self.dropout(x)