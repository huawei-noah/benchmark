
import torch
from torch import nn as nn
import pytorch_lightning as pl
from .embedding import BERTEmbedding, SimpleEmbedding
from .new_transformer import TransformerBlock


class BERT(pl.LightningModule):
    def __init__(self,
        max_len: int = None,
        num_items: int = None,
        n_layer: int = None,
        n_head: int = None,
        n_b: int = None,
        d_model: int = None,
        dropout: float = .0,
        battn: bool = None,
        bpff: bool = None,
        brpb: bool = None, 
    ):
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        self.n_b = n_b
        self.battn = battn
        self.bpff = bpff
        self.brpb = brpb
        
        vocab_size = num_items + 1 + n_b # add padding and mask 
        # if self.brpb:
        if True:
            # simple embedding, adding behavioral relative positional bias in transformer blocks
            self.embedding = SimpleEmbedding(vocab_size=vocab_size, embed_size=d_model, dropout=dropout)
        else:
            # embedding for BERT, sum of positional, token embeddings
            self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, max_len=max_len, dropout=dropout)
        # multi-layers transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_model * 4, n_b, battn, bpff, brpb, dropout) for _ in range(n_layer)])

    def forward(self, x, b_seq):
        # get padding masks
        mask = (x > 0)
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, b_seq, mask)
        return x