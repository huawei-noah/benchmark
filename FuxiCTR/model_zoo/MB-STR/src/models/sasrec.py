# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn as nn
import pytorch_lightning as pl
from .embedding import BERTEmbedding
from .transformer import TransformerBlock



class SAS(pl.LightningModule):
    def __init__(self,
        max_len,
        num_items,
        n_layer,
        n_head,
        d_model,
        dropout
    ):
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        
        vocab_size = num_items + 1
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, max_len=max_len, dropout=dropout)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_model * 4, dropout) for _ in range(n_layer)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        tl = x.shape[1] # time dim len for enforce causality
        mask *= torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x
