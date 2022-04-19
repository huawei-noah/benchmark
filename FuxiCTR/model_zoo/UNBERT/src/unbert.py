# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2021. The Chinese University of Hong Kong. All rights reserved.
#
# Authors: Qi Zhangqi <Huawei Noah's Ark Lab>
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
# =========================================================================

import logging
import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_bert import BertEncoder, BertPooler, BertPreTrainedModel


BertLayerNorm = torch.nn.LayerNorm


class UNBertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # add segment embeddings
        self.segment_embeddings = nn.Embedding(64, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, news_segment_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        if news_segment_ids is not None:
            segment_embeddings = self.segment_embeddings(news_segment_ids)
            embeddings += segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UNBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = UNBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.encoder_news = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        news_segment_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        sentence_input=None,
        news_mode='nseg',
        att_mapping=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, 
            position_ids=position_ids, 
            token_type_ids=token_type_ids, 
            inputs_embeds=inputs_embeds,
            news_segment_ids=news_segment_ids,
        )
        
        # word-level matching
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        sentence_ids, sentence_mask, sentence_segment = sentence_input

        def reduce_mean(embed, sentence_id, weight=None):
            tmp = torch.zeros_like(sentence_id)
            tmp[:-1] = sentence_id[1:]
            size = tmp - sentence_id
            tmp = size.clone()
            tmp[tmp < 0] = 0
            size[size < 0] = embed.size(0) - tmp.sum()
            ind = torch.arange(len(size), device=device).repeat_interleave(size)
            y = torch.zeros((len(size), embed.size(-1)), dtype=torch.float, device=device)
            if weight is None:
                news_embed = y.index_add_(0, ind, embed) / (size.float().unsqueeze(-1) + 1e-6)
            else:
                news_embed = y.index_add_(0, ind, embed) 
                w = torch.zeros((len(size), 1), dtype=torch.float, device=device)
                news_embed /= (w.index_add_(0, ind, weight) + 1e-6)
            return news_embed.unsqueeze(0)
        
        # aggregate word to news
        if news_mode == 'mean':
            batch_data = [reduce_mean(embed, sentence_id) 
                                for embed, sentence_id in zip(sequence_output, sentence_ids)]
            sentence_embedding = torch.cat(batch_data, 0)
        elif news_mode == 'attention':
            assert att_mapping is not None
            embed_flat = torch.reshape(sequence_output, 
                                (-1, sequence_output.size(1) * sequence_output.size(2)))
            embed_flat = embed_flat.clone()
            weights = att_mapping(embed_flat).unsqueeze(-1)
            sequence_output *= weights
            batch_data = [reduce_mean(embed, sentence_id, weight)
                        for embed, sentence_id, weight in zip(sequence_output, sentence_ids, weights)]
            sentence_embedding = torch.cat(batch_data, 0)
        elif news_mode == 'nseg': 
            dummy = sentence_ids.unsqueeze(2).expand(sentence_ids.size(0), 
                                                     sentence_ids.size(1), 
                                                     sequence_output.size(2))
            sentence_embedding = torch.gather(sequence_output, 1, dummy)
        else:
            raise NotImplementedError(news_mode + " not implemented!")

        # news-level matching
        sentence_outputs = self.encoder_news(
            sentence_embedding, 
            attention_mask=self.get_extended_attention_mask(sentence_mask, 
                sentence_ids.size(), sentence_ids.device),
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        sentence_output = sentence_outputs[0]

        # sequence_output, sentence_output, pooled_output, (hidden_states), (attentions)
        outputs = (sequence_output, sentence_output, pooled_output,) + encoder_outputs[1:]  
        return outputs  
