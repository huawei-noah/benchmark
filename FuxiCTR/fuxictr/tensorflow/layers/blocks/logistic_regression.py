# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import tensorflow as tf
from tensorflow.keras.layers import Layer
from fuxictr.tensorflow.layers import FeatureEmbedding


class LogisticRegression(Layer):
    def __init__(self, feature_map, use_bias=True, regularizer=None):
        super(LogisticRegression, self).__init__()
        self.bias = tf.Variable(tf.zeros(1)) if use_bias else None
        self.embedding_layer = FeatureEmbedding(feature_map, 1, use_pretrain=False, 
                                                use_sharing=False,
                                                embedding_regularizer=regularizer,
                                                name_prefix="lr_")

    def call(self, X):
        embed_weights = self.embedding_layer(X)
        output = tf.reduce_sum(embed_weights, axis=1)
        if self.bias is not None:
            output += self.bias
        return output

