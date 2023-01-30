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


import h5py
import os
import logging
import numpy as np
import gc
import multiprocessing as mp


def save_h5(darray_dict, data_path):
    logging.info("Saving data to h5: " + data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with h5py.File(data_path, 'w') as hf:
        for key, arr in darray_dict.items():
            hf.create_dataset(key, data=arr)


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0, 
                     test_size=0, split_type="sequential"):
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def transform_h5(feature_encoder, ddf, filename, preprocess=False, block_size=0):
    def _transform_block(feature_encoder, df_block, filename, preprocess):
        if preprocess:
            df_block = feature_encoder.preprocess(df_block)
        darray_dict = feature_encoder.transform(df_block)
        save_h5(darray_dict, os.path.join(feature_encoder.data_dir, filename))

    if block_size > 0:
        pool = mp.Pool(mp.cpu_count() // 2)
        block_id = 0
        for idx in range(0, len(ddf), block_size):
            df_block = ddf[idx: (idx + block_size)]
            pool.apply_async(_transform_block, args=(feature_encoder,
                                                     df_block,
                                                     '{}/part_{}.h5'.format(filename, block_id),
                                                     preprocess))
            block_id += 1
        pool.close()
        pool.join()
    else:
        _transform_block(feature_encoder, ddf, filename + ".h5", preprocess)


def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None, valid_size=0, 
                  test_size=0, split_type="sequential", **kwargs):
    """ Build feature_map and transform h5 data """
    
    # Load csv data
    train_ddf = feature_encoder.read_csv(train_data, **kwargs)
    valid_ddf = None
    test_ddf = None

    # Split data for train/validation/test
    if valid_size > 0 or test_size > 0:
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
        train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
                                                          valid_size, test_size, split_type)
    
    # fit and transform train_ddf
    train_ddf = feature_encoder.preprocess(train_ddf)
    feature_encoder.fit(train_ddf, **kwargs)
    block_size = int(kwargs.get("data_block_size", 0)) # Num of samples in a data block
    transform_h5(feature_encoder, train_ddf, 'train', preprocess=False, block_size=block_size)
    del train_ddf
    gc.collect()

    # Transfrom valid_ddf
    if valid_ddf is None and (valid_data is not None):
        valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
    if valid_ddf is not None:
        transform_h5(feature_encoder, valid_ddf, 'valid', preprocess=True, block_size=block_size)
        del valid_ddf
        gc.collect()

    # Transfrom test_ddf
    if test_ddf is None and (test_data is not None):
        test_ddf = feature_encoder.read_csv(test_data, **kwargs)
    if test_ddf is not None:
        transform_h5(feature_encoder, test_ddf, 'test', preprocess=True, block_size=block_size)
        del test_ddf
        gc.collect()
    logging.info("Transform csv data to h5 done.")
    
    # Return processed data splits
    if block_size > 0:
        return os.path.join(feature_encoder.data_dir, "train/*.h5"), \
               os.path.join(feature_encoder.data_dir, "valid/*.h5"), \
               os.path.join(feature_encoder.data_dir, "test/*.h5")
    else:
        return os.path.join(feature_encoder.data_dir, 'train.h5'), \
               os.path.join(feature_encoder.data_dir, 'valid.h5'), \
               os.path.join(feature_encoder.data_dir, 'test.h5')

