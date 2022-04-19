# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from time import gmtime, strftime
import argparse
import pandas as pd
import numpy as np
import os

from src.data_loader import MindDataset
from src.model import UNBERT
from src.eval import dev, test

class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: str = False,
        num_workers: int = 0
    ) -> None:
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = dataset.collate
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--split', type=str, default='small')
    parser.add_argument('--pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('--level_state', type=str, default='word',
                        help='word, news or both')
    parser.add_argument('--news_mode', type=str, default='nseg',
                        help='nseg, mean or attention')
    parser.add_argument('--news_max_len', type=int, default=20)
    parser.add_argument('--hist_max_len', type=int, default=20)
    parser.add_argument('--seq_max_len', type=int, default=300)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--epoch', type=int, default=5) # always set 5 in small dataset, 2 in large
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_every', type=int, default=10000)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    log_file = os.path.join(args.output, "{}-{}-{}.log".format(
                    args.mode, args.split, strftime('%Y%m%d%H%M%S', gmtime())))
    os.makedirs(args.output, exist_ok=True)
    def printzzz(log):
        with open(log_file, "a") as fout:
            fout.write(log + "\n")
        print(log)

    printzzz(str(args))
    model = UNBERT(pretrained=args.pretrain, 
                 level_sate=args.level_state,
                 news_mode=args.news_mode,
                 max_len=args.seq_max_len)
    if args.restore is not None and os.path.isfile(args.restore):
        printzzz("restore model from {}".format(args.restore))
        state_dict = torch.load(args.restore, map_location=torch.device('cpu'))
        st = {}
        for k in state_dict:
            if k.startswith('bert'):
                st['_model'+k[len('bert'):]] = state_dict[k]
            elif k.startswith('classifier'):
                st['_dense'+k[len('classifier'):]] = state_dict[k]
            else:
                st[k] = state_dict[k]
        model.load_state_dict(st)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    if args.mode == "train":
        printzzz('reading training data...')
        train_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='train',
            split=args.split,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            seq_max_len=args.seq_max_len
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
        )
        printzzz('reading dev data...')
        dev_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='dev',
            split=args.split,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            seq_max_len=args.seq_max_len
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )

        loss_fn = nn.CrossEntropyLoss()
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        m_scheduler = get_linear_schedule_with_warmup(m_optim, 
                    num_warmup_steps=len(train_set)//args.batch_size*2, 
                    num_training_steps=len(train_set)*args.epoch//args.batch_size)
        loss_fn.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            loss_fn = nn.DataParallel(loss_fn)
        printzzz("start training...")

        best_auc = 0.0
        for epoch in range(args.epoch):
            avg_loss = 0.0
            batch_iterator = tqdm(train_loader, disable=False)
            for step, train_batch in enumerate(batch_iterator):
                batch_score = model(train_batch['input_ids'].to(device), 
                                    train_batch['input_mask'].to(device), 
                                    train_batch['segment_ids'].to(device),
                                    train_batch['news_segment_ids'].to(device),
                                    train_batch['sentence_ids'].to(device),
                                    train_batch['sentence_mask'].to(device),
                                    train_batch['sentence_segment_ids'].to(device),
                                    )
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
                if torch.cuda.device_count() > 1:
                    batch_loss = batch_loss.mean()
                avg_loss += batch_loss.item()
                batch_loss.backward()
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()
            auc = dev(model, dev_loader, device, args.output, is_epoch=True)
            printzzz("Epoch {}, AUC: {:.4f}".format(epoch+1, auc))
            final_path = os.path.join(args.output, "epoch_{}.bin".format(epoch+1))
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), final_path)
            else:
                torch.save(model.state_dict(), final_path)
        printzzz("train success!")
    elif args.mode == "dev":
        printzzz('reading dev data...')
        dev_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='dev',
            split=args.split,
            news_max_len=args.news_max_len,
            hist_max_len=args.hist_max_len,
            seq_max_len=args.seq_max_len
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        auc = dev(model, dev_loader, device, args.output, is_epoch=True)
        printzzz("dev AUC: {:.4f}".format(auc))
        printzzz("dev success!")
    else:
        printzzz('reading test data...')
        test_set = MindDataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            part_tag=args.part_tag
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        test(model, test_loader, device, args.output)
        printzzz("test success!")

if __name__ == "__main__":
    main()
