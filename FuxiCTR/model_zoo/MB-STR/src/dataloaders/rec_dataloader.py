

from .base import AbstractDataloader

import torch
import numpy as np
import torch.utils.data as data_utils

class RecDataloader(AbstractDataloader):
    def __init__(
            self,
            dataset,
            seg_len,
            mask_prob,
            num_items,
            num_workers,
            val_negative_sampler_code,
            val_negative_sample_size,
            train_batch_size,
            val_batch_size,
            predict_only_target=False,
        ):
        super().__init__(dataset,
            val_negative_sampler_code,
            val_negative_sample_size)
        self.target_code = self.bmap.get('buy') if self.bmap.get('buy') else self.bmap.get('pos')
        self.seg_len = seg_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_only_target = predict_only_target
    
    def get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=self.num_workers)
        return dataloader

    def _get_train_dataset(self):
        dataset = RecTrainDataset(self.train, self.train_b, self.seg_len, self.mask_prob, self.num_items, self.target_code, self.predict_only_target)
        return dataset

    def get_val_loader(self):
        dataset = self._get_eval_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.val_batch_size,
                                           shuffle=False, num_workers=self.num_workers)
        return dataloader

    def _get_eval_dataset(self):
        dataset = RecEvalDataset(self.train, self.train_b, self.val, self.val_b, self.val_num, self.seg_len, self.num_items, self.target_code, self.val_negative_samples)
        return dataset

class RecTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2b, max_len, mask_prob, num_items, target_code, predict_only_target):
        self.u2seq = u2seq
        self.u2b = u2b
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.target_code = target_code
        self.predict_only_target = predict_only_target

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        b_seq = self.u2b[user]

        tokens = []
        behaviors = []
        labels = []
        for s,b in zip(seq, b_seq):
            prob = np.random.rand()
            if prob < self.mask_prob and not self.predict_only_target:
                tokens.append(self.num_items+1)
                labels.append(s)
            elif prob < self.mask_prob and self.predict_only_target and b == self.target_code:
                tokens.append(self.num_items+1)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)
            behaviors.append(b)

        if len(tokens) <= self.max_len or np.random.rand()<0.8:
        # if len(tokens) <= self.max_len:
        # if True:
            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]
            behaviors = behaviors[-self.max_len:]

            padding_len = self.max_len - len(tokens)

            tokens = [0] * padding_len + tokens
            labels = [0] * padding_len + labels
            behaviors = [0] * padding_len + behaviors
        else:
            begin_idx = np.random.randint(0, len(tokens)-self.max_len+1)
            tokens = tokens[begin_idx:begin_idx+self.max_len]
            labels = labels[begin_idx:begin_idx+self.max_len]
            behaviors = behaviors[begin_idx:begin_idx+self.max_len]

        return {
            'input_ids':torch.LongTensor(tokens),
            'labels':torch.LongTensor(labels),
            'behaviors':torch.LongTensor(behaviors)
        }


class RecEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2b, u2answer, u2ab, val_num, max_len, num_items, target_code, negative_samples):
        self.u2seq = u2seq
        self.u2b = u2b
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.u2ab = u2ab
        self.val_num = val_num
        self.max_len = max_len
        self.negative_samples = negative_samples
        self.num_items = num_items
        self.target_code = target_code

    def __len__(self):
        return self.val_num

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.num_items + 1]
        seq = seq[-self.max_len:]
        seq_b = self.u2b[user] + self.u2ab[user]
        seq_b = seq_b[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        seq_b = [0] * padding_len + seq_b

        return {
            'input_ids':torch.LongTensor(seq),
            'candidates':torch.LongTensor(candidates), 
            'labels':torch.LongTensor(labels),
            'behaviors': torch.LongTensor(seq_b)
        }