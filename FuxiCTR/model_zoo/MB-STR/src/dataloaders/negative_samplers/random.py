

from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(1, self.user_count+1):
            if user not in self.val.keys():
                continue
            seen = set(self.train[user])
            seen.update(self.val[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

# class RandomNegativeSamplerTrain(AbstractNegativeSampler):
#     @classmethod
#     def code(cls):
#         return 'random_train'

#     def generate_negative_samples(self):
#         negative_samples = {}
#         print('Sampling negative items')
#         for user in trange(1, self.user_count+1):
#             seen = set(self.train[user])
#             if user in self.val.keys():
#                 seen.update(self.val[user])
#             samples = []
#             for _ in range(self.sample_size):
#                 item = np.random.choice(self.item_count) + 1
#                 while item in seen or item in samples:
#                     item = np.random.choice(self.item_count) + 1
#                 samples.append(item)

#             negative_samples[user] = samples

#         return negative_samples