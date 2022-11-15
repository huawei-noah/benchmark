
from .random import RandomNegativeSampler


NEGATIVE_SAMPLERS = {
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, train, val, user_count, item_count, sample_size, save_folder):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, user_count, item_count, sample_size, save_folder)
