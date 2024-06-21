from collections import namedtuple


NormStats = namedtuple('NormStats',
                       ('reward_mean', 'reward_std', 'feature_mean', 'feature_std'))
