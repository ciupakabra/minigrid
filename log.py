import numpy as np
import tensorflow as tf
from collections.abc import Iterable

from collections import Counter


class Logger():

    def __init__(self):
        self.rewards = []
        self.batch_means = []
        self.temp_means = []

    def update(self, rewards):
        if isinstance(rewards, Iterable):
            self.rewards.extend(rewards)
        else:
            self.rewards.append(rewards)
        self.batch_means.append(np.mean(rewards))
        self.temp_means.append(np.mean(self.rewards))
