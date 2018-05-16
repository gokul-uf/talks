from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle as pkl
import scipy.io as sio
import numpy as np


class SVHN_Processor(object):
    def __init__(self, data_dir, batch_size=100):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 10

    def _one_hot_encode(self, val):
        one_hot = [0] * self.num_classes
        if val == 10:
            one_hot[0] = 1
        else:
            one_hot[val[0] - 1] = 1
        return one_hot

    def get_train_batch(self):
        file = self.data_dir + "/train_32x32.mat"
        data = sio.loadmat(file)

        labels = data['y']
        labels = np.asarray([self._one_hot_encode(label) for label in labels])

        data = data['X']
        data = data.transpose(3, 0, 1, 2)

        data = data.astype(np.float32)
        data = (data - 127.0) / 127.0  # Not 127, as (255 / 127) > 1

        assert np.min(data) >= -1, "min is {}".format(np.min(data))
        assert np.max(data) <= 255 / 127, "max is {}".format(np.max(data))

        while True:
            for i in range(0, len(data), self.batch_size):
                yield data[i:i + self.batch_size], labels[i:
                                                          i + self.batch_size]

    def get_val_batch(self):
        file = self.data_dir + "/test_32x32.mat"
        data = sio.loadmat(file)

        labels = data['y']
        labels = np.asarray([self._one_hot_encode(label) for label in labels])

        data = data['X']
        data = data.transpose(3, 0, 1, 2)

        data = data.astype(np.float32)
        data = (data - 127.0) / 127.0  # Not 127, as (255 / 127) > 1
        self.test_size = len(data)

        assert np.min(data) >= -1, "min is {}".format(np.min(data))
        assert np.max(data) <= 255 / 127, "max is {}".format(np.max(data))

        for i in range(0, len(data), self.batch_size):
            yield data[i:i + self.batch_size], labels[i:i + self.batch_size]
