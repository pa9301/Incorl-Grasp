import json
import os
from abc import *
from multiprocessing import Manager, Lock

import numpy as np


class DataBuffer(object):

    def __init__(self, root_path, capacity):
        self._root_path = root_path
        self._capacity = capacity

        self._lock = Lock()
        self._data_list = Manager().list()

    @abstractmethod
    def store(self, transition):
        pass

    @abstractmethod
    def prior_sampling(self, n):
        pass

    @abstractmethod
    def uniform_sampling(self, n):
        pass

    def read_all_data(self):
        pass

    def get_data_count(self):
        return len(self._data_list)


# For multiprocess training
class TransitionBuffer(DataBuffer):

    def __init__(self, root_path, capacity, name, backup=None):
        super(TransitionBuffer, self).__init__(root_path, capacity)
        self._backup = backup
        self.name = name

    def store(self, transition):
        self._lock.acquire()
        data_list_len = len(self._data_list)

        if data_list_len > self._capacity:
            if self._backup is not None:
                # store to online data
                # if online data size is full, delete oldest data in online data.
                # and deleted data added to offline data.
                data = self._data_list[0]
                self._backup.store(data)
            del self._data_list[0]

        self._data_list.append(transition)
        self._lock.release()

    def prior_sampling(self, n):
        pass

    def uniform_sampling(self, n):
        q_size = len(self._data_list)
        if n > q_size:
            return None, None

        self._lock.acquire()
        idx_list = np.random.choice(q_size, n)
        b_memory = [self._data_list[i] for i in idx_list]
        self._lock.release()

        return b_memory, None

    def read_all_data(self):
        datalist = []

        for (path, _, files) in os.walk(self._root_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.txt':
                    full_filename = os.path.join(path, filename)
                    datalist.append(full_filename)

        self._lock.acquire()
        self._data_list += datalist
        self._lock.release()

        print('Read data count : %d' % len(datalist))

    @staticmethod
    def read_data(path):
        with open(path, 'r') as f:
            data_dict = json.load(f)

        if data_dict is None:
            raise NameError('data parsing fail')

        state_before = data_dict['state before']
        state_after = data_dict['state after']
        action = data_dict['action']
        prev_internal = data_dict['internal state before']
        after_internal = data_dict['internal state after']
        reward = data_dict['reward']
        terminal = data_dict['terminal']

        state_before = np.array(state_before)
        state_after = np.array(state_after)
        action = np.array(action)
        prev_internal = np.array(prev_internal)
        after_internal = np.array(after_internal)
        reward = np.array([reward])
        terminal = np.array([terminal])

        data_dict = {'state before': state_before,
                     'state after': state_after,
                     'action': action,
                     'internal state before': prev_internal,
                     'internal state after': after_internal,
                     'reward': reward,
                     'terminal': terminal}

        return data_dict
