
import numpy as np
import random
import os

from abc import *
from multiprocessing import Manager, Lock
from multiprocessing.managers import BaseManager
import json

class DataBuffer(object):
    def __init__(self, rootpath, capacity):
        self._rootpath = rootpath
        self._Capacity = capacity

        self._Lock = Lock()
        self._DataList = Manager().list()

    @abstractmethod
    def store(self, transition):
        pass

    @abstractmethod
    def PriorSampling(self, n):
        pass

    @abstractmethod
    def UniformSampling(self, n):
        pass

    def ReadAllData(self):
        pass

    def GetDataCount(self):
        return len(self._DataList)

    def GetCapacity(self):
        return self._Capacity

# For multiprocess training
class TransitionBuffer(DataBuffer):
    def __init__(self, rootpath, capacity, name, backup = None):
        super(TransitionBuffer, self).__init__(rootpath, capacity)
        self._backup = backup
        self.name = name

    def store(self, transition):
        self._Lock.acquire()
        datalistlen = len(self._DataList)

        if datalistlen > self._Capacity:
            if self._backup is not None:
                # store to online data
                # if onlindata size is full, delete oldest data in online data. and deleted data added to offline data
                data = self._DataList[0]
                self._backup.store(data)
            del self._DataList[0]

        self._DataList.append(transition)
        self._Lock.release()

    def PriorSampling(self, n):
        pass

    def UniformSampling(self, n):
        q_size = len(self._DataList)
        if n > q_size:
            return None, None

        self._Lock.acquire()
        idxlist = np.random.choice(q_size, n)
        b_memory = [self._DataList[i] for i in idxlist]
        self._Lock.release()

        return b_memory, None

    def ReadAllData(self):
        datalist = []

        for (path, dir, files) in os.walk(self._rootpath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.txt':
                    full_filename = os.path.join(path, filename)
                    datalist.append(full_filename)

        self._Lock.acquire()
        self._DataList += datalist
        self._Lock.release()

        print('Read data count : %d' % len(datalist))

    def ReadData(self, path):
        #    f = open(path, 'r')ReadData
        #    data_dict = json.load(f)

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
