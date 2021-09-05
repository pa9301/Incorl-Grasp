# qtopt memory wrapper
# this class has offline memory, online memory, offline memory manager, online memory manager
import sys

import numpy as np

from memory.memory_manager import MemoryManager
from memory.replay_buffer import TransitionBuffer


class MemoryFactory:

    def __init__(self, offline_data_path, online_memory_size, ratio=0.5):
        self._sampling_ratio = ratio

        self.offline_memory = TransitionBuffer(offline_data_path, sys.maxsize, 'offline')
        self.online_memory = TransitionBuffer(None, online_memory_size, 'online', backup=self.offline_memory)

        # load offline data
        if offline_data_path is not None:
            self.offline_memory.read_all_data()

        # batch size is not important
        self.off_mem_manager = MemoryManager(self.offline_memory, 1, 100, MemoryManager.UNIFORM_SAMPLING, 4, 'offline')
        self.on_mem_manager = MemoryManager(self.online_memory, 1, 100, MemoryManager.UNIFORM_SAMPLING, 4, 'online')

        # background thread start
        self.off_mem_manager.start_load_data()
        self.on_mem_manager.start_load_data()

    def get_data(self, batch_size):
        offline_cnt = int(batch_size * self._sampling_ratio)

        # exception handling - offline data count insufficient
        if self.offline_memory.get_data_count() < offline_cnt:
            online_data = self.get_data_online_only(batch_size)
            batch_data = online_data
        else:
            online_cnt = batch_size - offline_cnt
            online_data = self.get_data_online_only(online_cnt)
            offline_data = self.get_data_offline_only(offline_cnt)

            list_keys_offline = [k for k in offline_data]
            list_keys_online = [k for k in online_data]

            if set(list_keys_offline) != set(list_keys_online):
                raise NameError('offline/offline data set is diffrent ')

            batch_data = []
            for key in list_keys_online:
                off = offline_data[key]
                on = online_data[key]

                off_on = np.array(list(off) + list(on))

                batch_data.append(off_on)

            batch_data = dict(zip(list_keys_online, batch_data))

        return batch_data

    def get_data_offline_only(self, batch_size):
        data = self.off_mem_manager.get_loaded_data(batch_size)
        return self._stacking_dict(data)

    def get_data_online_only(self, batch_size):
        data = self.on_mem_manager.get_loaded_data(batch_size)
        return self._stacking_dict(data)

    @staticmethod
    def _stacking_dict(datadict):
        list_keys = [k for k in datadict[0]]
        list_data = []
        for d in datadict:
            each_data = [v for v in d.values()]
            list_data += [each_data]

        list_data_t = list(map(list, zip(*list_data)))

        # reshape list to data
        list_data = [np.array(d) for d in list_data_t]
        datadict = dict(zip(list_keys, list_data))

        return datadict
