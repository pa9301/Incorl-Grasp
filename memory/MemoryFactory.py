# qtopt memory wrapper
# this class has offline memory, online memory, offline memory manager, online memory manager
import sys
import numpy as np

from memory.ReplayBuffer import TransitionBuffer
from memory.MemoryManager import MemoryManager

class MemoryFactory:
    def __init__(self, offlineDatapath, onlineMemorySize, ratio=0.5):
        self._sampling_ratio = ratio

        self.offlineMemory = TransitionBuffer(offlineDatapath, sys.maxsize, 'offline')
        self.onlineMemory = TransitionBuffer(None, onlineMemorySize, 'online', backup=self.offlineMemory)

        #load offline data
        if offlineDatapath is not None:
            self.offlineMemory.ReadAllData()

        #batch size is not important
        self.offMemManager = MemoryManager(self.offlineMemory, 1, 100, MemoryManager.UNIFORM_SAMPLING, 4, 'offline')
        self.onMemoryManager = MemoryManager(self.onlineMemory, 1, 100, MemoryManager.UNIFORM_SAMPLING, 4, 'online')

        #background thread start
        self.offMemManager.StartLoadData()
        self.onMemoryManager.StartLoadData()

    def getdata(self, batchsize):
        offline_cnt = int(batchsize * self._sampling_ratio)

        # exception handling - offline data count insufficient
        if self.offlineMemory.GetDataCount() < offline_cnt:
#            offline_cnt = self.offlineMemory.GetDataCount()
#            online_cnt = batchsize - offline_cnt
            onlineData = self.getdata_online_only(batchsize)
            batchdata = onlineData
        else:
            cnt = self.offlineMemory.GetDataCount()
            online_cnt = batchsize - offline_cnt
            onlineData = self.getdata_online_only(online_cnt)
            offlineData = self.getdata_offline_only(offline_cnt)

            list_keys_offline = [k for k in offlineData]
            list_keys_online = [k for k in onlineData]

            if set(list_keys_offline) != set(list_keys_online):
                raise NameError('offline/offline data set is diffrent ')

            batchdata = []
            for key in list_keys_online:
                off = offlineData[key]
                on = onlineData[key]

                offon = np.array(list(off) + list(on))

                batchdata.append(offon)

            batchdata = dict(zip(list_keys_online, batchdata))

        return batchdata

    def getdata_offline_only(self, batchsize):
        data = self.offMemManager.getLoadedData(batchsize)
        return self._stackingDict(data)

    def getdata_online_only(self, batchsize):
        data = self.onMemoryManager.getLoadedData(batchsize)
        return self._stackingDict(data)

    def _stackingDict(self, datadict):
        list_keys = [k for k in datadict[0]]
        list_data = []
        for d in datadict:
            each_data = [v for v in d.values()]
            list_data += [each_data]

        list_data_T = list(map(list, zip(*list_data)))

        # reshape list to data
        list_data = [np.array(d) for d in list_data_T]
        datadict = dict(zip(list_keys, list_data))

        return datadict



