###########################################################
# this class prepare data                                 #
# load transition data to buffer using backgroud procsess #
###########################################################

"""Memory에서 샘플링해와서 해당 주소에 있는 파일들을 백그라운드에서 미리 버퍼에 올려놓는 프로세스 관리자"""
from multiprocessing import Manager, Lock
from concurrent.futures import ThreadPoolExecutor
from utils.utils import ReadData
from time import sleep
import numpy as np

class MemoryManager:
    # Sampling Method
    UNIFORM_SAMPLING = 0
    PRIORITY_SAMPLING = 1

    def __init__(self, originMemory, backProcNum, buffersize, samplingType, batchsize, name):
        self._procNum = backProcNum
        self._DataList = Manager().list()
        self._processList = []

        self._memory = originMemory
        self._buffersize = buffersize
        self._sampingType = samplingType
        self._batchsize = batchsize
        self.name = name
        self._Lock = Lock()

    def __del__(self):
        self.KillAllProcess()

    def KillAllProcess(self):
        self.pool.shutdown(wait=True)
        del self._DataList[:]

    def getLoadedData(self, count=1):
        # wait? or pass? bellow code make pass if data list size is smaller than query count
        while True:
            listlen = len(self._DataList)
            if listlen < count:
                sleep(1)
            else:
                break
        BatchData = self._DataList[:count]
        del self._DataList[:count]

        return BatchData

    def StartLoadData(self):
        self.pool = ThreadPoolExecutor(self._procNum)
        args = (self._memory, self._DataList, self._buffersize, self._sampingType, self._batchsize, self.name, self._Lock)
        for i in range(self._procNum):
            partial_args = (i,) + args
            proc = self.pool.submit(DataLoadProcess, partial_args)
            self._processList.append(proc)

def DataLoadProcess(args):
    id = args[0]
    memory = args[1]
    buffer = args[2]
    buffersize = args[3]
    mode = args[4]
    batchsize = args[5]
    name = args[6]
    Lock = args[7]

    while True:
        sleep(0.1)

        while len(buffer) > buffersize:
            sleep(1)

        b_idx = None
        b_memory = None
        ISWeights = None
        if mode == MemoryManager.UNIFORM_SAMPLING:
            b_memory, b_idx = memory.UniformSampling(batchsize)
        elif mode == MemoryManager.PRIORITY_SAMPLING:
            b_idx, b_memory, ISWeights = memory.PriorSampling(batchsize)

        if b_idx is None and b_memory is None:
            sleep(1)
            continue

        """b_memory structure
        img_s1 / seg_s1 / action / img_s2 / seg_s2 / reward"""
        for mem in b_memory:

            DataDict = memory.ReadData(mem)
            buffer.append(DataDict.copy())



