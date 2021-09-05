###########################################################
# this class prepare data                                 #
# load transition data to buffer using background process #
###########################################################


from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Lock
from time import sleep


class MemoryManager:
    """Memory 에서 샘플링해와서 해당 주소에 있는 파일들을 백그라운드에서 미리 버퍼에 올려놓는 프로세스 관리자"""

    # Sampling Method
    UNIFORM_SAMPLING = 0
    PRIORITY_SAMPLING = 1

    def __init__(self, origin_memory, back_proc_num, buffer_size, sampling_type, batch_size, name):
        self._proc_num = back_proc_num
        self._data_list = Manager().list()
        self._processList = []

        self._memory = origin_memory
        self._buffer_size = buffer_size
        self._sampling_type = sampling_type
        self._batch_size = batch_size
        self.name = name
        self._lock = Lock()

    def __del__(self):
        # Kill all processes.
        self.pool.shutdown(wait=True)
        del self._data_list[:]

    def get_loaded_data(self, count=1):
        # wait? or pass? below code make pass if data list size is smaller than query count
        while True:
            if len(self._data_list) < count:
                sleep(1)
            else:
                break
        batch_data = self._data_list[:count]
        del self._data_list[:count]

        return batch_data

    def start_load_data(self):
        self.pool = ThreadPoolExecutor(self._proc_num)
        args = (
            self._memory, self._data_list, self._buffer_size,
            self._sampling_type, self._batch_size, self.name, self._lock
        )
        for i in range(self._proc_num):
            partial_args = (i,) + args
            proc = self.pool.submit(data_load_process, partial_args)
            self._processList.append(proc)


def data_load_process(args):
    memory = args[1]
    buffer = args[2]
    buffer_size = args[3]
    mode = args[4]
    batch_size = args[5]

    while True:
        sleep(0.1)

        while len(buffer) > buffer_size:
            sleep(1)

        b_idx = None
        b_memory = None
        if mode == MemoryManager.UNIFORM_SAMPLING:
            b_memory, b_idx = memory.uniform_sampling(batch_size)
        elif mode == MemoryManager.PRIORITY_SAMPLING:
            b_idx, b_memory, _ = memory.prior_sampling(batch_size)

        if b_idx is None and b_memory is None:
            sleep(1)
            continue

        """b_memory structure
        img_s1 / seg_s1 / action / img_s2 / seg_s2 / reward"""
        for mem in b_memory:
            data_dict = memory.read_data(mem)
            buffer.append(data_dict.copy())
