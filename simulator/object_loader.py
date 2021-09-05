import inspect
import os

import random
from glob import glob

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


class ObjLoader:

    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.objTotalCount = 0
        self.obj_list_idx = 0

        self._read_obj_path()

    # = 불러올 오브젝트를 정렬하고 인덱싱 한 뒤 섞음
    def _read_obj_path(self):
        pattern = "*.urdf"
        unordered_list = []  # : 순서정렬을 위한 변수
        for dir_val, _, _ in os.walk(self.obj_path):
            files = glob(os.path.join(dir_val, pattern))  # 경로 안의 모든 물체
            unordered_list.extend(files)  # 순서정렬 리스트에 추가

        self.obj_list = sorted(unordered_list)  # 리스트를 순서정렬함
        self.objTotalCount = len(self.obj_list)  # : 정렬된 오브젝트들의 전체 갯수 저장
        # allocate unique id
        for idx in range(self.objTotalCount):  # : 원래 오브젝트 유니크 아이디를 순서에따라 정렬
            self.obj_list[idx] = [self.obj_list[idx], idx]  # : [정렬된 유니크아이디, 순서]

        random.shuffle(self.obj_list)  # : 랜덤하게 섞음

    def load_single_obj(self):
        _str = self.obj_list[self.obj_list_idx][0]
        unique_id = self.obj_list[self.obj_list_idx][1]
        self.obj_list_idx += 1
        if self.obj_list_idx >= self.objTotalCount:
            self.obj_list_idx = 0
        return _str, unique_id
