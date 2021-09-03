import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import random
from glob import glob


class ObjLoader:
    def __init__(self, objpath, ObjBacthSize):
        self.objpath = objpath
        self.ObjBacthSize = ObjBacthSize            # 한 번 Load할 때 불러올 Object 갯수  ## 안쓰고있음 (실질적으로)
        self.objTotalCount = 0
        self.objlistIdx = 0                         ## 확인필요 ( LoadSingleObj 에서만 쓰고있음 )

        self._ReadObjPath()

    # = 불러올 오브젝트를 정렬하고 인덱싱 한 뒤 섞음
    def _ReadObjPath(self):
        pattern = "*.urdf"
        unorderedlist = []                                  # : 순서정렬을 위한 변수
        for dir_val, _, _ in os.walk(self.objpath):         ## 변경 dir -> dir val (dir이 내장으로 존재함)
            files = glob(os.path.join(dir_val, pattern))    # : 경로 안의 모든 물체
            unorderedlist.extend(files)                     # : 순서정렬 리스트에 추가

        self.objlist = sorted(unorderedlist)                # : 리스트를 순서정렬함
        self.objTotalCount = len(self.objlist)              # : 정렬된 오브젝트들의 전체 갯수 저장
        # allocate unique id
        for idx in range(self.objTotalCount):               # : 원래 오브젝트 유니크 아이디를 순서에따라 정렬
            self.objlist[idx] = [self.objlist[idx], idx]    # : [정렬된 유니크아이디, 순서]

        random.shuffle(self.objlist)                        # : 랜덤하게 섞음

    def LoadObjURDF(self):                              ## 안쓰고있음
        for idx in range(self.ObjBacthSize):
            self._GetObjectPath()

    def _GetObjectPath(self):                           ## 확인필요 ( LoadSingleObj 에서만 쓰고있음 )
        str = self.objlist[self.objlistIdx][0]
        uniqueid = self.objlist[self.objlistIdx][1]
        self.objlistIdx += 1
        if self.objlistIdx >= self.objTotalCount:
            self.objlistIdx = 0
 #           random.shuffle(self.objlist)
        return str, uniqueid

    def LoadSingleObj(self):
        return self._GetObjectPath()

    def GetObjTotalCount(self):                         ## 안쓰고있음
        return self.objTotalCount