import os
import random
import cv2
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from abc import *
import queue
from datetime import datetime
import json


def ImageLoadProcess(args):     # : 데이터 로드를 위한 프로세스, Normalize 및 크기, 형식 지정
    id = args[0]
    procCount = args[1]
    imgCnt = args[2]
    q = args[3]
    buffersize = args[4]
    ImgList = args[5]
    batchsize = args[6]

    cur_count = 0
    imgcount = imgCnt
    ImgQueue = q
    buffersize = buffersize / batchsize
    ImgList = ImgList

    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

    if len(ImgList) == 0:
        # 읽어들인 이미지가 없음
        return 0

    loadimglist = []
    loadseglist = []

    while True:
        while ImgQueue.qsize() > buffersize:
            sleep(1)

        idx = (cur_count * procCount + id)
        cur_count += 1

        if idx >= imgcount:
            idx -= imgcount
            cur_count = 0

        if idx == 0:
            random.shuffle(ImgList)

        path = ImgList[idx]
        img = cv2.imread(path['rgb'])
        if img is None:
            continue

        img = img[..., ::-1]   # bgr to rgb

        seg = None
        if 'seg' in path and path['seg']:
            seg = np.load(path['seg'])

        img, seg, _ = random_crop(img, seg)
        img = ImgNormalize(img, 255.)

        loadimglist.append(img)
        if seg is not None:
            seg = segmentLabeling(seg)
            seg = ImgNormalize(seg, 3.)  # -1 에서  1사이의 값을 가지도록
            loadseglist.append(seg)

        if len(loadimglist) >= batchsize:
            imgarray = np.array(loadimglist)
            segarray = None
            if len(loadseglist) >= batchsize:
                segarray = np.array(loadseglist)
                seg_shape = segarray.shape
                segarray = np.reshape(segarray, (seg_shape[0], seg_shape[1], seg_shape[2], 1))

            ImgQueue.put([id, imgarray, segarray])
            loadimglist = []
            loadseglist = []


class MemoryLoadBase:
    __metaclass__ = ABCMeta

    def __init__(self, batchsize, numprocess):
        super().__init__()
        self._DataQueue = queue.Queue()
        self._procList = []

        self._batchsize = batchsize
        self._numProcess = numprocess

    def __del__(self):
        self.KillAllProcess()       # : 프로세스 종료

    @abstractmethod
    def _ReadAllDataPath(self):     ## 안쓰고있음
        raise NotImplementedError()

    @abstractmethod
    def StartLoadData(self):        ## 안쓰고있음
        raise NotImplementedError()

    def RunProcess(self, func, proc_args):  # : StartLoadData 중 프로세스 시작을 위해 필요
        self.pool = ThreadPoolExecutor(self._numProcess)    # : 스레드 풀을 지정
        for i in range(self._numProcess):
            partial_args = (i,) + proc_args
            proc = self.pool.submit(func, partial_args)
            self._procList.append(proc)                     # :

    def getLoadedData(self):        ## 안쓰고있음
        while True:
            Qsize = self.getCollectedSize()
            if Qsize > 0:
                break
            sleep(1)

        return self._DataQueue.get()

    def getCollectedSize(self):     ## 안쓰고있음
        return self._DataQueue.qsize()

    def KillAllProcess(self):
        self.pool.shutdown(wait=True)   # : 프로세스 종료

        while not self._DataQueue.empty():
            self._DataQueue.get()


class ImageCollector(MemoryLoadBase):
    def __init__(self, rootpath, numProcess, buffersize, batchsize, bCollectSeg = False):
        super().__init__(batchsize, numProcess)

        self._rootpath = rootpath
        self._buffersize = buffersize

        self._collect_segflag = bCollectSeg

        ImgList, self._imgcount = self._ReadAllImgPath()    # : 읽어온 데이터경로와 데이터 갯수를 반환
        self._ImgList = ImgList

        assert not (self._imgcount == 0), 'Load data count is ZERO.'

    def _ReadAllImgPath(self):      # : 읽어올 데이터들의 이미지 및 경로를 파악
        img_list = []

        for (path, dir, files) in os.walk(self._rootpath):
            for filename in files:
                namepair = dict()
                ext = os.path.splitext(filename)[-1]
                if ext == '.png' or ext == '.jpg' or ext == '.bmp':
                    full_filename = os.path.join(path, filename)
                    namepair['rgb'] = full_filename

                    if self._collect_segflag:
                        except_ext = filename.replace(ext,".npy")
                        full_filename = os.path.join(path, except_ext)
                        if os.path.exists(full_filename):
                            namepair['seg'] = full_filename
                        else:
                            print('seg image count != rgb image count')
                    img_list.append(namepair)

        return img_list, len(img_list)  # : 데이터들의 경로, 데이터들의 갯수

    def StartLoadData(self):    # : 데이터 로딩과 프로세스를 시작함
        random.shuffle(self._ImgList)               # : 데이터들의 순서를 랜덤하게 섞어줌
        self.RunProcess(ImageLoadProcess, proc_args=(self._numProcess,      # : 데이터 로드 프로세스를 실행
                                                     self._imgcount, self._DataQueue,
                                                     self._buffersize, self._ImgList,
                                                     self._batchsize))

    def getDataCnt(self):       ## 안쓰고있음
        return self._imgcount   # : 읽어온 데이터 갯수 반환


def random_crop(img, seg=None, dep=None):          # : 1280 720
    target_crop_w = 444 #360 #472
    target_crop_h = 360 #360 #472
    pad = int((target_crop_w - target_crop_h)/2)

    temp_rgb_img = np.zeros((target_crop_w, target_crop_w, 4), dtype=np.uint8)
    temp_rgb_img.fill(255)
    temp_seg_img = np.zeros((target_crop_w, target_crop_w), dtype=np.uint8)
    temp_seg_img.fill(0)
    temp_dep_img = np.zeros((target_crop_w, target_crop_w), dtype=np.uint8)
    temp_dep_img.fill(0.9999)

    h, w, c = img.shape
    c_x = w / 2  # + 30
    c_y = h / 2
    # random crop
    #    crop_cx = random.randint(0, 21) - 10
    #    crop_cy = random.randint(0, 7) - 3
    # random crop 대신에 고정된 crop을 현재는 사용함
    crop_cx = 10
    crop_cy = 3

    crop_w = [int(c_x - (target_crop_w / 2)), int(c_x + (target_crop_w / 2))]
    crop_h = [int(c_y - (target_crop_h / 2)), int(c_y + (target_crop_h / 2))]

    croped_w = [0, int(crop_w[1] - crop_w[0])]
    croped_h = [0, int(crop_h[1] - crop_h[0])]

    temp_rgb_img[croped_h[0]+pad:croped_h[1]+pad, croped_w[0]:croped_w[1], :] = img[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1], :]
    if seg is not None:
        temp_seg_img[croped_h[0]+pad:croped_h[1]+pad, croped_w[0]:croped_w[1]] = seg[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
    if dep is not None:
        temp_dep_img[croped_h[0]+pad:croped_h[1]+pad, croped_w[0]:croped_w[1]] = dep[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]

    return temp_rgb_img, temp_seg_img, temp_dep_img


def ImgNormalize(img, scale):
#    img = img.astype('float') / (scale / 2.0)
#    img -= 1
#    img = img.astype('float')
    img = img / 127.5 - 1.0

    return img


def ImgInverNormalize(img, scale):
    img += 1
    img *= (scale / 2.0)
    img = img.astype('uint8')
    return img


def segmentLabeling(seg):
#    seg[seg == 1] = -1
#    seg[seg == 3] = 1
#    seg[seg == -1] = 3
    seg[seg > 3] = 4
# ground id : 0 / table id : 1 / tray id : 2 / robot id : 3 / obj id : 4 ~ 64
    seg[seg == -1] = 1      # 카메라 뷰 포인트로 안찍히는 곳은 테이블과 동일한 아이디로 보냄
    seg[seg == 0] = 1       # ground는 table과 동일한 아이디로
# 변경 후.
# ground id & table id : 1 / tray : 2 / robot:3 / obj : 4 ~64
# 0, -1번에는 아무것도 존재하지 않음.
    seg = seg - 1.0

    return seg


def WriteData(rootpath, prev_state, aftor_state, prev_seg, after_seg,
              action, prev_internal, after_internal, reward, terminal, idx, imgext = '.bmp'):
    pid = os.getpid()
    nowtime = datetime.now()
    strNowTime = nowtime.strftime("%y%m%d_%H%M%S")
    FileName = '%d_%s_%d' % (pid, strNowTime, idx)

    data_dict = {'state before': prev_state,
                 'state after': aftor_state,
                 'action': action,
                 'internal state before': prev_internal,
                 'internal state after': after_internal,
                 'reward': reward,
                 'terminal': terminal}

    path = rootpath + '/file/' + FileName + '.txt'

    with open(path, 'w') as f:
        f.write(json.dumps(data_dict))

    return path


# = img write
def WriteImg(rootpath, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, idx, imgext = '.png'):
    pid = os.getpid()
    nowtime = datetime.now()
    strNowTime = nowtime.strftime("%y%m%d_%H%M%S")
    FileName = '%d_%s_%d' % (pid, strNowTime, idx)

    img_before_path = ImgOnlyWrite(rootpath + "/before/attention", FileName, img, imgext)
    ext_pos_before_path = ImgOnlyWrite(rootpath + "/before/external_pose", FileName, ext_pos_img, imgext)
    ext_vis_path = ImgOnlyWrite(rootpath + "/before/external_vis", FileName, ext_vis_img, imgext)
    int_pos_path = ImgOnlyWrite(rootpath + "/before/internal_pose", FileName, int_pos_img, imgext)
    int_vis_path = ImgOnlyWrite(rootpath + "/before/internal_vis", FileName, int_vis_img, imgext)


# = SFPN
def WriteSFPNData(rootpath, images, internal_states, actions, is_success, idx, imgext = '.png'):
    pid = os.getpid()
    nowtime = datetime.now()
    strNowTime = nowtime.strftime("%y%m%d_%H%M%S")

    size = len(images)

    for i in range(size):
        img_filepath = None
        info_filepath = None
        if is_success is True:
            img_filepath = rootpath + "/success/image"
            info_filepath = rootpath + "/success/info/"
        else:
            img_filepath = rootpath + "/fail/image"
            info_filepath = rootpath + "/fail/info/"

        image = images[i]
        FileName = '%d_%s_%d' % (pid, strNowTime, idx+i)
        _ = ImgWrite(img_filepath, FileName, image, imgext)

        action = actions[i]
        internal_state = internal_states[i]
        data_dict = {'action': action,
                     'internal_state': internal_state}

        path = info_filepath + FileName + '.txt'

        with open(path, 'w') as f:
            f.write(json.dumps(data_dict))


def ReadData(path):

    data_dict = {}
    with open(path, 'r') as f:
        for line in f:
            (key, val) = line.split(':')
            data_dict[key] = val

    f.close()

#    print(type(data_dict))

    if data_dict is None:
        raise NameError('data parsing fail')

    state_before = data_dict['state before']
    state_after = data_dict['state after']
#    seg_before_path = data_dict['seg before']
#    seg_after_path = data_dict['seg after']
    action = data_dict['action']
    prev_internal = data_dict['internal state before']
    after_internal = data_dict['internal state after']
    reward = data_dict['reward']
    terminal = data_dict['terminal']

#    img_before = cv2.imread(img_before_path)
#    img_after = cv2.imread(img_after_path)
#    seg_before = np.load(seg_before_path)
#    seg_after = np.load(seg_after_path)
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


def ImgWrite(rootpath, filename, rgb, seg, imgext='.bmp'):
    img_path = rootpath + '/' + filename + imgext
    seg_path = rootpath + '/' + filename

    # img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, rgb)
    np.save(seg_path, seg)

    seg_path += '.npy'

    return img_path, seg_path


def ImgOnlyWrite(rootpath, filename, rgb, imgext='.bmp'):
    img_path = rootpath + '/' + filename + imgext

    # img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, rgb)

    return img_path

def getPoints(label_image_array, object_index):
    return np.argwhere(label_image_array == object_index)

def refine_segmented_image_by_connected_component(org_label_array, obj_id, num_noise_pixel):
    shape = (256, 256)
    binary_image_array = np.zeros(shape=shape, dtype=np.uint8)

    pointList = getPoints(org_label_array, obj_id)
    binary_image_array.fill(0)

    binary_image_array[pointList[:, 0], pointList[:, 1]] = obj_id

    connectivity = 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image_array, connectivity, cv2.CV_32S)
    # numlabel : The first cell is the number of labels
    # The second cell is the label matrix

    if num_labels > 1:

        for m in range(0, num_labels):
            pixels_num = stats[m, cv2.CC_STAT_AREA]
            if pixels_num < num_noise_pixel:
                pointList2 = getPoints(labels, m)
                binary_image_array[pointList2[:, 0], pointList2[:, 1]] = 0


    return binary_image_array #, new_label_color_image_array

def ChangeName(obj_name):
    if obj_name == 4:
        name = 'apple'
    elif obj_name == 5:
        name = 'baseball'
    elif obj_name == 6:
        name = 'banana'
    elif obj_name ==7:
        name = 'eraser'
    elif obj_name == 8:
        name = 'tomatosoup'
    elif obj_name == 9:
        name = 'clamp'
    elif obj_name == 10:
        name = 'cube'
    elif obj_name == 11:
        name = 'redcup'
    elif obj_name == 12:
        name = 'driver'
    elif obj_name == 13:
        name = 'boxdiget'
    elif obj_name == 14:
        name = 'diget'
    elif obj_name == 15:
        name = 'dumbel'
    elif obj_name == 16:
        name = 'gotica'
    elif obj_name == 17:
        name = 'fork'
    elif obj_name == 18:
        name = 'lock'
    elif obj_name == 19:
        name = 'ladle'
    elif obj_name == 20:
        name = 'orange'
    elif obj_name == 21:
        name = 'spam'
    elif obj_name == 22:
        name = 'woodglue'
    elif obj_name == 23:
        name = 'vitamin_water'
    elif obj_name == 24:
        name = 'rubberduck'
    elif obj_name == 25:
        name = 'sauce'
    elif obj_name == 26:
        name = 'scoop'
    elif obj_name == 27:
        name = 'spoon'
    else:
        name = 'unknown'
    return name