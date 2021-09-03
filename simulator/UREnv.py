import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import gym
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
import simulator.UR as UR
import cv2
import math

from simulator.ObjectLoader import ObjLoader
from utils.utils import random_crop, refine_segmented_image_by_connected_component

EPSILON = 0.007     # : 거리 스레스 홀드
STEPLIMIT = 10      # : action 스텝 수 지정

class URGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,  # : 시뮬레이션 환경 초기화 코드
                 urdfRoot=pybullet_data.getDataPath(),                      # : 시뮬레이션 URDF 파일 위치
                 objRoot=os.path.join(currentdir, "./objects"),  # : object URDF 파일 위치
#                 objRoot_train = os.path.join(currentdir, "./objects_train"),      # : object URDF 파일 위치
#                 objRoot_test = os.path.join(currentdir, "./objects_test"),  # : object URDF 파일 위치
                 actionRepeat=1,                                            # : step 실행 중 액션 반복 횟수
                 isEnableSelfCollision=True,
                 renders=False,                                             # : 랜더링 flag
                 useUniqueObjID=True,                                       # : 물체 고유 ID 지정 flag
                 objBacthCount = 3,                                         # : 시뮬레이션 기본 object Load 갯수
                 safetypoint1=np.array([0.1895, -0.3880, -0.1370]),  # : Tray내 Safezone 범위지정1            ## v+
                 safetypoint2=np.array([0.7355, 0.3580, 0.2500])):  # : Tray내 Safezone 범위지정

        self._timeStep = 1. / 240.                                  # : 시뮬레이션 내부 시간 지정 Unit = sec
        self._jumptime = 0.05
        self._urdfRoot = urdfRoot                                   # : 시뮬레이션 URDF 파일 위치
        self._actionRepeat = actionRepeat
        ## self._isEnableSelfCollision = isEnableSelfCollision      #### ???? // 안쓰고있음
        self._observation = []                                      # : RGB 카메라 이미지 저장 변수
        self._depthobservation = []                                 # : depth 카메라 이미지 저장 변수
        self._segmetobservation = []                                # : segmentation 카메라 이미지 저장 변수
        self._envStepCounter = 0                                    # : step진행 카운터
        self._renders = renders                                     # : 랜더링 flag
        self._useUniqueObjID = useUniqueObjID                       # : 물체 고유 ID 지정 flag
        self.terminated = 0                                         # : termination flag
        self._width = 640#1280          # 640                           # : 뽑을 데이터 너비   ## v
        self._height = 360#720          # 480                           # : 뽑을 데이터 높이   ## v
        self._objRoot = objRoot  # : object URDF 파일 위치
#        self._objRoot_train = objRoot_train                                     # : object URDF 파일 위치
#        self._objRoot_test = objRoot_test  # : object URDF 파일 위치
        self._objBatchCount = objBacthCount                         # : 시뮬레이션 기본 object Load 갯수 // argument 에서 지정
        self._robot_id_list = []                                    # : 로봇의 id 리스트 생성
        self._uniqueid = -1                                         # : observe 할 object 선정 변수

        # obj data set 관련
        self.objectloader = ObjLoader(self._objRoot, 1)
        #        self.objectloader_train = ObjLoader(self._objRoot_train, 1)             # : 오브젝트 [유니크아이디 설정 ,순서] 정렬
#        self.objectloader_test = ObjLoader(self._objRoot_test, 1)  # : 오브젝트 [유니크아이디 설정 ,순서] 정렬

        # view matrixt 관련 - 고정 카메라
        # self._cam_src_position = np.array([0.446, 0.428, 0.6666])
        # self._cam_src_position = np.array([0.457, 0.418, 0.72])
        self._cam_src_position = np.array([0.4625, 0.6230, 0.4485])

        # self._cam_dst_position = np.array([0.457, 0.04, -0.155])    ## v
        # self._cam_dst_position = np.array([0.457, 0.054, -0.158])  ## v
        self._cam_dst_position = np.array([0.4625, -0.0450, -0.1370])  ## v

        up_vec = np.cross((self._cam_dst_position - self._cam_src_position), np.array([-1, 0, 0]))
        self._viewMat = p.computeViewMatrix(self._cam_src_position, self._cam_dst_position, up_vec)

        # intrinsic matrix d435 16:9-> 4:3 카메라 매트릭스
        # cx: 315.663, cy: 239.674, fx: 611.604, fy: 609.939, n: 0.53, f: 10.0
        # proj_matrix [1x1 : fx / cx, 2x2 : fy / cy, 3x3 : -(f+n) / (f-n), 3x4 : -2fn / (f-n), 4x3 : -1]
        # self._projMatrix = [1.9375, 0.0, 0.0, 0.0, 0.0, 2.5448, 0.0, 0.0, 0.0, 0.0, -1.89193, -1.10932, 0.0, 0.0,
        #                     -1.0, 0.0]

        fovh = 69.4  ## v
        fovv = 42.5
        n = 0.1
        f = 10
        self._projMatrix = [1 / math.tan((fovh / 2) * (math.pi / 180)), 0.0, 0.0, 0.0,
                            0.0, 1 / math.tan((fovv / 2) * (math.pi / 180)), 0.0, 0.0,
                            0.0, 0.0, -(f + n) / (f - n), -1,
                            0.0, 0.0, -(2 * f * n) / (f - n), 0.0]

        # = 이미지 출력 관련 변수
        self.img_size = 444  # 720     # = 472     # 잘라 낼 이미지 크기   ## v
        self.img_size2 = 360
        self.crop_size = 128  # = 128     # 집중 할 이미지 크기
        self._int_vis = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)
        self._int_pos = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)
        self._int = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)
        self._ext_white_vis = np.zeros((self.crop_size, self.crop_size, 4), dtype=np.uint8)
        self._ext_vis = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)
        self._ext_obj_pos = np.zeros((self.img_size, self.img_size, 4), dtype=np.uint8)
        self._ext_attend_pos = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)
        self._ext = np.zeros((self.img_size, self.img_size, 4), dtype=np.uint8)
        self._whole_state = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)

        self.extorg = np.zeros((self.img_size2, self.img_size2, 4), dtype=np.uint8)

        self.ext_vis128 = np.zeros((self.crop_size, self.crop_size, 4), dtype=np.uint8)

        # + 20190722 v2
        self.ext_vis128pad15 = np.zeros((self.crop_size, self.crop_size, 4), dtype=np.uint8)
        self.ext_vis_re128pad15 = np.zeros((self.crop_size, self.crop_size, 4), dtype=np.uint8)
        self.object_section = np.zeros((self.crop_size, self.crop_size, 4), dtype=np.uint8)

        self.fakeReal_pad = np.ones((self.img_size2 + 256, self.img_size2 + 256, 3), dtype=np.uint8)
        self.ext_attend_pos_pad = np.ones((self.img_size2 + 256, self.img_size2 + 256, 4), dtype=np.uint8)
        self.whole_attend_pad = np.ones((self.img_size2 + 256, self.img_size2 + 256, 4), dtype=np.uint8)
        self.ext_vis_pad = np.ones((self.img_size2 + 256, self.img_size2 + 256, 4), dtype=np.uint8)

        self._p = p                             # : pybullet
        if self._renders:                       # : GUI 환경 랜더링 실행시
            cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setRealTimeSimulation(0)              # : 리얼타임 시뮬레이션 설정 - real time clock

        ## +
        p.resetDebugVisualizerCamera(cameraDistance=1.00, cameraYaw=180.00, cameraPitch=-50.00,
                                     cameraTargetPosition=[0.4625, 0.0000, -0.1400])  # [0, 0, 0])

        # safety zone
        self._safetyPoint1 = safetypoint1       # : tray 내 safe point 범위 설정1
        self._safetyPoint2 = safetypoint2       # : tray 내 safe point 범위 설정2

        self._seed()                            # : random seed
        self.IsInitSuccess = self.resetEnv()       # : 리셋 후 리셋성공여부 반환 // 반환값은 안씀

        rgb_to_change_ = cv2.imread(
            'C:/Users/USER/Desktop/AI_Project/IncorlGrasp_VAE_L3/GAN/background/rgb_background_big.png')
        self.rgb_to_change = cv2.cvtColor(rgb_to_change_, cv2.COLOR_BGR2RGB)

    def __del__(self):
        p.disconnect()

    def resetEnv(self):    # : 시뮬레이션 환경 초기화/재초기화 함수
        self.terminated = 0
        self.blockUid = []  # [0] : pybullet uid / [1] : unique id      # : 오브젝트 별 유니크 아이디 저장변수
        p.resetSimulation()                                             # : 모든 물체를 제거하고 빈 공간에서 재시작함
        p.setPhysicsEngineParameter(numSolverIterations=150)            # : Choose the maximum number of constraint solver iterations
        p.setTimeStep(self._timeStep)                                   # : 시뮬레이션 내부 timestep 지정 unit = sec
        if p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1]) < 0:
            return False                                                # : plane 로드 실패시 False반환

        if p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"),
                      0.5000000, 0.00000, -0.861000, 0.000000, 0.000000, 0.0, 1.0) < 0:
            return False                                                # : table 로드 실패시 False반환

        p.setGravity(0, 0, -10)                                         # : 중력설정 (x, y, z) unit = m/s^2
        self._UR = UR.UR(urdfRootPath=self._urdfRoot,
                            safetypoint1=self._safetyPoint1,
                            safetypoint2=self._safetyPoint2)

        if self._UR._isInitSuccess is False:                           # : 리셋이 실패하면, reset False 반환
            return False

        self.traypos = self._UR.getTrayPosition()                      # : Tray 포즈에 대해 계산
        self._sucessCheckHeight = self.traypos[2] + 0.15                # : Tray Z 포즈에 offset을 더함 (성공 판별 Threshold)
        self.openClossGripper(False)  ## 추가 - 원위치 동작 전 그리퍼는 무조건 열기 (던지기 방지)
        self.ActionNGetFinalState([np.pi/2, 0, 0, 0, 0, 0])  # 비켜주기   : 첫번째 조인트의 위치를
        self._robot_id_list.append(self._UR.URUid)

        self._envStepCounter = 0

        # safe zone draw
        self.drawSafeZone()     ## 확인필요 : 세이프존 디버깅

        # Camera line
        up_vec = np.cross((self._cam_dst_position - self._cam_src_position), np.array([1, 0, 0]))
        self._viewMat = p.computeViewMatrix(self._cam_src_position, self._cam_dst_position, up_vec)

        trayAABB = self._p.getAABB(self._UR.trayUid)                   # : 트레이를 기준으로 bounding box 형성
        xlen = (trayAABB[1][0] - trayAABB[0][0]) * 0.6
        ylen = (trayAABB[1][1] - trayAABB[0][1]) * 0.6

        _objBatchCount = np.random.randint(low=1, high=self._objBatchCount, size=1)[0]
        random.shuffle(self.objectloader.objlist)

        for i in range(_objBatchCount):                            # : 오브젝트를 불러올때
            xpos = self.traypos[0] + xlen * (random.random() - 0.5)     # : 트레이 부근으로 X
            ypos = self.traypos[1] + ylen * (random.random() - 0.5)     # :                 Y 좌표를 랜덤으로 형성
            zpos = self.traypos[2] + 0.05 + (0.07 * (i + 1))            # : 트레이 부근으로 Z 를 조금씩 variation
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()           #### ???? 3.14가 두개? 2파이 * 0.5?      #### 3.14159265358979
            ## rgb = np.random.sample(3).tolist() + [1]
            orn = p.getQuaternionFromEuler([0, 0, ang])                 # : z축회전으로 쿼터니언을 오일러좌표계로 변환

            objpath, idx = self.objectloader.LoadSingleObj()
#            if is_train is True:
#                objpath, idx = self.objectloader_train.LoadSingleObj()
#            else:
#                objpath, idx = self.objectloader_test.LoadSingleObj()

            blockUid = p.loadURDF(objpath, xpos, ypos, zpos, orn[0], orn[1], orn[2], orn[3])
            # p.changeVisualShape(blockUid, -1, rgbaColor=rgb)
            self.blockUid.append([blockUid, idx])

        self.objectloader.objlistIdx = 0
        self.performTimeStep()                                          # : # : 시간 측정 함수
        self.WaitUntilObjectSafe()                                      # : 물건이 움직이지 않을때 까지 기다림

        self.openClossGripper(False)    ## 추가 - 원위치 동작 전 그리퍼는 무조건 열기 (던지기 방지)
        self.MoveToHomePosition()                                       # : 원위치로 가는 함수

        return True

    def safeReset(self, is_train):
        safeTrialCount = 10
        for i in range(safeTrialCount):
            bReset = self.resetEnv(is_train)
            if bReset:
                return

        raise NameError('Simulator safe reset fail! check urdf path exist')

    def safeReset(self):
        safeTrialCount = 10
        for i in range(safeTrialCount):
            bReset = self.resetEnv()
            if bReset:
                return

        raise NameError('Simulator safe reset fail! check urdf path exist')

    def IsInitComplete(self):       ## 안쓰고있음
        return self.IsInitSuccess

    def MoveToHomePosition(self):   # : 원위치로 가는 함수
        homeposition = self.getInitPos()                # : 초기 설정 포즈를 홈포지션으로 지정
        self.ActionNGetFinalState(homeposition)         # : 홈포지션으로 이동

    def WaitUntilObjectSafe(self):
        prevstate = []          # : 이전 상태 변수
        isSafe = False          # : 물체 안정 flag

        for i in self.blockUid:
            prevstate.append(p.getBasePositionAndOrientation(i[0]))     # : 각 물체의 상태 저장 #### try excpet 필요

        self.performTimeStep()  # : 시뮬레이션 dynamics 진행

        spendtime = 0           # : 경과시간 변수
        start = time.time()     # : 현재시간 기록
        while not isSafe or spendtime < 1:              # : 안정되지 않았거나, 시간 측정이 1초 이하일경우
            self.performTimeStep()                      # : 시뮬레이션 dynamics 진행

            #### ???? 음...
            spendtime = (time.time() - start)           #### 이미 True False 상태로 저장
            if spendtime > 2.0:                         #### True False 를 다시?
                break

            isSafe = True

            for i in range(len(self.blockUid)):
                id = self.blockUid[i][0]
                initial = prevstate[i]
                curstate = p.getBasePositionAndOrientation(id)          ####

                diffpos = np.array(initial[0]) - np.array(curstate[0])
                ## diffori = np.array(initial[1]) - np.array(curstate[1])

                diffpos_max = np.amax(diffpos)
                ## diffori_max = np.amax(diffori)

                prevstate[i] = curstate

                '''+or diffori_max > 0.5'''
                if diffpos_max > 0.1:
                    isSafe = False
                    break

    def _seed(self, seed=None):     # : 랜덤 시드 설정
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_pos_vis_int_image(self, image):
        IMAGE_SIZE_Y = image.shape[0]  # ?
        IMAGE_SIZE_X = image.shape[1]  # ?
        NUM_CHANNELS = image.shape[2]
        bg_value = 255

        # = 복사
        img_back = np.zeros([IMAGE_SIZE_Y, IMAGE_SIZE_X, NUM_CHANNELS], dtype=np.uint8) + bg_value

        # = 위치 영역 이미지 생성
        img_pose = img_back.copy()
        # = 로봇이 나오는 영역 이미지 생성
        img_vis = img_back.copy()
        # img_align[h_min:h_max, w_min:w_max] = img_robot[:, :]

        list = np.argwhere(image[:, :, 3] == 0)

        if list.size > 0:
            # = 로봇이 나온 영역의 x, y 최대 최소값
            h_max = list[:, 0].max()  # y
            h_min = list[:, 0].min()  # y
            w_max = list[:, 1].max()  # x
            w_min = list[:, 1].min()  # x

            # = 로봇이 나오는 영역
            img_roi = image[h_min:h_max, w_min:w_max]
            # = 로봇 위치 영역
            img_pose[h_min:h_max + 1, w_min:w_max + 1] = (0, 0, 0, 255)
        else:
            img_roi = image

        img_h = int(IMAGE_SIZE_Y / 2) - 1
        img_c = int(IMAGE_SIZE_X / 2) - 1
        h_c = int(img_roi.shape[0] / 2) - 1
        w_c = int(img_roi.shape[1] / 2) - 1

        y = img_roi.shape[0]
        x = img_roi.shape[1]
        yy = IMAGE_SIZE_Y - (img_h - h_c + y)
        xx = IMAGE_SIZE_X - (img_c - w_c + x)

        if yy > IMAGE_SIZE_Y-1:
            y = y - (yy - (IMAGE_SIZE_Y-1))
        if xx > IMAGE_SIZE_X-1:
            x = x - (xx - (IMAGE_SIZE_X-1))

        starty = img_h - h_c
        endy = img_h - h_c + y
        startx = img_c - w_c
        endx = img_c - w_c + x
        img_vis[starty:endy, startx:endx, :] = img_roi[0:y, 0:x, :]

        return img_pose, img_vis

    def make_padding(self, will_crop_img):
        make_size = self.crop_size + will_crop_img.shape[0] + self.crop_size
        pad_img = np.ones((make_size, make_size), dtype=np.uint8) * 255

        half_make_size = make_size // 2
        c_img_h = will_crop_img.shape[0] // 2

        pad = half_make_size - c_img_h

        pad_img[pad:make_size - pad, pad:make_size - pad] = will_crop_img[:, :]

        return pad_img

    def make_padding_img(self, pad_list, will_crop_img, padsize=128, channel=4):
        make_size = will_crop_img.shape[0] + (2 * padsize)
        pad_img = pad_list[0]
        pad_img.fill(255)
        #		pad_img = np.ones((make_size, make_size, channel), dtype=np.uint8) * 255

        half_make_size = make_size // 2
        c_img_h = will_crop_img.shape[0] // 2

        pad = half_make_size - c_img_h
        pad_img[pad:make_size - pad, pad:make_size - pad] = will_crop_img[:, :]

        pad_list.append(pad_img)
        return pad_list

    def maxmin(self, h, w):

        h_max = h.max()
        h_min = h.min()
        w_max = w.max()
        w_min = w.min()

        return h_max, h_min, w_max, w_min

    def init_value(self):
        self._int_vis.fill(255)
        self._int_pos.fill(255)
        self._int.fill(255)
        self._ext_vis.fill(255)
        self._ext_white_vis.fill(255)
        self._ext_obj_pos.fill(255)
        self._ext_attend_pos.fill(255)
        self._ext.fill(255)
        self._whole_state.fill(255)

        ## v+
        self.extorg.fill(255)

        self.ext_vis128.fill(255)
        # + v2
        self.ext_vis128pad15.fill(255)
        self.ext_vis_re128pad15.fill(255)
        self.object_section[..., 3].fill(255)
        self.object_section[..., 1:3].fill(0)
        self.object_section[..., 0].fill(128)

        self.fakeReal_pad.fill(255)
        self.ext_attend_pos_pad.fill(255)
        self.whole_attend_pad.fill(255)
        self.ext_vis_pad.fill(255)


    def getObservation(self, _pid, unique_obj_id=None, use_attention=True):
        img_arr = p.getCameraImage(width=self._width, height=self._height, viewMatrix=self._viewMat,  # : 640 360
                                   projectionMatrix=self._projMatrix)
        rgb = img_arr[2]
        depth = img_arr[3]
        segment_org = img_arr[4]
        self.init_value()

        # = 카메라 랜더링을 가져옴
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        self._observation = np_img_arr  # : bgr 640*360
        np_depth_arr = np.reshape(depth, (self._height, self._width))
        self._depthobservation = np_depth_arr
        np_segment_arr = np.reshape(segment_org, (self._height, self._width))
        self._segmetobservation = np_segment_arr  # : 0~27

        # = 이미지 크기 처리 640*360 -> 444
        rgb_crop, seg_crop_, dep_crop = random_crop(self._observation, self._segmetobservation,
                                                    self._depthobservation)  # : 640*360 -> 444*444

        self.seg_crop = self.mappingUniqueID(seg_crop_)

        rgb_crop_for_GAN = rgb_crop.copy()
        index = np.argwhere(self.seg_crop > 3.0)
        rgb_crop_for_GAN[index[:, 0], index[:, 1], 0:3] = self.rgb_to_change[index[:, 0], index[:, 1], :]
        self.rgb_crop_for_GAN360 = cv2.resize(rgb_crop_for_GAN[:, :, 0:3], (360, 360), interpolation=cv2.INTER_CUBIC)

        # = simulation data
        self.rgb_crop_re360 = cv2.resize(rgb_crop, (360, 360), interpolation=cv2.INTER_CUBIC)
        self.rgb_crop_re256 = cv2.resize(rgb_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        self.seg_crop_re360 = cv2.resize(self.seg_crop, (360, 360), interpolation=cv2.INTER_NEAREST)  # : 0~27
        self.seg_crop_re256 = cv2.resize(self.seg_crop, (256, 256), interpolation=cv2.INTER_NEAREST)

#        index = np.argwhere(self.seg_crop_re256 == 2)
#        self.seg_SegNet_256[index[:, 0], index[:, 1]] = 255

        self.rgb_crop_re_ = self.rgb_crop_re360[..., 0:3]
        self.rgb_crop_256_ = self.rgb_crop_re256[..., 0:3]

        return self.rgb_crop_re_, self.rgb_crop_256_, self.rgb_crop_for_GAN360

    def getExtendedObservation(self, _pid, boxes, scores, classes, fakeReal_360, arm_seg_256, unique_obj_id=None,
                               use_attention=True):

        index = np.argwhere(self.seg_crop_re360 > 3.0)
        fakeReal_360[index[:, 0], index[:, 1], :] = self.rgb_crop_re360[index[:, 0], index[:, 1], 0:3]

        target_obj_box = None
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        index = np.argwhere(classes == unique_obj_id - 3)
        index = np.squeeze(index)
        num_candidate = index.size

        if num_candidate > 0:
            target_obj_score = scores[:, index]
            candi_target_obj_box = boxes[index, :]
            index2 = np.argmax(target_obj_score)
            if num_candidate > 1:
                target_obj_box = candi_target_obj_box[index2, :]
            else:
                target_obj_box = candi_target_obj_box

            target_obj_left_top_y = int(359 * target_obj_box[0])
            target_obj_left_top_x = int(359 * target_obj_box[1])
            target_obj_right_bottom_y = int(359 * target_obj_box[2])
            target_obj_right_bottom_x = int(359 * target_obj_box[3])

        else:
            target_obj_left_top_y = 0
            target_obj_left_top_x = 0
            target_obj_right_bottom_y = 0
            target_obj_right_bottom_x = 0

        fakeReal_pad_list = []
        fakeReal_pad_list.append(self.fakeReal_pad)
        fakeReal_pad_list = self.make_padding_img(fakeReal_pad_list, fakeReal_360, padsize=self.crop_size,
                                                  channel=3)
        self.fakeReal_pad = fakeReal_pad_list[0]
#        self.rgb_SegNet_256 = cv2.resize(fakeReal_360, (256, 256), interpolation=cv2.INTER_CUBIC)

        '''
        # = external 을 만들어주기위한 작업
        ext_attend_pos_pad_list = []
        ext_attend_pos_pad_list.append(self.ext_attend_pos_pad)
        ext_attend_pos_pad_list = self.make_padding_img(ext_attend_pos_pad_list, self._ext_attend_pos, padsize=self.crop_size)    # : 360 -> 616
        self.ext_attend_pos_pad = ext_attend_pos_pad_list[0]
        # = whole_state 를 만들어주기위한 작업
        whole_attend_pad_list = []
        whole_attend_pad_list.append(self.whole_attend_pad)
        whole_attend_pad_list = self.make_padding_img(whole_attend_pad_list, self._whole_state, padsize=self.crop_size)         # : 360 -> 616
        self.whole_attend_pad = whole_attend_pad_list[0]

        # + v2
        ext_vis_pad = self.make_padding_img(self._ext_vis, padsize=self.crop_size)  # : 616
        '''
        if use_attention and unique_obj_id is not None:

            if num_candidate > 0:
                # ---- ---- ---- ---- 15pad  External img 15pad (resize only for all) ---- ---- ---- ----
                extend_attend_size = self.crop_size // 2  # - extend_attend_size = 64
                extend_attend_sizev2 = 15

                h_pad_min = target_obj_left_top_y + 128
                w_pad_min = target_obj_left_top_x + 128
                h_pad_max = target_obj_right_bottom_y + 128
                w_pad_max = target_obj_right_bottom_x + 128

                h_padc = (h_pad_min + h_pad_max) // 2  # : 616 내 지정 물체의 h 중앙
                w_padc = (w_pad_min + w_pad_max) // 2  # : 616 내 지정 물체의 w 중앙

                ph_minv2 = h_pad_min - extend_attend_sizev2  # : 616  내 오브젝트 패딩15 h min
                ph_maxv2 = h_pad_max + extend_attend_sizev2  # : 616  내 오브젝트 패딩15 h max
                pw_minv2 = w_pad_min - extend_attend_sizev2  # : 616  내 오브젝트 패딩15 w min
                pw_maxv2 = w_pad_max + extend_attend_sizev2  # : 616  내 오브젝트 패딩15 w max

                self.ext_attend_pos_pad[ph_minv2: ph_maxv2, pw_minv2: pw_maxv2] = (
                    0, 0, 0, 255)  # : 616 x 616, 15pad ext pose
                self.whole_attend_pad[ph_minv2: ph_maxv2, pw_minv2: pw_maxv2, 0:3] = self.fakeReal_pad[
                                                                                     ph_minv2: ph_maxv2,
                                                                                     pw_minv2: pw_maxv2,
                                                                                     :]  # : 616 pad 에 15pad ext visual 영역을 whole에 추가
                self.ext_vis_pad[ph_minv2: ph_maxv2, pw_minv2: pw_maxv2, 0:3] = self.fakeReal_pad[
                                                                                ph_minv2: ph_maxv2,
                                                                                pw_minv2: pw_maxv2,
                                                                                :]  # : 616 ext_vis 15패딩 이미지

                self.ext_vis_re128pad15 = cv2.resize(self.ext_vis_pad[ph_minv2: ph_maxv2, pw_minv2: pw_maxv2],
                                                     (128, 128),
                                                     interpolation=cv2.INTER_CUBIC)  # : ext_vis 15패딩 128리사이징 결과이미지
            # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

            else:
                self.ext_vis_re128pad15 = self._ext_white_vis  # : 128

            s1 = self.ext_attend_pos_pad.shape[0] // 2 - self.img_size2 // 2  ## v
            e1 = s1 + self.img_size2  ## v
            self._ext_attend_pos = self.ext_attend_pos_pad[s1:e1, s1:e1]  # : 616 ext pose -> 360
            self._whole_state = self.whole_attend_pad[s1:e1, s1:e1]  # : 616 whole -> 360

            # == 임시
            self.extorg = self._whole_state.copy()  # : 360

            arm_seg_360 = cv2.resize(arm_seg_256, (360, 360), interpolation=cv2.INTER_NEAREST)  # : 0~2, 360
            mani_h, mani_w = np.where(arm_seg_360 == 2)  # : int seg
            self._int[mani_h, mani_w, 0:3] = self.rgb_crop_re360[mani_h, mani_w, 0:3]  # : 360 int visual
            self._int[mani_h, mani_w, 3] = 0
            self._int[mani_h, mani_w, 0:3] = fakeReal_360[mani_h, mani_w, :]  # : 360 int visual
            self._whole_state[mani_h, mani_w, 0:3] = fakeReal_360[mani_h, mani_w,
                                                     :]  # : 360 에 int visual을 whole로 추가

            # = 472 int data 만드는 함수
            [self._int_pos, self._int_vis] = self.get_pos_vis_int_image(self._int)

        # == 전체 이미지 부분
        rgb_org = cv2.resize(self.rgb_crop_re360.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        whole_state_img = cv2.resize(self._whole_state.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)

        # == 논문용 / int ext
        int_org = cv2.resize(self._int.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        ext_org = cv2.resize(self.extorg.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)

        # == VAE 이미지 부분
        ext_attend_pose_img = cv2.resize(self._ext_attend_pos.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        # extvis_pad15_img = cv2.resize(self.ext_vis128pad15.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)         # - v2 - 128 128 로 고정된 15pad ext vis
        extvis_pad15re_img = cv2.resize(self.ext_vis_re128pad15.copy(), (128, 128),
                                        interpolation=cv2.INTER_CUBIC)  # - v2 - 128 128 로 리사이징된 15pad ext vis
        intpose_img = cv2.resize(self._int_pos.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        intvis_img = cv2.resize(self._int_vis.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)

        # ----
        rgb_org_ = rgb_org[..., 0:3]
        whole_state_img_ = whole_state_img[..., 0:3]
        int_org_ = int_org[..., 0:3]
        ext_org_ = ext_org[..., 0:3]
        ext_attend_pose_img_ = ext_attend_pose_img[..., 0:3]
        extvis_pad15re_img_ = extvis_pad15re_img[..., 0:3]
        intpose_img_ = intpose_img[..., 0:3]
        intvis_img_ = intvis_img[..., 0:3]

        return rgb_org_, whole_state_img_, ext_attend_pose_img_, extvis_pad15re_img_, intpose_img_, intvis_img_, int_org_, ext_org_
#        return rgb_org, whole_state_img, ext_attend_pose_img, extvis_pad15re_img, intpose_img, intvis_img,\
#               int_org, ext_org, whole_state_img_red_bgr

    def bgra2rgb(self, img):
        img = img[..., 0:3]
#        img = img[..., ::-1]

        return img

    def get_internal_state(self):
        # internal state = {gripper open/ close, endeffector.z - tray.z)
        init_state = []
        gripper_state = 0.0
        if self._UR.IsGripperCloesed():
            gripper_state = 1.0
        z_value = self._UR.getEndEffectorState()[0][2] - self.getTrayPosition()[2]

        init_state.append(gripper_state)
        init_state.append(z_value)
        return init_state

    ##
    # def SetViewMatrix(self, viewMat=None, projMat=None):
    #     if viewMat:
    #         self._viewMat = viewMat
    #     if projMat:
    #         self._projMatrix = projMat

    def step(self, action=None,  blockUid=None, collision_check=False, useInverseKinematics=False):
        done = False

        for i in range(self._actionRepeat):
            if action:
                self._UR.applyAction(action, useInverseKinematics)
            if collision_check:                                         # : 엔드이펙터의 충돌확인
                effstate = self._UR.getEndEffectorState()               # : 엔드이펙터의 포지션, 벨로시티 등 을 받음
                if effstate[0][2] < 0.0:
                    #                if self._termination():
                    break
            self._envStepCounter += 1   # : 스텝진행을 count
        if self._renders:
            time.sleep(self._timeStep)

        if collision_check:
            effstate = self._UR.getEndEffectorState()
            if effstate[0][2] < 0.0:
                #                if self._termination():
                done = True
        #            done = self._termination()
        else:
            done = False

        self.performTimeStep()

        # reward check
        reward = -0.05
        isSuccess, isException = self.IsGraspSuccess(self._sucessCheckHeight, blockUid)   ## 수정지점
        if isSuccess and isException:
            reward = 1.0

        return done, reward

    def MoveRobotExactPosition(self, action, blockUid, collision_check=False, tol_dist=0.02):
        dstpos = np.array([action[0], action[1], action[2]])
        dstang = np.array([])
        dstang_quaternion = np.array([])
        reward = -0.05

        if len(action) > 3:
            dstang = np.array([action[3], action[4], action[5]])
            dstang_quaternion = np.array(self._p.getQuaternionFromEuler(dstang))

        for i in range(STEPLIMIT):
            done, reward = self.step(action, blockUid, collision_check, True)
            if done:
                return True, reward

            effstate = self._UR.getEndEffectorState()
            effpos = np.array(effstate[0])
            effrot = effstate[1]

            dist = effpos - dstpos
            dist = np.linalg.norm(dist)
            dist_ang = 0
            sum_ang = 0
            if len(dstang) != 0:
                dist_ang = effrot - dstang_quaternion
                sum_ang = effrot + dstang_quaternion
                dist_ang = np.linalg.norm(dist_ang)
                sum_ang = np.linalg.norm(sum_ang)

            if dist <= tol_dist and (dist_ang < EPSILON or sum_ang < EPSILON):
                for i in range(STEPLIMIT):
                    _, vel = self.getState()                # : UR의 state 를 반환
                    if all(abs(x) < EPSILON for x in vel):
                        break
                break

        return False, reward

    def _termination(self):     # : 터미네이션 flag 에 따라 image 랜더링 실행
        if self.terminated:     # : 터미네이션 flag 판별
#            self._observation = self.getExtendedObservation(self._uniqueid)
            return True         # : 터메네이션 성공시 True 반환
        ## maxDist = 0.0005
        # closestPoints = p.getClosestPoints(self._UR.trayUid, self._UR.URUid, maxDist)
        closestPoints = p.getContactPoints(self._UR.trayUid, self._UR.URUid)     # : 매니퓰레이터와 트레이 접점 확인

        # tray와 robot이 일정 거리이상 가까워지면 Terminate 시킨다
        if len(closestPoints):  # (actualEndEffectorPos[2] <= -0.43):           # : 접점이 하나라도 있다면 터미네이션
            self.terminated = 1                                                 # : 터미네이션 flag 1

#            self._observation = self.getExtendedObservation(self._uniqueid)     # : 유니크아이디에 대해 랜더링
            return True

        if not self._UR.CheckSafetyZone():                                     # : 엔드 포즈가 Safety를 벗어날시
            return True                                                         # : 터미네이션 True

        return False

    def getRobotInfo(self):     ## 안쓰고있음
        return self._UR.getRobotMotorInfo()


    def ActionNGetFinalState(self, action):     # : a
        done = False
        bVelEnd = False
        bPosDiffEnd = False
        starttime = -1

        for i in range(STEPLIMIT):
            self._UR.applyAction(action)               # : action을 UR로봇에 적용함
            self.step()
            pos, vel = self.getState()                  # : UR의 state 를 반환

            done = False  # self._termination()
            if done:
                break

            action_pos_diff = []
            for x in range(len(pos)):
                action_pos_diff.append(abs(pos[x] - action[x]))

            if all(abs(x) < EPSILON for x in vel):
                bVelEnd = True
            if all(action_pos_diff[x] < 0.05 for x in range(len(action_pos_diff))):
                bPosDiffEnd = True

            if bVelEnd and bPosDiffEnd:
                break
            elif bVelEnd is False and bPosDiffEnd is False:
                starttime = -1
            elif bVelEnd or bPosDiffEnd:
                if starttime > 0:
                    runtime = time.time() - starttime
                    if runtime > 5:
                        break
                else:
                    starttime = time.time()

        return done
    ##

    def getState(self):
        self.performTimeStep(self._timeStep)
        return self._UR.getState()                 # : UR의 state 를 반환

    def getRobotCurrentPos(self):                   ## 안쓰고있음
        pos, _ = self.getState()                    # : UR의 state 를 반환
        return pos

    def setSafetyZone(self, point1, point2):        ## 안쓰고있음
        self._UR.setSafetyZone(point1, point2)

    def drawSafeZone(self):                         ## 실질적으로 안쓰고있음
        safezone_volume = self._UR.calcSafeZoneVol()
        boundingboxPoint = self._UR.getSafePoint()
        if safezone_volume > 0:
            box_color = [0.5, 0.5, 0.5]
            f = [boundingboxPoint[0][0], boundingboxPoint[0][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[1][0], boundingboxPoint[0][1], boundingboxPoint[0][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [boundingboxPoint[0][0], boundingboxPoint[0][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[0][0], boundingboxPoint[1][1], boundingboxPoint[0][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [boundingboxPoint[0][0], boundingboxPoint[0][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[0][0], boundingboxPoint[0][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[0][0], boundingboxPoint[0][1], boundingboxPoint[1][2]]
            t = [boundingboxPoint[0][0], boundingboxPoint[1][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[0][0], boundingboxPoint[0][1], boundingboxPoint[1][2]]
            t = [boundingboxPoint[1][0], boundingboxPoint[0][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[1][0], boundingboxPoint[0][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[1][0], boundingboxPoint[0][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[1][0], boundingboxPoint[0][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[1][0], boundingboxPoint[1][1], boundingboxPoint[0][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[1][0], boundingboxPoint[1][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[0][0], boundingboxPoint[1][1], boundingboxPoint[0][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[0][0], boundingboxPoint[1][1], boundingboxPoint[0][2]]
            t = [boundingboxPoint[0][0], boundingboxPoint[1][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [boundingboxPoint[1][0], boundingboxPoint[1][1], boundingboxPoint[1][2]]
            t = [boundingboxPoint[0][0], boundingboxPoint[1][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [boundingboxPoint[1][0], boundingboxPoint[1][1], boundingboxPoint[1][2]]
            t = [boundingboxPoint[1][0], boundingboxPoint[0][1], boundingboxPoint[1][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [boundingboxPoint[1][0], boundingboxPoint[1][1], boundingboxPoint[1][2]]
            t = [boundingboxPoint[1][0], boundingboxPoint[1][1], boundingboxPoint[0][2]]

            p.addUserDebugLine(f, t, box_color)

    def getInitPos(self):
        return self._UR.getInitPos()

    def getTrayPosition(self):
        return self.traypos

    ## 추가
    def Removetragetobject(self, blockUid):     # : 오브젝트 제거 함수
        for i in range(len(self.blockUid)):
            id = self.blockUid[i][0]
            if id == blockUid:
                p.changeVisualShape(self.blockUid[i][0], -1, rgbaColor=[0, 0, 0, 0])  # 이미지 렌더링 해결 코드
                self.openClossGripper(False)
                p.removeBody(self.blockUid[i][0])
                del self.blockUid[i]
                break

    def IsGraspSuccess(self, height, blockUid):     # : 그래스핑 성공 여부

        exception = False
        if blockUid is not None:
            try:
                objstate = p.getBasePositionAndOrientation(blockUid)    #### 오류지점 : 파묻혀서 오리엔테이션 사라지는 경우
                objheight = objstate[0][2]
                if objheight > height:
                    return True, not exception
            except:
                print('\x1b[1;33m' + "-->>sys: Object's Orientation Not Detected !!" + '\x1b[0;m')
                return False, exception

        ## - ?
        #        successList = self.GetSuccessObjID(height)
        #        if len(successList) != 0:
        #            return True

        return False, not exception

    def getObjPositionblockUid(self):
        objpos = []
        blockUids = []

        try:
            for i in self.blockUid:
                objstate = p.getBasePositionAndOrientation(i[0])
                objpos.append(objstate[0])
                blockUids.append(i[0])

            return objpos, blockUids
        except:
            print('\x1b[1;31m' + "!!>>sys: Object's Orientation Not Detected !! ----해결필요----" + '\x1b[0;m')   #### except 처리 애매함
            return objpos, blockUids


    def performTimeStep(self, sec=None):    # : 시간 측정 함수
        if sec is None:                     # : parameter를 지정 하지않아주면
            sec = self._timeStep            # : 시뮬레이션 time step을 기준으로 설정

        for i in range(int(self._jumptime / sec)):  # : 점프타임을 타임스텝으로 나누었을때, 횟수만큼
            p.stepSimulation()                      # : Step the simulation using forward dynamics.

    def getUniqueIDbyUID(self, uid):
        uniqueid = -1

        num_obj = len(self.blockUid)
        if num_obj > 0:
            baseid = min(self.blockUid, key=lambda t: t[0])[0]

        for id in self.blockUid:
            if uid == id[0]:
                uniqueid = id[1] + 4
                self._uniqueid = uniqueid
                break
        return uniqueid

    def mappingUniqueID(self, seg):
        num_obj = len(self.blockUid)
        if num_obj > 0:
            baseid = 4

            seg_temp = seg.copy()
            for id in self.blockUid:
                #seg_temp = 8 -> 9
                seg_temp[seg == id[0]] = id[1] + baseid
            seg = seg_temp

        return seg

    # True : Closs / False : Open
    def openClossGripper(self, bOpenClose):
        done = False
        starttime = -1
        FingerMaxForce = self._UR.fingerForce
        fingerVal = 1.0 if bOpenClose else 0
        GRIPPER_EPSILON = 0.01

        for i in range(STEPLIMIT):
            self._UR.MoveFinger(fingerVal)         # : True 일때 닫고 False 일때 열기
            self.step()                             # : action 수행

            pos, vel, jntTorque = self._UR.getGripperState()   # : 그리퍼 joint들의 angle, 각속도, 토크 확인

            done = self._termination()              # : action 마침 확인
            if done:                                                            # : action이 끝났을때 break
                break
            ## if all(abs(x) < self._UR.GRIPPER_EPSILON for x in vel):
            if all(abs(x) < GRIPPER_EPSILON for x in vel):                      # : action수행시 체크한 joint들의 속도가 설정 이하로 작을 시 break
                time.sleep(0.1)
                break
            elif any(abs(tq) > (FingerMaxForce - 1) for tq in jntTorque):       # : action수행시 체크한 joint들의 토크가 설정 이상으로 클시 break
                time.sleep(0.1)
                break
            else:                                                               # : action수행시 수행한 시간이 5초이상 일 시
                if starttime > 0:
                    runtime = time.time() - starttime
                    if runtime > 5:
                        break
                else:
                    starttime = time.time()

        return done

    # object bounding box must be exist inside tray bounding box
    # = 트레이 내, 오브젝트가 어느것이라도 있는지 확인
    def checkObjectSafe(self):
        # any object exist in simulator?
        isSafe = False
        if len(self.blockUid) != 0:
            # all object check bounding box in tray
            trayAABB = self._p.getAABB(self._UR.trayUid)    # : 트레이 시뮬레이션 상 위치 확인
            for id in self.blockUid:
                objAABB = self._p.getAABB(id[0])            # : 오브젝트 시뮬레이션 상 위치 확인

                if trayAABB[0][0] < objAABB[0][0] and objAABB[1][0] < trayAABB[1][0]:       # check x
                    if trayAABB[0][1] < objAABB[0][1] and objAABB[1][1] < trayAABB[1][1]:  # check y
                        isSafe = True
                        break

        return isSafe
