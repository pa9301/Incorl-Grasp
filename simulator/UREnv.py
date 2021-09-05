import inspect
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)

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

from simulator.object_loader import ObjLoader
from utils.utils import random_crop


class URGymEnv(gym.Env):
    _EPSILON = 0.007  # : 거리 스레스 홀드
    _STEP_LIMIT = 10  # : action 스텝 수 지정

    def __init__(self, urdf_root=pybullet_data.getDataPath(), obj_root=os.path.join(current_dir, "./objects"),
                 action_repeat=1, renders=False, obj_batch_count=3, safety_point1=np.array([0.1895, -0.3880, -0.1370]),
                 safety_point2=np.array([0.7355, 0.3580, 0.2500])):  # Tray 내 Safe zone 범위지정

        self._timeStep = 1. / 240.  # 시뮬레이션 내부 시간 지정 Unit = sec
        self._jump_time = 0.05
        self._urdf_root = urdf_root  # 시뮬레이션 URDF 파일 위치
        self._action_repeat = action_repeat
        self._observation = []  # RGB 카메라 이미지 저장 변수
        self._depth_observation = []  # depth 카메라 이미지 저장 변수
        self._segment_observation = []  # segmentation 카메라 이미지 저장 변수
        self._renders = renders  # 랜더링 flag
        self._terminated = 0  # termination flag
        self._width = 640  # 뽑을 데이터 너비
        self._height = 360  # 뽑을 데이터 높이
        self._objBatchCount = obj_batch_count  # 시뮬레이션 기본 object Load 갯수 // argument 에서 지정
        self._robot_id_list = []  # 로봇의 id 리스트 생성

        # obj data set 관련
        self._object_loader = ObjLoader(obj_root)
        self._cam_src_position = np.array([0.4625, 0.6230, 0.4485])
        self._cam_dst_position = np.array([0.4625, -0.0450, -0.1370])

        up_vec = np.cross((self._cam_dst_position - self._cam_src_position), np.array([-1, 0, 0]))
        self._viewMat = p.computeViewMatrix(self._cam_src_position, self._cam_dst_position, up_vec)

        fov_h = 69.4
        fov_v = 42.5
        n = 0.1
        f = 10
        self._projMatrix = [1 / math.tan((fov_h / 2) * (math.pi / 180)), 0.0, 0.0, 0.0,
                            0.0, 1 / math.tan((fov_v / 2) * (math.pi / 180)), 0.0, 0.0,
                            0.0, 0.0, -(f + n) / (f - n), -1,
                            0.0, 0.0, -(2 * f * n) / (f - n), 0.0]

        # 이미지 출력 관련 변수
        self._img_size = 444  # 잘라 낼 이미지 크기
        self._img_size2 = 360
        self._crop_size = 128  # 집중 할 이미지 크기
        self._int_vis = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)
        self._int_pos = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)
        self._int = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)
        self._ext_white_vis = np.zeros((self._crop_size, self._crop_size, 4), dtype=np.uint8)
        self._ext_vis = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)
        self._ext_obj_pos = np.zeros((self._img_size, self._img_size, 4), dtype=np.uint8)
        self._ext_attend_pos = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)
        self._ext = np.zeros((self._img_size, self._img_size, 4), dtype=np.uint8)
        self._whole_state = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)

        self._ext_org = np.zeros((self._img_size2, self._img_size2, 4), dtype=np.uint8)
        self._ext_vis128 = np.zeros((self._crop_size, self._crop_size, 4), dtype=np.uint8)

        self.ext_vis128pad15 = np.zeros((self._crop_size, self._crop_size, 4), dtype=np.uint8)
        self.ext_vis_re128pad15 = np.zeros((self._crop_size, self._crop_size, 4), dtype=np.uint8)
        self.object_section = np.zeros((self._crop_size, self._crop_size, 4), dtype=np.uint8)

        self.fake_real_pad = np.ones((self._img_size2 + 256, self._img_size2 + 256, 3), dtype=np.uint8)
        self.ext_attend_pos_pad = np.ones((self._img_size2 + 256, self._img_size2 + 256, 4), dtype=np.uint8)
        self.whole_attend_pad = np.ones((self._img_size2 + 256, self._img_size2 + 256, 4), dtype=np.uint8)
        self.ext_vis_pad = np.ones((self._img_size2 + 256, self._img_size2 + 256, 4), dtype=np.uint8)

        self._p = p
        if self._renders:  # GUI 환경 랜더링 실행시
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setRealTimeSimulation(0)  # 리얼타임 시뮬레이션 설정 - real time clock

        p.resetDebugVisualizerCamera(cameraDistance=1.00, cameraYaw=180.00, cameraPitch=-50.00,
                                     cameraTargetPosition=[0.4625, 0.0000, -0.1400])

        # safety zone
        self._safetyPoint1 = safety_point1  # tray 내 safe point 범위 설정1
        self._safetyPoint2 = safety_point2  # tray 내 safe point 범위 설정2

        self.np_random, _ = seeding.np_random(None)  # random seed
        self.block_uid = []
        self._tray_pos = None
        self._success_check_height = 0
        self.reset_env()  # 리셋

        rgb_to_change_ = cv2.imread('./GAN/background/rgb_background_big.png')
        self.rgb_to_change = cv2.cvtColor(rgb_to_change_, cv2.COLOR_BGR2RGB)

    def __del__(self):
        p.disconnect()

    def reset_env(self):  # 시뮬레이션 환경 초기화/재초기화 함수
        self._terminated = 0
        self.block_uid = []  # [0] : pybullet uid / [1] : unique id      # : 오브젝트 별 유니크 아이디 저장변수
        p.resetSimulation()  # : 모든 물체를 제거하고 빈 공간에서 재시작함
        p.setPhysicsEngineParameter(
            numSolverIterations=150)  # : Choose the maximum number of constraint solver iterations
        p.setTimeStep(self._timeStep)  # : 시뮬레이션 내부 time step 지정 unit = sec
        if p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1]) < 0:
            return False  # : plane 로드 실패시 False 반환

        if p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"),
                      0.5000000, 0.00000, -0.861000, 0.000000, 0.000000, 0.0, 1.0) < 0:
            return False  # : table 로드 실패시 False 반환

        p.setGravity(0, 0, -10)  # : 중력설정 (x, y, z) unit = m/s^2
        self.UR = UR.UR(safety_point1=self._safetyPoint1, safety_point2=self._safetyPoint2)

        if self.UR.is_init_success is False:  # : 리셋이 실패하면, reset False 반환
            return False

        self._tray_pos = self.UR.get_tray_position()  # : Tray 포즈에 대해 계산
        self._success_check_height = self._tray_pos[2] + 0.15  # : Tray Z 포즈에 offset 을 더함 (성공 판별 Threshold)
        self._open_close_gripper(False)  # 원위치 동작 전 그리퍼는 무조건 열기 (던지기 방지)
        self._action_n_get_final_state([np.pi / 2, 0, 0, 0, 0, 0])
        self._robot_id_list.append(self.UR.UR_uid)

        # safe zone draw
        self._draw_safe_zone()

        # Camera line
        up_vec = np.cross((self._cam_dst_position - self._cam_src_position), np.array([1, 0, 0]))
        self._viewMat = p.computeViewMatrix(self._cam_src_position, self._cam_dst_position, up_vec)

        trayAABB = self._p.getAABB(self.UR.tray_uid)  # : 트레이를 기준으로 bounding box 형성
        x_len = (trayAABB[1][0] - trayAABB[0][0]) * 0.6
        y_len = (trayAABB[1][1] - trayAABB[0][1]) * 0.6

        _objBatchCount = np.random.randint(low=1, high=self._objBatchCount, size=1)[0]
        random.shuffle(self._object_loader.obj_list)

        for i in range(_objBatchCount):  # : 오브젝트를 불러올때
            x_pos = self._tray_pos[0] + x_len * (random.random() - 0.5)  # 트레이 부근으로 X
            y_pos = self._tray_pos[1] + y_len * (random.random() - 0.5)  # Y 좌표를 랜덤으로 형성
            z_pos = self._tray_pos[2] + 0.05 + (0.07 * (i + 1))  # : 트레이 부근으로 Z 를 조금씩 variation
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn = p.getQuaternionFromEuler([0, 0, ang])  # : z축 회전으로 쿼터니언을 오일러좌표계로 변환

            obj_path, idx = self._object_loader.load_single_obj()

            block_uid = p.loadURDF(obj_path, x_pos, y_pos, z_pos, orn[0], orn[1], orn[2], orn[3])
            # p.changeVisualShape(blockUid, -1, rgbaColor=rgb)
            self.block_uid.append([block_uid, idx])

        self._object_loader.obj_list_idx = 0
        self._perform_time_step()  # 시간 측정 함수
        self.wait_until_object_safe()  # 물건이 움직이지 않을때 까지 기다림

        self._open_close_gripper(False)  # 원위치 동작 전 그리퍼는 무조건 열기 (던지기 방지)
        self.move_to_home_position()  # 원위치로 가는 함수

        return True

    def safe_reset(self):
        safe_trial_count = 10
        for i in range(safe_trial_count):
            if self.reset_env():
                return

        raise NameError('Simulator safe reset fail! check urdf path exist')

    def move_to_home_position(self):  # 원위치로 가는 함수
        home_position = self._get_init_pos()  # 초기 설정 포즈를 홈포지션으로 지정
        self._action_n_get_final_state(home_position)  # 홈포지션으로 이동

    def wait_until_object_safe(self):
        prev_state = []  # 이전 상태 변수
        is_safe = False  # 물체 안정 flag

        for i in self.block_uid:
            prev_state.append(p.getBasePositionAndOrientation(i[0]))  # 각 물체의 상태 저장

        self._perform_time_step()  # 시뮬레이션 dynamics 진행

        spend_time = 0  # 경과시간 변수
        start = time.time()  # 현재시간 기록
        while not is_safe or spend_time < 1:  # 안정되지 않았거나, 시간 측정이 1초 이하일경우
            self._perform_time_step()  # 시뮬레이션 dynamics 진행

            spend_time = (time.time() - start)
            if spend_time > 2.0:
                break

            is_safe = True

            for i in range(len(self.block_uid)):
                _id = self.block_uid[i][0]
                initial = prev_state[i]
                cur_state = p.getBasePositionAndOrientation(_id)

                diff_pos = np.array(initial[0]) - np.array(cur_state[0])
                diff_pos_max = np.amax(diff_pos)

                prev_state[i] = cur_state

                if diff_pos_max > 0.1:
                    is_safe = False
                    break

    @staticmethod
    def get_pos_vis_int_image(image):
        image_size_y = image.shape[0]
        image_size_x = image.shape[1]
        num_channels = image.shape[2]
        bg_value = 255

        # 복사
        img_back = np.zeros([image_size_y, image_size_x, num_channels], dtype=np.uint8) + bg_value

        # 위치 영역 이미지 생성
        img_pose = img_back.copy()
        # 로봇이 나오는 영역 이미지 생성
        img_vis = img_back.copy()

        _list = np.argwhere(image[:, :, 3] == 0)

        if _list.size > 0:
            # 로봇이 나온 영역의 x, y 최대 최소값
            h_max = _list[:, 0].max()  # y
            h_min = _list[:, 0].min()  # y
            w_max = _list[:, 1].max()  # x
            w_min = _list[:, 1].min()  # x

            # 로봇이 나오는 영역
            img_roi = image[h_min:h_max, w_min:w_max]
            # 로봇 위치 영역
            img_pose[h_min:h_max + 1, w_min:w_max + 1] = (0, 0, 0, 255)
        else:
            img_roi = image

        img_h = int(image_size_y / 2) - 1
        img_c = int(image_size_x / 2) - 1
        h_c = int(img_roi.shape[0] / 2) - 1
        w_c = int(img_roi.shape[1] / 2) - 1

        y = img_roi.shape[0]
        x = img_roi.shape[1]
        yy = image_size_y - (img_h - h_c + y)
        xx = image_size_x - (img_c - w_c + x)

        if yy > image_size_y - 1:
            y = y - (yy - (image_size_y - 1))
        if xx > image_size_x - 1:
            x = x - (xx - (image_size_x - 1))

        start_y = img_h - h_c
        end_y = img_h - h_c + y
        start_x = img_c - w_c
        end_x = img_c - w_c + x
        img_vis[start_y:end_y, start_x:end_x, :] = img_roi[0:y, 0:x, :]

        return img_pose, img_vis

    @staticmethod
    def make_padding_img(pad_list, will_crop_img, pad_size=128):
        make_size = will_crop_img.shape[0] + (2 * pad_size)
        pad_img = pad_list[0]
        pad_img.fill(255)

        half_make_size = make_size // 2
        c_img_h = will_crop_img.shape[0] // 2

        pad = half_make_size - c_img_h
        pad_img[pad:make_size - pad, pad:make_size - pad] = will_crop_img[:, :]

        pad_list.append(pad_img)
        return pad_list

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

        self._ext_org.fill(255)

        self._ext_vis128.fill(255)
        self.ext_vis128pad15.fill(255)
        self.ext_vis_re128pad15.fill(255)
        self.object_section[..., 3].fill(255)
        self.object_section[..., 1:3].fill(0)
        self.object_section[..., 0].fill(128)

        self.fake_real_pad.fill(255)
        self.ext_attend_pos_pad.fill(255)
        self.whole_attend_pad.fill(255)
        self.ext_vis_pad.fill(255)

    def get_observation(self, _pid):
        img_arr = p.getCameraImage(width=self._width, height=self._height,
                                   viewMatrix=self._viewMat, projectionMatrix=self._projMatrix)
        rgb = img_arr[2]
        depth = img_arr[3]
        segment_org = img_arr[4]
        self.init_value()

        # 카메라 랜더링을 가져옴
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        self._observation = np_img_arr  # : bgr 640*360
        np_depth_arr = np.reshape(depth, (self._height, self._width))
        self._depth_observation = np_depth_arr
        np_segment_arr = np.reshape(segment_org, (self._height, self._width))
        self._segment_observation = np_segment_arr  # : 0~27

        # = 이미지 크기 처리 640*360 -> 444
        rgb_crop, seg_crop_, _ = random_crop(self._observation, self._segment_observation,
                                                    self._depth_observation)  # : 640*360 -> 444*444

        self.seg_crop = self._mapping_unique_id(seg_crop_)

        rgb_crop_for_gan = rgb_crop.copy()
        index = np.argwhere(self.seg_crop > 3.0)
        rgb_crop_for_gan[index[:, 0], index[:, 1], 0:3] = self.rgb_to_change[index[:, 0], index[:, 1], :]
        self.rgb_crop_for_GAN360 = cv2.resize(rgb_crop_for_gan[:, :, 0:3], (360, 360), interpolation=cv2.INTER_CUBIC)

        # = simulation data
        self.rgb_crop_re360 = cv2.resize(rgb_crop, (360, 360), interpolation=cv2.INTER_CUBIC)
        self.rgb_crop_re256 = cv2.resize(rgb_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        self.seg_crop_re360 = cv2.resize(self.seg_crop, (360, 360), interpolation=cv2.INTER_NEAREST)

        self.rgb_crop_re_ = self.rgb_crop_re360[..., 0:3]
        self.rgb_crop_256_ = self.rgb_crop_re256[..., 0:3]

        return self.rgb_crop_re_, self.rgb_crop_256_, self.rgb_crop_for_GAN360

    def get_extended_observation(self, _pid, boxes, scores, classes, fake_real_360, arm_seg_256, unique_obj_id=None,
                                 use_attention=True):

        index = np.argwhere(self.seg_crop_re360 > 3.0)
        fake_real_360[index[:, 0], index[:, 1], :] = self.rgb_crop_re360[index[:, 0], index[:, 1], 0:3]

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

        fake_real_pad_list = [self.fake_real_pad]
        fake_real_pad_list = self.make_padding_img(fake_real_pad_list, fake_real_360, pad_size=self._crop_size)
        self.fake_real_pad = fake_real_pad_list[0]

        if use_attention and unique_obj_id is not None:

            if num_candidate > 0:
                # ---- ---- ---- ---- 15pad  External img 15pad (resize only for all) ---- ---- ---- ----
                extend_attend_size_v2 = 15

                h_pad_min = target_obj_left_top_y + 128
                w_pad_min = target_obj_left_top_x + 128
                h_pad_max = target_obj_right_bottom_y + 128
                w_pad_max = target_obj_right_bottom_x + 128

                ph_min_v2 = h_pad_min - extend_attend_size_v2  # : 616  내 오브젝트 패딩15 h min
                ph_max_v2 = h_pad_max + extend_attend_size_v2  # : 616  내 오브젝트 패딩15 h max
                pw_min_v2 = w_pad_min - extend_attend_size_v2  # : 616  내 오브젝트 패딩15 w min
                pw_max_v2 = w_pad_max + extend_attend_size_v2  # : 616  내 오브젝트 패딩15 w max

                # : 616 x 616, 15pad ext pose
                self.ext_attend_pos_pad[ph_min_v2: ph_max_v2, pw_min_v2: pw_max_v2] = (0, 0, 0, 255)
                # : 616 pad 에 15pad ext visual 영역을 whole 에 추가
                self.whole_attend_pad[ph_min_v2: ph_max_v2, pw_min_v2: pw_max_v2, 0:3] \
                    = self.fake_real_pad[ph_min_v2: ph_max_v2, pw_min_v2: pw_max_v2, :]
                self.ext_vis_pad[ph_min_v2: ph_max_v2, pw_min_v2: pw_max_v2, 0:3] \
                    = self.fake_real_pad[ph_min_v2: ph_max_v2, pw_min_v2: pw_max_v2, :]  # : 616 ext_vis 15패딩 이미지

                self.ext_vis_re128pad15 = cv2.resize(
                    self.ext_vis_pad[ph_min_v2: ph_max_v2, pw_min_v2: pw_max_v2],
                    (128, 128),
                    interpolation=cv2.INTER_CUBIC
                )  # : ext_vis 15패딩 128리사이징 결과이미지
            # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

            else:
                self.ext_vis_re128pad15 = self._ext_white_vis  # 128

            s1 = self.ext_attend_pos_pad.shape[0] // 2 - self._img_size2 // 2
            e1 = s1 + self._img_size2
            self._ext_attend_pos = self.ext_attend_pos_pad[s1:e1, s1:e1]  # 616 ext pose -> 360
            self._whole_state = self.whole_attend_pad[s1:e1, s1:e1]  # 616 whole -> 360

            self._ext_org = self._whole_state.copy()  # 360

            arm_seg_360 = cv2.resize(arm_seg_256, (360, 360), interpolation=cv2.INTER_NEAREST)  # 0~2, 360
            mani_h, mani_w = np.where(arm_seg_360 == 2)  # int seg
            self._int[mani_h, mani_w, 0:3] = self.rgb_crop_re360[mani_h, mani_w, 0:3]  # 360 int visual
            self._int[mani_h, mani_w, 3] = 0
            self._int[mani_h, mani_w, 0:3] = fake_real_360[mani_h, mani_w, :]  # 360 int visual
            self._whole_state[mani_h, mani_w, 0:3] = fake_real_360[mani_h, mani_w, :]  # 360 에 int visual 을 whole 로 추가

            # 472 int data 만드는 함수
            [self._int_pos, self._int_vis] = self.get_pos_vis_int_image(self._int)

        # == 전체 이미지 부분
        rgb_org = cv2.resize(self.rgb_crop_re360.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        whole_state_img = cv2.resize(self._whole_state.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)

        # == 논문용 / int ext
        int_org = cv2.resize(self._int.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        ext_org = cv2.resize(self._ext_org.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)

        # == VAE 이미지 부분
        ext_attend_pose_img = cv2.resize(self._ext_attend_pos.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        ext_vis_pad15re_img = cv2.resize(self.ext_vis_re128pad15.copy(), (128, 128),
                                         interpolation=cv2.INTER_CUBIC)  # - v2 - 128 128 로 리사이징된 15pad ext vis
        int_pose_img = cv2.resize(self._int_pos.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)
        int_vis_img = cv2.resize(self._int_vis.copy(), (128, 128), interpolation=cv2.INTER_CUBIC)

        rgb_org_ = rgb_org[..., 0:3]
        whole_state_img_ = whole_state_img[..., 0:3]
        int_org_ = int_org[..., 0:3]
        ext_org_ = ext_org[..., 0:3]
        ext_attend_pose_img_ = ext_attend_pose_img[..., 0:3]
        ext_vis_pad15re_img_ = ext_vis_pad15re_img[..., 0:3]
        int_pose_img_ = int_pose_img[..., 0:3]
        int_vis_img_ = int_vis_img[..., 0:3]

        return rgb_org_, whole_state_img_, ext_attend_pose_img_, ext_vis_pad15re_img_, \
            int_pose_img_, int_vis_img_, int_org_, ext_org_

    def get_internal_state(self):
        init_state = []
        gripper_state = 0.0
        if self.UR.is_gripper_closed():
            gripper_state = 1.0
        z_value = self.UR.get_end_effector_state()[0][2] - self.get_tray_position()[2]

        init_state.append(gripper_state)
        init_state.append(z_value)
        return init_state

    def _step(self, action=None, block_uid=None, collision_check=False, use_inverse_kinematics=False):
        done = False

        for i in range(self._action_repeat):
            if action:
                self.UR.apply_action(action, use_inverse_kinematics)
            if collision_check:  # : 엔드이펙터의 충돌확인
                effector_state = self.UR.get_end_effector_state()  # : 엔드이펙터의 포지션, 벨로시티 등 을 받음
                if effector_state[0][2] < 0.0:
                    break
        if self._renders:
            time.sleep(self._timeStep)

        if collision_check:
            effector_state = self.UR.get_end_effector_state()
            if effector_state[0][2] < 0.0:
                done = True
        else:
            done = False

        self._perform_time_step()

        # reward check
        reward = -0.05
        is_success, is_exception = self._is_grasp_success(self._success_check_height, block_uid)
        if is_success and is_exception:
            reward = 1.0

        return done, reward

    def move_robot_exact_position(self, action, block_uid, collision_check=False, tol_dist=0.02):
        dest_pos = np.array([action[0], action[1], action[2]])
        dest_ang = np.array([])
        dest_ang_quaternion = np.array([])
        reward = -0.05

        if len(action) > 3:
            dest_ang = np.array([action[3], action[4], action[5]])
            dest_ang_quaternion = np.array(self._p.getQuaternionFromEuler(dest_ang))

        for i in range(self._STEP_LIMIT):
            done, reward = self._step(action, block_uid, collision_check, True)
            if done:
                return True, reward

            effector_state = self.UR.get_end_effector_state()
            effector_pos = np.array(effector_state[0])
            effector_rot = effector_state[1]

            dist = effector_pos - dest_pos
            dist = np.linalg.norm(dist)
            dist_ang = 0
            sum_ang = 0
            if len(dest_ang) != 0:
                dist_ang = effector_rot - dest_ang_quaternion
                sum_ang = effector_rot + dest_ang_quaternion
                dist_ang = np.linalg.norm(dist_ang)
                sum_ang = np.linalg.norm(sum_ang)

            if dist <= tol_dist and (dist_ang < self._EPSILON or sum_ang < self._EPSILON):
                for _ in range(self._STEP_LIMIT):
                    _, vel = self._get_state()  # : UR의 state 를 반환
                    if all(abs(x) < self._EPSILON for x in vel):
                        break
                break

        return False, reward

    def _termination(self):  # : 터미네이션 flag 에 따라 image 랜더링 실행
        if self._terminated:  # : 터미네이션 flag 판별
            return True  # : 터메네이션 성공시 True 반환
        closest_points = p.getContactPoints(self.UR.tray_uid, self.UR.UR_uid)  # : 매니퓰레이터와 트레이 접점 확인

        # tray 와 robot 이 일정 거리이상 가까워지면 Terminate 시킨다
        if len(closest_points):  # : 접점이 하나라도 있다면 터미네이션
            self._terminated = 1  # : 터미네이션 flag 1
            return True

        if not self.UR.check_safety_zone():  # : 엔드 포즈가 Safety 를 벗어날시
            return True  # : 터미네이션 True

        return False

    def _action_n_get_final_state(self, action):
        done = False
        b_vel_end = False
        b_pos_diff_end = False
        start_time = -1

        for i in range(self._STEP_LIMIT):
            self.UR.apply_action(action)  # : action 을 UR 로봇에 적용함
            self._step()
            pos, vel = self._get_state()  # : UR의 state 를 반환

            done = False
            if done:
                break

            action_pos_diff = []
            for x in range(len(pos)):
                action_pos_diff.append(abs(pos[x] - action[x]))

            if all(abs(x) < self._EPSILON for x in vel):
                b_vel_end = True
            if all(action_pos_diff[x] < 0.05 for x in range(len(action_pos_diff))):
                b_pos_diff_end = True

            if b_vel_end and b_pos_diff_end:
                break
            elif b_vel_end is False and b_pos_diff_end is False:
                start_time = -1
            elif b_vel_end or b_pos_diff_end:
                if start_time > 0:
                    runtime = time.time() - start_time
                    if runtime > 5:
                        break
                else:
                    start_time = time.time()

        return done

    def _get_state(self):
        self._perform_time_step(self._timeStep)
        return self.UR.get_state()  # : UR의 state 를 반환

    def _draw_safe_zone(self):
        safe_zone_volume = self.UR.calc_safe_zone_vol()
        bounding_box_point = self.UR.bounding_box_point
        if safe_zone_volume > 0:
            box_color = [0.5, 0.5, 0.5]
            f = [bounding_box_point[0][0], bounding_box_point[0][1], bounding_box_point[0][2]]
            t = [bounding_box_point[1][0], bounding_box_point[0][1], bounding_box_point[0][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [bounding_box_point[0][0], bounding_box_point[0][1], bounding_box_point[0][2]]
            t = [bounding_box_point[0][0], bounding_box_point[1][1], bounding_box_point[0][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [bounding_box_point[0][0], bounding_box_point[0][1], bounding_box_point[0][2]]
            t = [bounding_box_point[0][0], bounding_box_point[0][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[0][0], bounding_box_point[0][1], bounding_box_point[1][2]]
            t = [bounding_box_point[0][0], bounding_box_point[1][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[0][0], bounding_box_point[0][1], bounding_box_point[1][2]]
            t = [bounding_box_point[1][0], bounding_box_point[0][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[1][0], bounding_box_point[0][1], bounding_box_point[0][2]]
            t = [bounding_box_point[1][0], bounding_box_point[0][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[1][0], bounding_box_point[0][1], bounding_box_point[0][2]]
            t = [bounding_box_point[1][0], bounding_box_point[1][1], bounding_box_point[0][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[1][0], bounding_box_point[1][1], bounding_box_point[0][2]]
            t = [bounding_box_point[0][0], bounding_box_point[1][1], bounding_box_point[0][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[0][0], bounding_box_point[1][1], bounding_box_point[0][2]]
            t = [bounding_box_point[0][0], bounding_box_point[1][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)

            f = [bounding_box_point[1][0], bounding_box_point[1][1], bounding_box_point[1][2]]
            t = [bounding_box_point[0][0], bounding_box_point[1][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [bounding_box_point[1][0], bounding_box_point[1][1], bounding_box_point[1][2]]
            t = [bounding_box_point[1][0], bounding_box_point[0][1], bounding_box_point[1][2]]
            p.addUserDebugLine(f, t, box_color)
            f = [bounding_box_point[1][0], bounding_box_point[1][1], bounding_box_point[1][2]]
            t = [bounding_box_point[1][0], bounding_box_point[1][1], bounding_box_point[0][2]]

            p.addUserDebugLine(f, t, box_color)

    def _get_init_pos(self):
        return self.UR.initPos

    def get_tray_position(self):
        return self._tray_pos

    def remove_target_object(self, block_uid):
        # 오브젝트 제거 함수
        for i in range(len(self.block_uid)):
            _id = self.block_uid[i][0]
            if _id == block_uid:
                p.changeVisualShape(self.block_uid[i][0], -1, rgbaColor=[0, 0, 0, 0])  # 이미지 렌더링 해결 코드
                self._open_close_gripper(False)
                p.removeBody(self.block_uid[i][0])
                del self.block_uid[i]
                break

    @staticmethod
    def _is_grasp_success(height, block_uid):  # 그래스핑 성공 여부

        exception = False
        if block_uid is not None:
            try:
                obj_state = p.getBasePositionAndOrientation(block_uid)  # 오류지점: 파묻혀서 오리엔테이션 사라지는 경우
                obj_height = obj_state[0][2]
                if obj_height > height:
                    return True, not exception
            except:
                print('\x1b[1;33m' + "-->>sys: Object's Orientation Not Detected !!" + '\x1b[0;m')
                return False, exception

        return False, not exception

    def get_obj_position_block_uid(self):
        obj_pos = []
        block_uids = []

        try:
            for i in self.block_uid:
                obj_state = p.getBasePositionAndOrientation(i[0])
                obj_pos.append(obj_state[0])
                block_uids.append(i[0])
            return obj_pos, block_uids

        except:
            print('\x1b[1;31m' + "!!>>sys: Object's Orientation Not Detected !! ----해결필요----" + '\x1b[0;m')
            return obj_pos, block_uids

    def _perform_time_step(self, sec=None):  # 시간 측정 함수
        if sec is None:  # parameter 를 지정 하지 않아주면
            sec = self._timeStep  # 시뮬레이션 time step 을 기준으로 설정

        for i in range(int(self._jump_time / sec)):  # : 점프타임을 타임스텝으로 나누었을때, 횟수만큼
            p.stepSimulation()  # : Step the simulation using forward dynamics.

    def get_unique_id_by_uid(self, uid):
        unique_id = -1
        for _id in self.block_uid:
            if uid == _id[0]:
                unique_id = _id[1] + 4
                break
        return unique_id

    def _mapping_unique_id(self, seg):
        num_obj = len(self.block_uid)
        if num_obj > 0:
            base_id = 4

            seg_temp = seg.copy()
            for _id in self.block_uid:
                # seg_temp = 8 -> 9
                seg_temp[seg == _id[0]] = _id[1] + base_id
            seg = seg_temp

        return seg

    def _open_close_gripper(self, b_open_close):
        # True : Close / False : Open
        done = False
        start_time = -1
        finger_max_force = self.UR.finger_force
        finger_val = 1.0 if b_open_close else 0
        gripper_epsilon = 0.01

        for i in range(self._STEP_LIMIT):
            self.UR.move_finger(finger_val)  # True 일때 닫고 False 일때 열기
            self._step()  # action 수행

            pos, vel, jnt_torque = self.UR.get_gripper_state()  # 그리퍼 joint 들의 angle, 각속도, 토크 확인

            done = self._termination()  # action 마침 확인
            if done:  # action 이 끝났을때 break
                break
            if all(abs(x) < gripper_epsilon for x in vel):
                # action 수행 시 체크한 joint 들의 속도가 설정 이하로 작을 시 break
                time.sleep(0.1)
                break
            elif any(abs(tq) > (finger_max_force - 1) for tq in jnt_torque):
                # action 수행 시 체크한 joint 들의 토크가 설정 이상으로 클 시 break
                time.sleep(0.1)
                break
            else:  # action 수행 시 수행한 시간이 5초이상 일 시
                if start_time > 0:
                    runtime = time.time() - start_time
                    if runtime > 5:
                        break
                else:
                    start_time = time.time()

        return done

    def check_object_safe(self):
        # object bounding box must be exist inside tray bounding box
        # = 트레이 내, 오브젝트가 어느것이라도 있는지 확인
        is_safe = False
        if len(self.block_uid) != 0:
            # all object check bounding box in tray
            trayAABB = self._p.getAABB(self.UR.tray_uid)  # : 트레이 시뮬레이션 상 위치 확인
            for _id in self.block_uid:
                objAABB = self._p.getAABB(_id[0])  # : 오브젝트 시뮬레이션 상 위치 확인

                if trayAABB[0][0] < objAABB[0][0] and objAABB[1][0] < trayAABB[1][0]:  # check x
                    if trayAABB[0][1] < objAABB[0][1] and objAABB[1][1] < trayAABB[1][1]:  # check y
                        is_safe = True
                        break

        return is_safe
