import random
import numpy as np
import math


class ScriptedPolicy:
    """Scripted policy 생성 클래스."""

    APPROACH_STEP = 3

    def __init__(self, env):  # : scripted policy 를 environment 에 맞춰서 초기화
        # using bellow parameter creating scripted policy
        self.Z_VALUE = 0.40  # 40cm

        self._env = env
        self._tray_pos = list(self._env.get_tray_position())   # 트레이 포즈를 저장

        # Step policy variable
        self._step_count = 0
        self._init_xyz_robo_pos = None
        self.target_obj_z_val = None

        tray_uid_num = env.UR.tray_uid
        self.tray_AABB = env._p.getAABB(tray_uid_num)

        self.rot = 0
        self.approach_pos = []

    def get_scripted_policy(self):
        action = []
        action_ext = []
        scripted_end = False
        block_uid = -1
        unique_uid = -1
        out_of_tray = False
        eps = 0.00001

        # 스텝의 처음 시작시
        if self._step_count == 0:
            # 1. Move UR5 arms over an object.
            self._init_xyz_robo_pos, self.target_obj_z_val, block_uid, unique_uid, out_of_tray \
                = self._initial_pos_sampling()

            x_pos = 0.52
            y_pos = -0.18

            self.rot = math.pi * (np.random.uniform() - 0.5)
            self._init_xyz_robo_pos = self._init_xyz_robo_pos + [0.0, math.pi / 2.0, 0.0] + [0]
            action = list(self._init_xyz_robo_pos)
            action[0] = x_pos
            action[1] = y_pos
            gripper_open = (np.random.uniform() / 2.0) + 0.5
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

        # 스텝이 진행 중이면서 다가가기 중일때
        elif 0 < self._step_count <= self.APPROACH_STEP:
            # 3. z approach ( N step )
            if self._step_count == 1:
                approach_action = self._init_xyz_robo_pos.copy()
                approach_action[3] = self.rot
                approach_depth = 0.254 - abs(self.target_obj_z_val)  # self.Z_VALUE * 0.5

                rand_num = np.random.randint(0, 10)
                if rand_num < 2:
                    approach_noise = np.random.normal() / 50.0
                    approach_depth += approach_noise

                approach_depth = min(approach_depth, 0.237)

                approach_ratio = self._separate_randomly(approach_depth, self.APPROACH_STEP)
                approach_ratio = approach_ratio.tolist()
                self.approach_pos = []

                approach_ratio_sum = 0.0
                for i in range(self.APPROACH_STEP):
                    approach_action_temp = approach_action.copy()
                    approach_ratio_sum += approach_ratio[i]
                    if i > 0:
                        approach_action_temp[2] -= approach_ratio_sum
                    self.approach_pos.append(approach_action_temp)

            action = self.approach_pos[self._step_count - 1]
            gripper_open = (np.random.uniform() / 2.0) + 0.5
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

        # 다가가기 스텝이 끝난 직후
        elif self._step_count == self.APPROACH_STEP + 1:
            eff_state = self._env.UR.get_end_effector_state()  # 4. grasp
            action = list(eff_state[0]) + list(self._env._p.getEulerFromQuaternion(eff_state[1])) + [1]

            gripper_open = (np.random.uniform() / 2.0) - eps
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

        # 다가가기 스텝이 끝난 뒤
        elif self.APPROACH_STEP + 2 <= self._step_count <= self.APPROACH_STEP * 2 + 1:
            # 들어올리기
            if self._step_count == self.APPROACH_STEP + 2:
                eff_state = self._env.UR.get_end_effector_state()
                approach_depth = self._init_xyz_robo_pos[2] - eff_state[0][2]
                approach_ratio = self._separate_randomly(approach_depth, self.APPROACH_STEP)
                approach_ratio = approach_ratio.tolist()
                self.approach_pos = []
                approach_action = list(eff_state[0]) + list(self._env._p.getEulerFromQuaternion(eff_state[1])) + [1]
                approach_ratio_sum = 0.0
                for i in range(self.APPROACH_STEP):
                    approach_action_temp = approach_action.copy()
                    approach_ratio_sum += approach_ratio[i]
                    approach_action_temp[2] += approach_ratio_sum
                    self.approach_pos.append(approach_action_temp)

            action = self.approach_pos[self._step_count - self.APPROACH_STEP - 2]
            gripper_open = (np.random.uniform() / 2.0) - eps
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

            if self._step_count == self.APPROACH_STEP * 2 + 1:
                gripper_open = (np.random.uniform() / 2.0) - eps
                gripper_close = 1.0 - gripper_open
                terminal = (np.random.uniform() / 2.0) - eps
                action_ext = [gripper_close, gripper_open, terminal]
                scripted_end = True

        # scriptedPolicy 가 끝나면
        if scripted_end:
            self._step_count = 0     # step flag 설정
        else:
            self._step_count += 1    # scriptedPolicy 진행중 step flag count
        return action, action_ext, block_uid, unique_uid, scripted_end, out_of_tray

    def _initial_pos_sampling(self):
        unique_uid = -1
        block_uid = - 1
        z_pos = 0
        sel_pos = []

        while True:
            obj_pos_list, block_uid_s = self._env.get_obj_position_block_uid()
            num_obj_pos_list = len(obj_pos_list)

            if num_obj_pos_list < 1:
                out_of_tray = True
                break

            index = np.random.randint(0, num_obj_pos_list)
            sel_pos = obj_pos_list[index]
            block_uid = block_uid_s[index]
            unique_uid = self._env.get_unique_id_by_uid(block_uid)

            x_pos = sel_pos[0]
            y_pos = sel_pos[1]
            true_z_pos = sel_pos[2]
            z_pos = abs(self._tray_pos[2]) - abs(sel_pos[2])
            if self.tray_AABB[0][0] < x_pos < self.tray_AABB[1][0]:
                if self.tray_AABB[0][1] < y_pos < self.tray_AABB[1][1]:
                    # z축 포즈를 이용, 트레이 밑으로 빠지는것까지 체크
                    if self.tray_AABB[0][2] < true_z_pos < 0.5:
                        out_of_tray = False
                        break

            _id = self._env.block_uid[index][0]
            self._env.remove_target_object(_id)
            block_uid = -1

        if not out_of_tray:
            sample_pos = [sel_pos[0], sel_pos[1], self._tray_pos[2] + self.Z_VALUE]
        else:
            sample_pos = [0, 0, self._tray_pos[2] + self.Z_VALUE]

        return sample_pos, z_pos, block_uid, unique_uid, out_of_tray

    @staticmethod
    def _separate_randomly(value, n):
        score_sum = 0
        score_list = []
        for i in range(n):
            rand_val = random.random()
            score_list.append(rand_val)
            score_sum += rand_val

        score_list = np.array(score_list)
        score_list /= score_sum
        ratio = score_list * value

        return ratio

    def reset(self):
        self._step_count = 0
