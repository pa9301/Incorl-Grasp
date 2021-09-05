import inspect
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)

import pybullet as p
import numpy as np
import math


class UR:
    initPos = list([0, -np.pi / 3 * 2, np.pi / 2, 0, 0, 0, 0])

    def __init__(self, safety_point1=np.array([0, 0, 0]), safety_point2=np.array([0, 0, 0])):
        self.UR_end_effector_index = 6
        self.finger_force = 50.  # 그리퍼 토크

        self._texture_list = self._load_texture(os.path.join(current_dir, 'tray/traytexture'))  # : 트레이 텍스쳐를 입힘
        self.bounding_box_point = []

        self._set_safety_zone(safety_point1, safety_point2)  # : tray 내 safe zone 형성

        self.motor_indices = []
        self.motor_names = []
        self.motor_lower_limit = []
        self.motor_upper_limit = []
        self.motor_max_force = []

        self.UR_uid = 0
        self.tray_uid = 0

        self.is_init_success = self.reset()  # : 리셋 성공여부 return

    def reset(self):
        # UR 로봇 초기화 함수
        self.motor_indices = []
        self.motor_names = []
        self.motor_lower_limit = []
        self.motor_upper_limit = []
        self.motor_max_force = []

        self.UR_uid = p.loadURDF(os.path.join(current_dir, "UR_Robot/robotiq_ur5.urdf"))
        if self.UR_uid < 0:
            return False

        self.tray_uid = p.loadURDF(os.path.join(current_dir, "tray/tray.urdf"),
                                   0.4625, -0.015, -0.1320,
                                   0.000000, 0.000000, 1.000000, 0.000000)  # : Tray 의 URDF Load # +가 카메라방향
        if self.tray_uid < 0:
            return False

        num_joints = p.getNumJoints(self.UR_uid)
        if self._texture_list:
            np.random.shuffle(self._texture_list)
            tid = self._texture_list[0]
            p.changeVisualShape(self.tray_uid, -1, textureUniqueId=tid)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.UR_uid, i)
            q_index = joint_info[3]
            if q_index > -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_indices.append(i)
                self.motor_lower_limit.append(joint_info[8])
                self.motor_upper_limit.append(joint_info[9])
                self.motor_max_force.append(30000)

        return True

    def get_tray_position(self):  # : Tray 의 x, y, z Pose 계산
        state = p.getBasePositionAndOrientation(self.tray_uid, 0)
        return state[0]

    def apply_action(self, motor_commands,
                     use_inverse_kinematics=False):  # : action 을 MotorCommands(x, y, z, roll, pitch, yaw)로 수행
        finger_angle = 0
        motor_value = []

        if use_inverse_kinematics:  # : IK를 사용할 시
            target_idx = 7
            rpy = [0, math.pi / 2, 0]

            # = motorCommands = x, y, z, gripper 혹은 x, y, z, roll, pitch, yaw, gripper 를 분할
            x = motor_commands[0]
            y = motor_commands[1]
            z = motor_commands[2]
            if len(motor_commands) == 4:  # : 4D command = x, y, z, gripper
                finger_angle = motor_commands[3]
            elif len(motor_commands) == 7:  # : 7D command = x, y, z, roll, pitch, yaw, gripper
                rpy = []
                for i in range(3, 6):  # : roll, pitch, yaw 를 따로 저장
                    rpy.append(motor_commands[i])
                finger_angle = motor_commands[6]

            pos = [x, y, z]
            orn = p.getQuaternionFromEuler(rpy)  # 일단 수직으로
            # : IK를 계산하여 각 UR의 Joint angle 계산
            joint_poses = p.calculateInverseKinematics(self.UR_uid, target_idx, pos, orn)

            for action in range(self.UR_end_effector_index):  # : 엔드이펙터의 인덱스까지 적용
                motor_value.append(joint_poses[action])

        else:  # : IK를 사용 안 할 시, (기본값)
            for action in range(self.UR_end_effector_index):  # 엔드이펙터의 인덱스까지 적용
                motor_upper = self.motor_upper_limit[action]  # 각 인덱스 별로 각도를 지정 UpperLimit
                motor_lower = self.motor_lower_limit[action]  # 각 인덱스 별로 각도를 지정 LowerLimit

                motor_value.append(motor_commands[action])
                if motor_value[action] < motor_lower:
                    motor_value[action] = motor_lower
                if motor_value[action] > motor_upper:
                    motor_value[action] = motor_upper

            if len(motor_commands) > self.UR_end_effector_index:
                finger_angle = motor_commands[self.UR_end_effector_index]

        for action in range(self.UR_end_effector_index):
            motor = self.motor_indices[action]
            p.setJointMotorControl2(self.UR_uid, motor, p.POSITION_CONTROL, targetPosition=motor_value[action],
                                    force=self.motor_max_force[action])
        self.move_finger(finger_angle)

    def get_state(self):
        pos_list = []
        vel_list = []
        for action in range(self.UR_end_effector_index):  # 엔드이펙터의 인덱스까지 적용
            motor = self.motor_indices[action]  # 엔드이펙터까지의 모터들
            pos, vel, _, _ = p.getJointState(self.UR_uid, motor)  # UR 내 엔드이펙터까지의 모터들의 angle 과 각속도를 저장

            pos_list.append(pos)  # 포즈리스트로 저장
            vel_list.append(vel)  # 속도리스트로 저장

        return pos_list, vel_list

    def get_gripper_state(self):
        pos_list = []
        vel_list = []
        jnt_torque = []
        gripper_check_list = [9, 14]

        for _id in gripper_check_list:
            pos, vel, _, jnt_motor_torque = p.getJointState(self.UR_uid, _id)

            pos_list.append(pos)
            vel_list.append(vel)
            jnt_torque.append(jnt_motor_torque)
        return pos_list, vel_list, jnt_torque

    def is_gripper_closed(self):
        pos, vel, jnt_torque = self.get_gripper_state()
        finger_max_force = self.finger_force

        if any(abs(tq) > (finger_max_force - 1) for tq in jnt_torque):
            return True
        if all(abs(angle) < 0.0001 for angle in pos):
            return False

        return False

    def get_end_effector_state(self):  # 앤드 이펙터의 link 상태를 반환 (각속도 속도 회전 등)
        return p.getLinkState(self.UR_uid, 7)

    # 손가락 오므리기
    def move_finger(self, angle):

        # 왼쪽그리퍼 모터 제어, 제어 부위 및 토크설정
        # Left 9,11(7),13 Right : 14, 16(10), 18
        # Left
        left_lower = self.motor_lower_limit[self.UR_end_effector_index]
        left_upper = self.motor_upper_limit[self.UR_end_effector_index]
        left_angle = left_lower + (left_upper - left_lower) * angle
        left_middle_angle = self._calc_middle_finger(left_angle, self.UR_end_effector_index, 3.5)

        p.setJointMotorControl2(self.UR_uid, 9, p.POSITION_CONTROL, targetPosition=left_angle, force=self.finger_force)
        p.setJointMotorControl2(self.UR_uid, 11, p.POSITION_CONTROL, targetPosition=left_middle_angle,
                                force=self.finger_force)
        p.setJointMotorControl2(self.UR_uid, 13, p.POSITION_CONTROL, targetPosition=-left_middle_angle,
                                force=self.finger_force)

        # 오른쪽 그리퍼 모터 제어, 제어 부위 및 토크설정
        right_lower = self.motor_lower_limit[self.UR_end_effector_index + 3]
        right_upper = self.motor_upper_limit[self.UR_end_effector_index + 3]
        right_angle = right_lower + (right_upper - right_lower) * angle
        right_middle_angle = self._calc_middle_finger(right_angle, self.UR_end_effector_index + 3, 4)

        p.setJointMotorControl2(self.UR_uid, 14, p.POSITION_CONTROL, targetPosition=right_angle, force=self.finger_force)
        p.setJointMotorControl2(self.UR_uid, 16, p.POSITION_CONTROL, targetPosition=right_middle_angle,
                                force=self.finger_force)
        p.setJointMotorControl2(self.UR_uid, 18, p.POSITION_CONTROL, targetPosition=-right_middle_angle,
                                force=self.finger_force)

    def _calc_middle_finger(self, base_angle, idx, coefficient):
        next_idx = idx + 1
        return \
            - (base_angle - self.motor_lower_limit[idx]) \
            / (self.motor_upper_limit[idx] - self.motor_lower_limit[idx]) \
            * (self.motor_upper_limit[next_idx] / coefficient)

    def _set_safety_zone(self, point1, point2):  # tray 내 safe 존 형성
        self.bounding_box_point.append(point1)  # bounding_box_point 변수 내에 둘다 이어붙임
        self.bounding_box_point.append(point2)

    def calc_safe_zone_vol(self):  # 세이프 존 크기 검사
        volume_vector = self.bounding_box_point[0] - self.bounding_box_point[1]
        return abs(np.dot(volume_vector, np.array([1, 1, 1])))

    def check_safety_zone(self):
        b_violate_safety = False
        volume = self.calc_safe_zone_vol()  # 세이프 존 크기검사시

        # Volume is zero, so always return True
        if volume <= 0:  # 크기가 0보다 작다면, True 반환
            return True

        pos, _ = self.get_state()  # UR의 state 를 반환
        x1 = min(self.bounding_box_point[0][0], self.bounding_box_point[1][0])  # 세이프존의 X
        x2 = max(self.bounding_box_point[0][0], self.bounding_box_point[1][0])
        y1 = min(self.bounding_box_point[0][1], self.bounding_box_point[1][1])  # 세이프존의 Y
        y2 = max(self.bounding_box_point[0][1], self.bounding_box_point[1][1])
        z1 = min(self.bounding_box_point[0][2], self.bounding_box_point[1][2])  # 세이프존의 Z 좌표를 계산
        z2 = max(self.bounding_box_point[0][2], self.bounding_box_point[1][2])

        end_effector_pos = p.getLinkState(self.UR_uid, self.UR_end_effector_index)  # 링크의 FK, 속도 계산
        end_effector_pos = end_effector_pos[0]  # 0번쨰 원소(FK후 x,y,z 좌표)를 저장
        if x1 < end_effector_pos[0] < x2:  # 세이프 존의 범위를 넘지 않는지 계산
            if y1 < end_effector_pos[1] < y2:
                if z1 < end_effector_pos[2] < z2:
                    b_violate_safety = True  # 바이올런스 미확인시 True

        return b_violate_safety  # : 바이올런시 비위반 True 위반 False

    @staticmethod
    def _load_texture(path):  # : Tray 의 텍스쳐를 불러들여옴
        texture_list = []

        filenames = os.listdir(path)
        for filename in filenames:
            full_filename = os.path.join(path, filename)
            _id = p.loadTexture(full_filename)
            if _id >= 0:
                texture_list.append(_id)

        return texture_list
