import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import math
import pybullet_data

GRIPPER_EPSILON = 0.01


class UR:
    initPos = list([0, -np.pi/3 * 2, np.pi/2, 0, 0, 0, 0])

    def __init__(self, urdfRootPath,
                 safetypoint1=np.array([0, 0, 0]),
                 safetypoint2=np.array([0, 0, 0])):
        self.urdfRootPath = urdfRootPath                        ## 안쓰고있음
        self.UREndEffectorIndex = 6
        self.maxForce = 200.                    #이게 maxforce값이 맞나 확인해봐야함    #### ???? // 안쓰고있음
        self.fingerForce = 50.                                  # : 그리퍼 토크
        ##여기에 로봇 정보들 기록해야함
        self._texturelist = self.LoadTexture(os.path.join(currentdir, 'tray/traytexture'))  # : 트레이 텍스쳐를 입힘
        self._boundingboxPoint = []                             # :
        self._textid = -1

        self.setSafetyZone(safetypoint1, safetypoint2)          # : tray 내 safezone 형성
        self._isInitSuccess = self.reset()                      # : 리셋 성공여부 return

    def reset(self):    # : UR 로봇 초기화 함수
        self.motorIndices = []
        self.motorNames = []
        self.motorLowerLimit = []
        self.motorUpperLimit = []
        self.motorMaxForce = []

        self.endEffectorAngle = 0

        # self.URUid = p.loadURDF(os.path.join(currentdir, "UR_Robot/robotiq_ur3.urdf"))   # : UR3 이용시
        self.URUid = p.loadURDF(os.path.join(currentdir, "UR_Robot/robotiq_ur5.urdf"))     # : UR5 이용시
        if self.URUid < 0:
            return False

        # self.trayUid = p.loadURDF(os.path.join(currentdir, "tray/tray.urdf"), 0.457, 0.0, -0.15500,
        #                            0.000000, 0.000000, 1.000000, 0.000000)                  # : Tray의 URDF Load
        self.trayUid = p.loadURDF(os.path.join(currentdir, "tray/tray.urdf"), 0.4625, -0.015, -0.1320,
                                  0.000000, 0.000000, 1.000000, 0.000000)  # : Tray의 URDF Load # +가 카메라방향
        # self.trayUid = p.loadURDF(os.path.join(currentdir, "tray/tray.urdf"), 0.0, -0.0, -0.0,
        #                           0.000000, 0.000000, 1.000000, 0.000000)  # : Tray의 원래 포즈 확인용
        if self.trayUid < 0:
            return False

        self.numJoints = p.getNumJoints(self.URUid)
        if self._texturelist:
            np.random.shuffle(self._texturelist)
            tid = self._texturelist[0]
            p.changeVisualShape(self.trayUid, -1, textureUniqueId=tid)

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.URUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)
                self.motorLowerLimit.append(jointInfo[8])
                self.motorUpperLimit.append(jointInfo[9])
                self.motorMaxForce.append(30000)

        return True

    def getTrayPosition(self):              # : Tray 의 x, y, z Pose 계산
        state = p.getBasePositionAndOrientation(self.trayUid, 0)
        return state[0]

    def getInitPos(self):
        return self.initPos

    ## 안쓰고있음 (아예 안만들어짐)
    # def getActionDimension(self):
    #
    #     return

    def applyAction(self, motorCommands, useInverseKinematics=False):   # : action을 MotorCommands(x, y, z, roll, pitch, yaw)로 수행
        fingerAngle = 0
        motorValue = []

        if useInverseKinematics:        # : IK를 사용할 시
            targetIdx = 7
            orn = []
            rpy = [0, math.pi/2, 0]

            # = motorCommands = x, y, z, gripper 혹은 x, y, z, roll, pitch, yaw, gripper를 분할
            x = motorCommands[0]
            y = motorCommands[1]
            z = motorCommands[2]
            if len(motorCommands) == 4:         # : 4D command = x, y, z, gripper
                fingerAngle = motorCommands[3]
            elif len(motorCommands) == 7:       # : 7D command = x, y, z, roll, pitch, yaw, gripper
                rpy = []
                for i in range(3, 6):           # : roll, pitch, yaw 를 따로 저장
                    rpy.append(motorCommands[i])
                fingerAngle = motorCommands[6]

            pos = [x, y, z]
            orn = p.getQuaternionFromEuler(rpy)  # -math.pi,yaw]) #일단 수직으로
            jointPoses = p.calculateInverseKinematics(self.URUid, targetIdx, pos, orn) # : IK를 계산하여 각 UR의 Joint angle 계산

            for action in range(self.UREndEffectorIndex):      # : 엔드이펙터의 인덱스까지 적용
                motorValue.append(jointPoses[action])

        else:                           # : IK를 사용 안 할 시, (기본값)
            for action in range(self.UREndEffectorIndex):      # : 엔드이펙터의 인덱스까지 적용
                motorUpper = self.motorUpperLimit[action]       # : 각 인덱스 별로 각도를 지정 UpperLimit
                motorLower = self.motorLowerLimit[action]       # : 각 인덱스 별로 각도를 지정 LowerLimit

                motorValue.append(motorCommands[action])
                if motorValue[action] < motorLower:
                    motorValue[action] = motorLower
                if motorValue[action] > motorUpper:
                    motorValue[action] = motorUpper

            if len(motorCommands) > self.UREndEffectorIndex:
                fingerAngle = motorCommands[self.UREndEffectorIndex]

        for action in range(self.UREndEffectorIndex):
            motor = self.motorIndices[action]
            p.setJointMotorControl2(self.URUid, motor, p.POSITION_CONTROL, targetPosition=motorValue[action],
                                    force=self.motorMaxForce[action])
        self.MoveFinger(fingerAngle)

        state = self.getEndEffectorState()
        actualEndEffectorPos = state[0]

    def getState(self):
        pos_list = []
        vel_list = []
        for action in range(self.UREndEffectorIndex):              # : 엔드이펙터의 인덱스까지 적용
            motor = self.motorIndices[action]                       # : 엔드이펙터까지의 모터들
            pos, vel, _, _ = p.getJointState(self.URUid, motor)    # : UR 내 엔드이펙터까지의 모터들의 angle과 각속도를 저장

            pos_list.append(pos)                                    # : 포즈리스트로 저장
            vel_list.append(vel)                                    # : 속도리스트로 저장

        return pos_list, vel_list

    def getGripperState(self):
        pos_list = []
        vel_list = []
        jntTorque = []
        GripperCheckList = [9, 14]

        for id in GripperCheckList:
            pos, vel, reactionForce, jntMoterTorque = p.getJointState(self.URUid, id)

            pos_list.append(pos)
            vel_list.append(vel)
            jntTorque.append(jntMoterTorque)
        return pos_list, vel_list, jntTorque

    def IsGripperCloesed(self):
        pos, vel, jntTorque = self.getGripperState()
        FingerMaxForce = self.fingerForce

        # if all(abs(x) < GRIPPER_EPSILON for x in vel):    # : 그리퍼 속도에 상관없이 힘들어가면 닫았다고한다고함 ...?
        if any(abs(tq) > (FingerMaxForce - 1) for tq in jntTorque):
            return True
        if all(abs(angle) < 0.0001 for angle in pos):
            return False

        return False


    def getEndEffectorState(self):  # : 앤드 이펙터의 link상태를 반환 (각속도 속도 회전 등)
        return p.getLinkState(self.URUid, 7)

    # 손가락 오므리기
    def MoveFinger(self, Angle):

        # = 왼쪽그리퍼 모터 제어, 제어 부위 및 토크설정
        #Left 9,11(7),13 Right : 14, 16(10), 18
        #Left
        LeftLower = self.motorLowerLimit[self.UREndEffectorIndex]
        LeftUpper = self.motorUpperLimit[self.UREndEffectorIndex]
        LeftAngle = LeftLower + (LeftUpper - LeftLower) * Angle
        Left_middle_angle = self._CalcMiddleFinger(LeftAngle, self.UREndEffectorIndex, 3.5)

        p.setJointMotorControl2(self.URUid, 9, p.POSITION_CONTROL, targetPosition=LeftAngle, force=self.fingerForce)
        p.setJointMotorControl2(self.URUid, 11, p.POSITION_CONTROL, targetPosition=Left_middle_angle,
                                force=self.fingerForce)
        p.setJointMotorControl2(self.URUid, 13, p.POSITION_CONTROL, targetPosition=-Left_middle_angle,
                                force=self.fingerForce)

        # = 오른쪽 그리퍼 모터 제어, 제어 부위 및 토크설정
        #Rightd
        RightLower = self.motorLowerLimit[self.UREndEffectorIndex + 3]
        RightUpper = self.motorUpperLimit[self.UREndEffectorIndex + 3]
        RightAngle = RightLower + (RightUpper - RightLower) * Angle
        Right_middle_angle = self._CalcMiddleFinger(RightAngle, self.UREndEffectorIndex + 3, 4)

        p.setJointMotorControl2(self.URUid, 14, p.POSITION_CONTROL, targetPosition=RightAngle, force=self.fingerForce)
        p.setJointMotorControl2(self.URUid, 16, p.POSITION_CONTROL, targetPosition=Right_middle_angle,
                                force=self.fingerForce)
        p.setJointMotorControl2(self.URUid, 18, p.POSITION_CONTROL, targetPosition=-Right_middle_angle,
                                force=self.fingerForce)

    def _CalcMiddleFinger(self, baseAngle, idx, coef):
        NextIdx = idx + 1
        return - (baseAngle - self.motorLowerLimit[idx]) / (self.motorUpperLimit[idx] - self.motorLowerLimit[idx]) * \
               (self.motorUpperLimit[NextIdx] / coef)

    # Name, Lower, Upper
    def getRobotMotorInfo(self):                            ## 안쓰고있음 (getRobotInfo)
        return self.motorNames, self.motorLowerLimit, self.motorUpperLimit

    def setSafetyZone(self, point1, point2):    # : tray 내 safe 존 형성
        self._boundingboxPoint.append(point1)               # : _boundingboxPoint 변수 내에 둘다 이어붙임
        self._boundingboxPoint.append(point2)

    def calcSafeZoneVol(self):                              # : 세이프 존 크기 검사
        volume_vector = self._boundingboxPoint[0] - self._boundingboxPoint[1]
        return abs(np.dot(volume_vector, np.array([1, 1, 1])))

    def getSafePoint(self):
        return self._boundingboxPoint

    def CheckSafetyZone(self):
        bViolateSafety = False
        volume = self.calcSafeZoneVol()                     # : 세이프 존 크기검사시

        # Volume is zero, so always return True
        if volume <= 0:                                     # : 크기가 0보다 작다면, True 반환
            return True

        pos, _ = self.getState()                            # : UR의 state 를 반환
        x1 = min(self._boundingboxPoint[0][0], self._boundingboxPoint[1][0])        # : 세이프존의 X
        x2 = max(self._boundingboxPoint[0][0], self._boundingboxPoint[1][0])
        y1 = min(self._boundingboxPoint[0][1], self._boundingboxPoint[1][1])        # : 세이프존의 Y
        y2 = max(self._boundingboxPoint[0][1], self._boundingboxPoint[1][1])
        z1 = min(self._boundingboxPoint[0][2], self._boundingboxPoint[1][2])        # : 세이프존의 Z 좌표를 계산
        z2 = max(self._boundingboxPoint[0][2], self._boundingboxPoint[1][2])

        end_effector_pos = p.getLinkState(self.URUid, self.UREndEffectorIndex)      # : 링크의 FK, 속도 계산
        end_effector_pos = end_effector_pos[0]                                      # : 0번쨰 원소(FK후 x,y,z 좌표)를 저장
        if x1 < end_effector_pos[0] < x2:                                           # : 세이프 존의 범위를 넘지 않는지 계산
            if y1 < end_effector_pos[1] < y2:
                if z1 < end_effector_pos[2] < z2:
                    bViolateSafety = True                                           # : 바이올런스 미확인시 True

        return bViolateSafety                               # : 바이올런시 비위반 True 위반 False

    def LoadTexture(self, path):    # : Tray의 텍스쳐를 불러들여옴
        list = []

        filenames = os.listdir(path)
        for filename in filenames:
            full_filename = os.path.join(path, filename)
            id = p.loadTexture(full_filename)
            if id >= 0:
                list.append(id)

        return list

