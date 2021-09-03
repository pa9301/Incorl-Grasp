"""Scripted policy 생성 클래스"""

import random
import numpy as np
import math
from simulator.UREnv import URGymEnv


class scriptedPolicy:

    APPROACH_STEP = 3

    def __init__(self, env, traypos=None):  # : scripted policy를 environment에 맞춰서 초기화
        # using bellow parameter creating scripted policy
        self.X_RANGE = 0.30  # 30cm
        self.Y_RANGE = 0.40
        self.Z_VALUE = 0.40

        self._env = env
        self._traypos = traypos
        if self._traypos is None:                               # : 트레이가 존재하지 않을때
            self._traypos = list(self._env.getTrayPosition())   # : 트레이 포즈를 저장  ## 확인필요, 두개의 같은이름 다른 선언이 존재
                                                                                        ## 현재는 UREnv 내로 돌아감을 확인

        #step policy variable
        self._stepcount = 0
        self._initpos = None
        self._initxyzrobopos = None
        self.target_obj_z_val = None

        ## v+ 추가
        tray_Uidnum = env._UR.trayUid  ## vv +
        self.tray_AABB = env._p.getAABB(tray_Uidnum)

    def RunScriptedPolicy(self):        ## 안쓰고있음
        dataset = []
        bInit = False

        while(True):
            done, data = self.stepScriptedPolicy()
            dataset += data

            if done:
                break
        if done:
            bInit = True

        return bInit, dataset

    # if collision detected or script end, return bIsDone = True
    def stepScriptedPolicy(self):       ## 안쓰고있음 (RunScriptedPolicy)
        bIsDone = False
        dataset = []

        action, action_ext, blockUid, _, scriptEnd, outOfTray = self.getScriptedPolicy()   ## 변경 _,추가 (action, action_ext, blockUid, uniqueUid, scriptedEnd, outOfTray) ## 수정 outOfTray -> _ : 실질적으로 안쓰는 변수
        collisionDetected, reward = self._env.MoveRobotExactPosition(action)

        if collisionDetected or scriptEnd:
            bIsDone = True
        if collisionDetected is False:
            rgb, depth, seg = self._env.getExtendedObservation()
            self.AppendData(dataset, rgb, seg, action, 0)

        return bIsDone, dataset

    def getScriptedPolicy(self):
        action = []
        action_ext = []
        scriptedEnd = False
        blockUid = -1
        uniqueUid = -1
        outOfTray = False
        eps = 0.00001

        if self._stepcount == 0:        # : 스텝의 처음 시작시
            # 1. Move UR5 arms over an object.
            self._initxyzrobopos, self.target_obj_z_val, blockUid, uniqueUid, outOfTray = self.InitialPosSampling()

            ## 확인필요 (바로 뒤에 후에 재선언됨)
            ## xpos = np.random.uniform(self._traypos[0] - 0.5 * self.X_RANGE, self._traypos[0] + 0.5 * self.X_RANGE, 1)
            ## ypos = np.random.uniform(self._traypos[1] - 0.5 * self.Y_RANGE, self._traypos[1] + 0.5 * self.Y_RANGE, 1)

            xpos = 0.52
            ypos = -0.18

            self.rot = math.pi * (np.random.uniform() - 0.5)
            self._initxyzrobopos = self._initxyzrobopos + [0.0, math.pi / 2.0, 0.0] + [0]
            action = list(self._initxyzrobopos)
            action[0] = xpos
            action[1] = ypos
            gripper_open = (np.random.uniform() / 2.0) + 0.5
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]
#            action_ext = [0, 1, 0]

        elif self._stepcount > 0 and self._stepcount <= self.APPROACH_STEP:     # : 스텝이 진행 중이면서 다가가기 중일때

            # 3. z approach ( N step )
            if self._stepcount == 1:
                approach_action = self._initxyzrobopos.copy()
                approach_action[3] = self.rot
                approachDepth = 0.254 - abs(self.target_obj_z_val)  # self.Z_VALUE * 0.5

                rand_num = np.random.randint(0, 10, 1)
                if rand_num[0] < 2:
                    approachNoise = np.random.normal() / 50.0
                else:
                    approachNoise = 0.0

                approachDepth += approachNoise

                if approachDepth > 0.237:
                    approachDepth = 0.237

                approach_ratio = self.SeperateRandomly(approachDepth, self.APPROACH_STEP)  # 어떤 기능을 하는 함수?
                approach_ratio = approach_ratio.tolist()
                self.approach_pos = []

                approach_ratio_sum = 0.0
                for i in range(self.APPROACH_STEP):
                    approach_action_temp = approach_action.copy()
                    approach_ratio_sum += approach_ratio[i]
                    if i > 0:
                        approach_action_temp[2] -= approach_ratio_sum
                    self.approach_pos.append(approach_action_temp)

            # approach_ratio를 보고 정해진 높이 만큼 내려왔으면 self._stepcount = self.APPROACH_STEP + 1 로 해준다  pyb
            action = self.approach_pos[self._stepcount - 1]
            gripper_open = (np.random.uniform() / 2.0) + 0.5
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]
#            action_ext = [0, 1, 0]
        elif self._stepcount == self.APPROACH_STEP + 1:                         # : 다가가기 스텝이 끝난 직후
            effstate = self._env._UR.getEndEffectorState()  # 4. grasp  pyb : 왜 6 X 3 차원인가?
            action = list(effstate[0]) + list(self._env._p.getEulerFromQuaternion(effstate[1])) + [1]

            gripper_open = (np.random.uniform() / 2.0) - eps
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]
#            action_ext = [1, 0, 0]

        elif self._stepcount >= self.APPROACH_STEP + 2 and self._stepcount <= self.APPROACH_STEP * 2 + 1:   # : 다가가기 스텝이 끝난뒤
            # 들어올리기
            if self._stepcount == self.APPROACH_STEP + 2:
                effstate = self._env._UR.getEndEffectorState()
                approachDepth = self._initxyzrobopos[2] - effstate[0][2]
                approach_ratio = self.SeperateRandomly(approachDepth, self.APPROACH_STEP)
                approach_ratio = approach_ratio.tolist()
                self.approach_pos = []
                approach_action = list(effstate[0]) + list(self._env._p.getEulerFromQuaternion(effstate[1])) + [1]
                approach_ratio_sum = 0.0
                for i in range(self.APPROACH_STEP):
                    approach_action_temp = approach_action.copy()
                    approach_ratio_sum += approach_ratio[i]
                    approach_action_temp[2] += approach_ratio_sum
                    self.approach_pos.append(approach_action_temp)

            action = self.approach_pos[self._stepcount - self.APPROACH_STEP - 2]
            gripper_open = (np.random.uniform() / 2.0) - eps
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]
 #           action_ext = [1, 0, 0]

            if self._stepcount == self.APPROACH_STEP * 2 + 1:
                gripper_open = (np.random.uniform() / 2.0) - eps
                gripper_close = 1.0 - gripper_open
                terminal = (np.random.uniform() / 2.0) - eps
                action_ext = [gripper_close, gripper_open, terminal]
#                action_ext = [1, 0, 0]
                scriptedEnd = True

        if scriptedEnd:             # : scriptedPolicy 가 끝나면
            self._stepcount = 0     # : step flag 설정
        else:
            self._stepcount += 1    # : scriptedPolicy 진행중 step flag count
        return action, action_ext, blockUid, uniqueUid, scriptedEnd, outOfTray

    def InitialPosSampling(self):
        outOfTray = True
        uniqueUid = -1
        blockUid = - 1
        zpos = 0

        while True:
            objposlist, blockUids = self._env.getObjPositionblockUid()
            num_objposlist = len(objposlist)

            if num_objposlist < 1:
                outOfTray = True
                break

            index = np.random.randint(0, num_objposlist, 1)
            selpos = objposlist[index[0]]
            blockUid = blockUids[index[0]]
            uniqueUid = self._env.getUniqueIDbyUID(blockUid)

            xpos = selpos[0]
            ypos = selpos[1]
            true_zpos = selpos[2]
            zpos = abs(self._traypos[2]) - abs(selpos[2])
            # if self._traypos[0] - 0.6 * self.X_RANGE < xpos < self._traypos[0] + 0.6 * self.X_RANGE:  ## 수정 0.5 -> 0.6 / RGB 순서로 XYZ
            #     if self._traypos[1] - 0.7 * self.Y_RANGE < ypos < self._traypos[1] + 0.7 * self.Y_RANGE:    ## 수정 0.5 -> 0.7
            #         outOfTray = False
            #         break

            ## v+ 20190902
            # = 물체가 safety zone 안에 들어있다면
            if self.tray_AABB[0][0] < xpos < self.tray_AABB[1][0]:
                if self.tray_AABB[0][1] < ypos < self.tray_AABB[1][1]:
                    if self.tray_AABB[0][2] < true_zpos < 0.5:  ## v+   # : z축 포즈를 이용, 트레이 밑으로 빠지는것까지 체크
                        outOfTray = False
                        break

            id = self._env.blockUid[index[0]][0]
            self._env.Removetragetobject(id)
            blockUid = -1  # :+

        if outOfTray == False:
            sample_pos = [selpos[0], selpos[1], self._traypos[2] + self.Z_VALUE]
        else:
            sample_pos = [0, 0, self._traypos[2] + self.Z_VALUE]
        #        self._env._p.addUserDebugLine(sample_pos, selpos, [0, 0, 1])

        return sample_pos, zpos, blockUid, uniqueUid, outOfTray

    def RandEndEffRot(self, ori):       ## 안쓰고있음
        euler = list(self._env._p.getEulerFromQuaternion(ori))
        euler[2] = math.pi * 2 / 3

        return list(euler)

    def MovingZnGrasp(self, zval):      ## 안쓰고있음
        totalz = zval * 0.3
        coef = np.random.random()
        moving_z = totalz * coef + zval * 0.2
        effstate = self._env._UR.getEndEffectorState()
        pos = list(effstate[0])
        ori = list(self._env._p.getEulerFromQuaternion(effstate[1]))
        pos[2] -= moving_z

        return pos + ori

    def AppendData(self, Dataset, img, seg, action, reward):
        data = {'action': action, 'rgb': img, 'seg': seg, 'reward': reward}
        Dataset.append(data)

    def SeperateRandomly(self, value, N):
        score_sum = 0
        scorelist = []
        for i in range(N):
            rand_val = random.random()
            scorelist.append(rand_val)
            score_sum += rand_val

        scorelist = np.array(scorelist)
        scorelist /= score_sum
        ratio = scorelist * value

        return ratio

    def reset(self):
        self._stepcount = 0
        self.blockUid = -1

