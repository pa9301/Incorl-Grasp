"""Actor process들의 관리자. actor factory가 actor policy(qt-opt)를 들고 있고 각 process가 입력을 넣고 action을 요구하면
계산해서 넣어준다. qt-opt는 하나만 들고있음.
(현재 이 부분이 공유자원일 때 어떻게 활용될 수 있는지 확인이 필요함. ex. session)

scripted policy의 경우 각 actor process가 직접 생성한다.

공유 자원은
1.Actor factory와 Actor process가 공유하는 qt-opt
2.Prioritized replay memory (메모리는 Actor process / Memory manager(main 소속) 둘이서 공유한다.
Memory Manager는 Main에서 qtopt용 pixelda용 두개를 생성한다.
실제 PER은 하나만 존재한다. (online / offline으로 구분할 경우 두개?)"""

import os
from multiprocessing import Value, Process, Lock, Queue, Manager
from time import sleep
import cv2
import numpy as np
import math
import json
import time
import cv2
import copy

hidden_sizes=(256, 256)

#OBJ_BATCH_COUNT = 20                         # number of loaded object
#RESETCNT = 5                                # trial in same environment
#USE_RENDERER = True

#TRIAL_COUNT = 3                             # delete?
#ROLLOUT_COUNT = 1000                        # delete?

#NUM_CYCLE = 10
#MAX_REGRASP = 3
#MAX_GRASP_STEP = 20

def normalize_action(x, max, min):
    divisor = max -min
    dividend = x -min
    zero_to_one = dividend / divisor,
    minus_one_to_one = (zero_to_one * 2.0) - 1.0
    return minus_one_to_one

def denormalize_action(minus_one_to_one, max, min):
    zero_to_one = (minus_one_to_one + 1.0) / 2.0
    x = zero_to_one * (max -min) + min
    return x

def resize_image(img, hei, wid):

    img = cv2.resize(img, (hei, wid))
#    img = tf.clip_by_value(img, 0, 255) / 127.5 - 1

    return img

def make_action__for_save(action_list, control, action_info, done, collision, min_max_action_position_range, min_max_action_angle_range):

    terminal = None

    z_max = min_max_action_position_range['z_max']
    z_min = min_max_action_position_range['z_min']
    x_max = min_max_action_position_range['x_max']
    x_min = min_max_action_position_range['x_min']
    y_max = min_max_action_position_range['y_max']
    y_min = min_max_action_position_range['y_min']

    roll_min = min_max_action_angle_range['roll_min']
    roll_max = min_max_action_angle_range['roll_max']
    pitch_min = min_max_action_angle_range['pitch_min']
    pitch_max = min_max_action_angle_range['pitch_max']
    yaw_min = min_max_action_angle_range['yaw_min']
    yaw_max = min_max_action_angle_range['yaw_max']

    action_list.append(normalize_action(control[0], x_max, x_min))
    action_list.append(normalize_action(control[1], y_max, y_min))
    action_list.append(normalize_action(control[2], z_max, z_min))
    action_list.append(normalize_action(control[3], roll_max, roll_min))
    action_list.append(action_info[0])
    action_list.append(action_info[2])
    if action_info[2] >= 0.5:
        terminal = 1.0
    else:
        terminal = 0.0

    if done == True:
        terminal = 1.0

    if collision == True:
        terminal = 1.0

    return copy.deepcopy(action_list), terminal


def make_conrol_actionInfo_from_netowrk_policy(netowrk_policy, min_max_action_position_range, min_max_action_angle_range):

    control = []
    action_info = np.zeros(3, dtype=np.float)

    z_max = min_max_action_position_range['z_max']
    z_min = min_max_action_position_range['z_min']
    x_max = min_max_action_position_range['x_max']
    x_min = min_max_action_position_range['x_min']
    y_max = min_max_action_position_range['y_max']
    y_min = min_max_action_position_range['y_min']

    roll_min = min_max_action_angle_range['roll_min']
    roll_max = min_max_action_angle_range['roll_max']
    pitch_min = min_max_action_angle_range['pitch_min']
    pitch_max = min_max_action_angle_range['pitch_max']
    yaw_min = min_max_action_angle_range['yaw_min']
    yaw_max = min_max_action_angle_range['yaw_max']

    control.append(denormalize_action(netowrk_policy[0], x_max, x_min))
    control.append(denormalize_action(netowrk_policy[1], y_max, y_min))
    control.append(denormalize_action(netowrk_policy[2], z_max, z_min))
    control.append(denormalize_action(netowrk_policy[3], roll_max, roll_min))
    control.append(math.pi / 2)
    control.append(0.0)
    action_info[0] = netowrk_policy[4]
    if netowrk_policy[4] >= 0.5:
        control.append(1.0)
    else:
        control.append(0.0)

    action_info[1] = 1.0 - netowrk_policy[4]
    action_info[2] = netowrk_policy[5]

    return copy.deepcopy(control), copy.deepcopy(action_info)

def ActorProcess(actor_index, args, bProcRun, endProcCnt, scripted_iter, replaybuffer, preimgqueue, imgqueue, latentqueue, policyqueue,
                 rewardqueue, nb_grasp_success_queue, detectqueue, rootpath, constructLock, graspcnt, actorLock, min_max_action_position_range, min_max_action_angle_range):
    from actor.scriptedPolicy import scriptedPolicy
    from simulator.UREnv import URGymEnv
    from utils.utils import WriteData, random_crop, ImgNormalize, ChangeName

    _scripted_iter = scripted_iter
    _replaybuffer = replaybuffer
    _databuffer = []
    _pid = os.getpid()

    # simulator param
    print("Actor process simulator create (%d)" % _pid)
    endProcCnt.value += 1
    current_count = 0
    DoInit = False
    constructLock.acquire()
    environment = URGymEnv(renders=args.use_renderer, objBacthCount=args.nb_obj_in_tray)
    constructLock.release()

    print("Actor process run start (%d)" % _pid)

    scriptPolicy = scriptedPolicy(environment)

    # action 을 normalize 하고 denomalize 하기 위한 value들
    z_max = scriptPolicy._traypos[2] + scriptPolicy.Z_VALUE
    z_min = z_max - 0.25
    x_max = scriptPolicy._traypos[0] + 0.7 * scriptPolicy.X_RANGE
    x_min = scriptPolicy._traypos[0] - 0.7 * scriptPolicy.X_RANGE
    y_max = scriptPolicy._traypos[1] + 0.7 * scriptPolicy.Y_RANGE
    y_min = scriptPolicy._traypos[1] - 0.7 * scriptPolicy.Y_RANGE

    roll_min = -math.pi / 2.0
    roll_max = math.pi / 2.0
    pitch_min = -math.pi / 2.0
    pitch_max = math.pi / 2.0
    yaw_min = -math.pi / 2.0
    yaw_max = math.pi / 2.0
    max_grasp_nb = args.max_grasp_nb / 3.5
    # action 을 normalize 하고 denomalize 하기 위한 value들

    epoch = -1
    nb_grasp = 0
    exploration_type_ratio = 1.0
    # 1 : script, 0 : network
    rewardqueue.put(args.min_reward)
    nb_grasp_success_queue.put(0.0)

    eval_grasp_info = np.zeros(args.nb_eval_epoch_cycles, dtype=np.float)
    is_train = True

    while not bProcRun.value == 1:
        sleep(0)
        is_train = True
#        environment.safeReset(is_train)
        environment.safeReset()
        epoch += 1

        same_IDcount = 0
        temp_id = 0

        for cycle in range(args.nb_epoch_cycles):
            sleep(0)

            is_train = True
            done = False
            collision = False
            collisionDetected = False
            scriptEnd = False
            remove_object = False
            obj_id = None
            unique_obj_id = None
            control = None
            action_info = None
            state = None
            is_first_step_in_grasp = True
            reward = args.min_reward
            scriptPolicy.reset()

            # action 을 scripted_policy 로 생성할지 q_network으로 생성할지에 대한 샘플링
            exploration_type = np.random.binomial(1, exploration_type_ratio, 1)[0]

            if exploration_type == 1:
                exploration_type = 'script'
            elif exploration_type == 0:
                exploration_type = 'network'

            # grasp 1번 수행 ( 여기에는 regreasp 을 포함하고 있음. 따라서 물리적으로 1번이상 파지를 수행할 수 있음)
            # action vector의 마지막 dimesion이 grasp을 종료할지에 대한 value임 따라서 이 값이 1이면 for문이 break 되고
            # 아니면 args.nb_rollout_steps 까지 수행
            first_action = None

            action_info = np.zeros(3, dtype=np.float)

            for step_in_rollout in range(args.nb_rollout_steps):

                control = []
                # 새로운 grasp를 시작해야 하는 조건 체크
                if done == True:
                    break

                if collision == True:
#                    environment.safeReset(is_train)
                    environment.safeReset()
                    break

                if environment.checkObjectSafe() is False:
#                    environment.safeReset(is_train)
                    environment.safeReset()
                    break
                # 새로운 grasp를 시작해야 하는 조건 체크

                action_list = []    #action을 파일로 쓰기 위한 최종 자료형
                action_temp = np.zeros(9, dtype=np.float)

                # grasp 의 첫번째 step은 임의의 포지션으로 이동하는 action
                # 첫번째 step은 로직상 특수하게 간주됨
                if is_first_step_in_grasp == True:
                    first_action, action_info, blockUid, uniqueUid, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()

                    # 가끔씩 아래와 같은 경우가 발생하는데 확인해 볼것
                    # 아래와 같은 경우에는 새로운 grasp을 수행
                    if blockUid < 0:
#                        environment.safeReset(is_train)
                        environment.safeReset()
                        break

                    obj_id = blockUid
                    unique_obj_id = uniqueUid
                    target_obj_name = ChangeName(uniqueUid)
#                    print('target_obj_name :', target_obj_name)
                    if temp_id is obj_id:
                        if 2 < same_IDcount:
                            environment.safeReset()
                            print('\x1b[1;30m' + '-->>env _ environment.safeReset() for retry' + '\x1b[0;m')
                            temp_id = 0
                            same_IDcount = 0
                            break
                        same_IDcount += 1
                    else:
                        temp_id = 0
                        same_IDcount = 0
                        temp_id = obj_id
                    # 임의의 포지션으로 이동하는 action
                    collisionDetected, reward = environment.MoveRobotExactPosition(first_action, obj_id, False)
                    previous_action = copy.deepcopy(first_action)
                    is_first_step_in_grasp = False
                    # 이렇게 첫번째 step은 design된 action이기 때문에 RL학습에는 사용되지 않음 따라서 저장하지 않고 continue
                    continue

                # is_first_step_in_grasp 이 False일 경우 여기에서부터 시작
                # action을 수행하기 이전의 사전 state
                # 특정 물체만 highlight 할 경우 obj_unique_id 가 필요함
#                _, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _, _ = environment.getExtendedObservation(obj_id, True)
                rgb_img_360, rgb_img_256, rgb_crop_for_GAN360 = environment.getObservation(_pid, unique_obj_id, None)
                # ++ 20190902 이후
                internal_state = environment.get_internal_state()

                # action을 생성
                if exploration_type == 'script':
#                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, None])
                    preimgqueue.put([rgb_img_360, rgb_img_256, rgb_crop_for_GAN360, internal_state, None])
                    boxes, scores, classes, num, fakeReal_360, arm_seg_256 = detectqueue.get()

                    _, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _ \
                        = environment.getExtendedObservation(_pid, boxes, scores, classes, fakeReal_360, arm_seg_256, unique_obj_id, True)

                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, None])
                    control, action_info, _, _, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()
                    state = latentqueue.get()

                elif exploration_type == 'network':

#                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, False])
                    preimgqueue.put([rgb_img_360, rgb_img_256, rgb_crop_for_GAN360, internal_state, None])
                    boxes, scores, classes, num, fakeReal_360, arm_seg_256 = detectqueue.get()

                    _, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _ \
                        = environment.getExtendedObservation(_pid, boxes, scores, classes, fakeReal_360, arm_seg_256, unique_obj_id, True)

                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, False])
                    state = latentqueue.get()
                    [netowrk_policy, q] = policyqueue.get()
                    netowrk_policy = netowrk_policy.astype(np.float)

                    [control, action_info] = make_conrol_actionInfo_from_netowrk_policy(copy.deepcopy(netowrk_policy), min_max_action_position_range, min_max_action_angle_range)

                # action을 수행
                if exploration_type == 'script':
                    collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    previous_action = copy.deepcopy(control)
                elif exploration_type == 'network':

                    effstate = environment._UR.getEndEffectorState()
                    # 아래로 내려가는 동작을 취하고 있을 경우에는 gripper를 우선 무조건 open하고 정해진 위치까지 이동

                    collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    previous_action = copy.deepcopy(control)

                # collision 체크하여 collision 했으면 새로운 grasp을 수행
                if collisionDetected:
                    collision = True
#                    environment.safeReset()
#                    break

                # grasp 의 종료에 관련된 상황 체크
                if exploration_type == 'script':
                    # script가 종료일 경우에만 reward를 받는다
                    if scriptEnd is False:
                        reward = args.min_reward
                    if scriptEnd:
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1
                        done = True
                        terminate_episode = np.random.uniform()
                        # action vector의 마지막 dimesion이 1 일 경우에만 reward 1을 받을 수 있음
                        if terminate_episode > args.min_ratio_for_terminate_epsode:
                            action_info[2] = (np.random.uniform() / 2.0) + 0.5  #episode를 종료한다는 결정을 내림
                            if reward == 1.0:
                                remove_object = True
                        else:
                            reward = args.min_reward

                elif exploration_type == 'network':
                    eff_pos = environment._UR.getEndEffectorState()[0]

                    # end effector의 z 값이 일정 높이 이상일 경우만 grasp 성공이 될 수 있음
                    if eff_pos[2] < args.min_z_value_for_grasp_success:
                        reward = args.min_reward

                    if action_info[2] >= 0.5:  # q network이 grasp 를 종료하고자 할 때
                        done = True
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1

                        # 만약 grasp 성공했으면 물체를 없앨것
                        if reward == 1.0:
                            remove_object = True
                            print('grasp success in exploration')
                    else:                   # q network이 grasp 를 종료하고자 하지 않을 때
                        reward = args.min_reward

                        # step_in_rollout 이 maximum step일 때
                    if step_in_rollout == args.nb_rollout_steps - 1:
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1
                        done = True
                        break           # break 를 하는 것이 좋은가 아니면 데이터를 저장하는 것이 좋은가?

                # grasp 의 종료에 관련된 상황 체크

                # grasp 성공했으면 물체를 삭제
                if remove_object:
                    environment.Removetragetobject(obj_id)

                # action을 수행하고 난 사후 상태
#                _, new_img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _, _ = environment.getExtendedObservation(obj_id, True)
                new_rgb_img_360, new_rgb_img_256, new_rgb_crop_for_GAN360 = environment.getObservation(_pid, obj_id,
                                                                                                       True)

                new_internal_state = environment.get_internal_state()
                preimgqueue.put([new_rgb_img_360, new_rgb_img_256, new_rgb_crop_for_GAN360, new_internal_state, None])
                boxes, scores, classes, num, fakeReal_360, arm_seg_256 = detectqueue.get()
#                imgqueue.put([new_img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, new_internal_state, None])
                _, new_img, new_ext_pos_img, new_ext_vis_img, new_int_pos_img, new_int_vis_img, _, _ \
                    = environment.getExtendedObservation(_pid, boxes, scores, classes, fakeReal_360, arm_seg_256, unique_obj_id, True)

                imgqueue.put([new_img, new_ext_pos_img, new_ext_vis_img, new_int_pos_img, new_int_vis_img, new_internal_state, None])
                new_state = latentqueue.get()

                [action_list, terminal] = make_action__for_save(action_list, copy.deepcopy(control), copy.deepcopy(action_info), done, collision, min_max_action_position_range,
                                          min_max_action_angle_range)

                current_count += 1
                path = WriteData(rootpath, state.tolist(), new_state.tolist(), None, None,
                                 action_list, internal_state, new_internal_state,
                                 reward, terminal, current_count, '.jpg')

                path = path.replace('\\', '/')
                _replaybuffer.store(path)

            graspcnt.value += 1
            ratio = (max_grasp_nb - graspcnt.value) / max_grasp_nb
            if ratio < 0.05:
                ratio = 0.05
            exploration_type_ratio = ratio

        if args.eval is True and epoch % 5 == 0 and epoch > 10:
            print('start eval!!!')
            is_train = False

#            environment.safeReset(is_train)
            environment.safeReset()
            time.sleep(0.1)

            mean_eval_returns = []
            mean_eval_qs = []
            eval_grasp_info.fill(0.0)
            for eval_cycle in range(args.nb_eval_epoch_cycles):

                eval_returns = []
                eval_qs = []
                done = False
                is_first_step_in_grasp = True
                obj_id = None
                unique_obj_id = None
                previous_action = None
                remove_object = False

                scriptPolicy.reset()
                action_info = np.zeros(3, dtype=np.float)

                for step_in_rollout in range(args.max_step_for_test_episode):
                    control = []
                    # 새로운 grasp를 시작해야 하는 조건 체크
                    if done == True:
                        break

                    if environment.checkObjectSafe() is False:
#                        environment.safeReset(is_train)
                        environment.safeReset()
                        break
                    # 새로운 grasp를 시작해야 하는 조건 체크

                    if is_first_step_in_grasp == True:
                        first_action, action_ext, blockUid, uniqueUid, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()

                        # 가끔씩 아래와 같은 경우가 발생하는데 확인해 볼것
                        # 아래와 같은 경우에는 새로운 grasp을 수행
                        if blockUid < 0:
                            environment.safeReset()
#                            environment.safeReset(is_train)
                            break

                        obj_id = blockUid
                        unique_obj_id = uniqueUid
                        if temp_id is obj_id:
                            if 2 < same_IDcount:
                                environment.safeReset()
                                print('\x1b[1;30m' + '-->>env _ environment.safeReset() for retry' + '\x1b[0;m')
                                temp_id = 0
                                same_IDcount = 0
                                break
                            same_IDcount += 1
                        else:
                            temp_id = 0
                            same_IDcount = 0
                            temp_id = obj_id
                        # 임의의 포지션으로 이동하는 action
                        collisionDetected, reward = environment.MoveRobotExactPosition(first_action, obj_id, False)
                        previous_action = copy.deepcopy(first_action)
                        is_first_step_in_grasp = False
                        # 이렇게 첫번째 step은 design된 action이기 때문에 RL학습에는 사용되지 않음 따라서 저장하지 않고 continue
                        continue

                    rgb_img_360, rgb_img_256, rgb_crop_for_GAN360 = environment.getObservation(_pid, unique_obj_id,
                                                                                               True)
                    internal_state = environment.get_internal_state()
                    preimgqueue.put([rgb_img_360, rgb_img_256, rgb_crop_for_GAN360, internal_state, None])
                    boxes, scores, classes, num, fakeReal_360, arm_seg_256 = detectqueue.get()

                    _, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _ \
                        = environment.getExtendedObservation(_pid, boxes, scores, classes, fakeReal_360, arm_seg_256, unique_obj_id, True)

                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, True])

                    state = latentqueue.get()
                    [netowrk_policy, q] = policyqueue.get()
                    netowrk_policy = netowrk_policy.astype(np.float)
                    eval_qs.append(q)

                    [control, action_info] = make_conrol_actionInfo_from_netowrk_policy(copy.deepcopy(netowrk_policy), min_max_action_position_range, min_max_action_angle_range)


                    effstate = environment._UR.getEndEffectorState()

                    collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    previous_action = copy.deepcopy(control)

                    if collisionDetected:
                        reward = args.min_reward
                        eval_returns.append(reward)
#                        environment.safeReset(is_train)
                        environment.safeReset()
                        break

                    eff_pos = environment._UR.getEndEffectorState()[0]


                    # end effector의 z 값이 일정 높이 이상일 경우만 grasp 성공이 될 수 있음
                    if eff_pos[2] < args.min_z_value_for_grasp_success:
                        reward = args.min_reward

                    if action_info[2] >= 0.5:  # q network이 grasp 를 종료하고자 할 때
                        eval_returns.append(reward)
                        done = True

                        # 만약 grasp 성공했으면 물체를 없앨것
                        if reward == 1.0:
                            remove_object = True
                            eval_grasp_info[eval_cycle] = 1.0
                            print('grasp success in test!!!')
                    else:                   # q network이 grasp 를 종료하고자 하지 않을 때
                        reward = args.min_reward

                    eval_returns.append(reward)

                    if remove_object:
                        environment.Removetragetobject(obj_id)
                        break

                if len(eval_returns) > 0:
                    mean_eval_returns.append(np.sum(eval_returns) / float(len(eval_returns)))
                    mean_eval_qs.append(np.sum(eval_qs) / float(len(eval_qs)))
                    print('eval_mean_return :', np.sum(eval_returns) / float(len(eval_returns)))
                else:
                    mean_eval_returns.append(0.0)
                    mean_eval_qs.append(0.0)

            summary_nb_grasp_success = np.sum(eval_grasp_info)
            summary_reward = np.mean(mean_eval_returns)
            summary_q = np.mean(mean_eval_qs)
            actorLock.acquire()
            try:
                rewardqueue.put(summary_reward)
                nb_grasp_success_queue.put(summary_nb_grasp_success)
            finally:
                actorLock.release()


    endProcCnt.value -= 1


def ManagerProcess(num_actor, weightsQ, gpuidx, bManagerStop, PreImgQueueList, ImgQueueList, LatentQueueList,
                   PolicyQueueList, DetectedQueueList, args, lock, EndProcCnt):
    # GAN network alloc
    from VAEs.VAE_model import VAE
    from object_detection.Detector_model import detector
    from arm_segmentation.arm_segmentation_model import arm_segmentation
    from GAN.GAN_model import GAN
    from D4PG.d4pg import D4PG
    from D4PG.noise import NormalActionNoise, AdaptiveParamNoiseSpec
    import tensorflow as tf
    from utils.utils import random_crop, ImgNormalize, ImgInverNormalize, refine_segmented_image_by_connected_component
    from D4PG.models import Actor, Critic

    _pid = os.getpid()
    print("Actor Manager process create (%d)" % _pid)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:2'):
        detector_model = detector(sess, 24)

    with tf.device('/gpu:1'):
        arm_segmentation_model = arm_segmentation(sess, args.arm_segmentation_dir)

    with tf.device('/gpu:0'):
        GAN_model = GAN(sess, args.GAN_dir)

    with tf.device('/gpu:2'):
        vae_model = VAE(sess, args.SRL_dir)

    action_noise = None
    param_noise = None

    nb_actions = args.ddpg_action_shape
    for current_noise_type in args.noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))


    with tf.device('/gpu:1'):
        d4pg = D4PG('gpu1_', args, observation_shape=args.latent_size,
                    action_shape=args.ddpg_action_shape, internal_state_shape=args.ddpg_internal_state_shape,
                    param_noise=param_noise, action_noise=action_noise,
                    gamma=args.ddpg_gamma, tau=args.ddpg_tau, normalize_observations=args.normalize_observations,
                    batch_size=args.ddpg_batch_size, critic_l2_reg=args.ddpg_q_l2_reg,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    d4pg.initialize(sess)

    EndProcCnt.value += 1

    counts_for_check_actor_alive = np.zeros(num_actor)

    while not bManagerStop.value == 1:
        sleep(0.01)
        weight = None
        lock.acquire()

        try:
            if not weightsQ.empty():
                weight = weightsQ.get()
        finally:
            lock.release()

        if weight is not None:
            with tf.device('/gpu:1'):
                start_time = time.time()
                d4pg.update_weight(weight)
                print("sess.run: %s seconds ---" % (time.time() - start_time))

        is_eval = None

        for i in range(num_actor):
            is_eval = None

            if PreImgQueueList[i].empty() is False:
                [rgb_img_360, rgb_img_256, rgb_crop_for_GAN360, internal_state, is_eval] = PreImgQueueList[i].get()  # receive img from actor
                boxes, scores, classes, num = detector_model.detection(rgb_img_360)

                arm_seg_256 = arm_segmentation_model.seg_img(rgb_img_256)  # : 0~2, 1*256*256
                arm_seg_256 = np.squeeze(arm_seg_256)
                arm_seg_256 = refine_segmented_image_by_connected_component(arm_seg_256, 2, 100)

                fakeReal_temp = GAN_model.fakereal_img(rgb_crop_for_GAN360)  # : 이미지 저장할 세그먼트
                fakeReal_temp = np.squeeze(fakeReal_temp)
                fakeReal_360 = ImgInverNormalize(fakeReal_temp, 255)

                DetectedQueueList[i].put([boxes, scores, classes, num, fakeReal_360, arm_seg_256])

#                image_detected = detector_model.visualize(rgb_img_360, boxes, scores, classes)
#                image_detected = image_detected[..., ::-1]
#                rgb_img_360 = rgb_img_360[..., ::-1]
#                cv2.imshow('rgb_img_360', rgb_img_360)
#                cv2.waitKey(1)

#                fakeReal_360 = fakeReal_360[..., ::-1]
#                cv2.imshow('fakeReal_360', fakeReal_360)
#                cv2.waitKey(1)

            if ImgQueueList[i].empty() is False:

                [img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, is_eval] = ImgQueueList[i].get()     # receive img from actor

                vae_img = np.concatenate((int_pos_img, int_vis_img, ext_pos_img, ext_vis_img), axis=2)
                latent = vae_model.latent(vae_img)
                LatentQueueList[i].put(latent)
                state = latent

                if i == 0 and args.show_recon_of_latent is True:

                    img_recon = vae_model.recon_img(vae_img).squeeze()

                    img_show = img.copy()
                    img_show = img_show[..., ::-1]
                    recon_show = ImgInverNormalize(img_recon, 255)

                    ex_recon_pos = recon_show[..., :3]
                    ex_recon_pos = ex_recon_pos[..., ::-1]
                    ex_recon_vis = recon_show[..., 3:6]
                    ex_recon_vis = ex_recon_vis[..., ::-1]

                    in_recon_pos = recon_show[..., 6:9]
                    in_recon_pos = in_recon_pos[..., ::-1]
                    in_recon_vis = recon_show[..., 9:12]
                    in_recon_vis = in_recon_vis[..., ::-1]

                    cv2.imshow('ex_recon_pos', ex_recon_pos)
                    cv2.imshow('ex_recon_vis', ex_recon_vis)
                    cv2.imshow('in_recon_pos', in_recon_pos)
                    cv2.imshow('in_recon_vis', in_recon_vis)
                    cv2.imshow('img', img_show)
                    cv2.waitKey(1)


                if is_eval is not None:

                    if i == 0 and is_eval:
                        img_show = img.copy()
                        cv2.imshow('img', img_show)
                        cv2.waitKey(1)

                    internal_state = np.asarray(internal_state)
                    internal_state = np.reshape(internal_state, (1, 2))
                    action, q = d4pg.policy(state, internal_state, is_eval=is_eval, compute_Q=True)
                    #TODO qtopt run action
                    PolicyQueueList[i].put([action, q]) #TODO
                    is_eval = None

        sleep(0.01)

    print('manager close')



# Manager process를 생성하고 Actor들을 생성한다
# Manager는 Session을 가지고 Actor들의 입력을 처리해주는 역할을 한다.
# bContinue == true 일 경우 Json으로부터 카운트를 읽어들여옴
class ActorFactory:
    def __init__(self, num_actor, replay_buffer, scripted_iter, gpuidx, rootpath, arg_ddpg, nb_grasp, dummy, bContinue = False):
        self.args_main = arg_ddpg
        self._bManagerStop = Value('i', 0)
        self._ManagerEndProcCnt = Value('i', 0)
        self.rewards = np.zeros(num_actor) + arg_ddpg.default_reward
        self.qs = np.zeros(num_actor)
        self.nb_success = np.zeros(num_actor)
        self._num_actor = num_actor

        self._ActorLock = Lock()
        self._ManagerLock = Lock()
        self._ConstructorLock = Lock()
        self._weight_queue = Queue()

        self._bActorProcRun = Value('i', 0)
        self._ActorEndProcCnt = Value('i', 0)
        self._procList = []
        self._scripted_iter = scripted_iter
        self._rootpath = rootpath
        self._graspcnt = Value('i', nb_grasp)
        self._graspcnt.value = nb_grasp

        self._replay_buffer = replay_buffer

        manager = Manager()
        self.min_max_action_position_range = manager.dict()
        self.min_max_action_angle_range = manager.dict()

        # action 을 normalize 하고 denomalize 하기 위한 value들
        #z_max = 0.245
        #z_min = z_max - 0.25
        #x_max = 0.7355
        #x_min = 0.1895
        #y_max = 0.358
        #y_min = -0.388

#        z_max = 0.245
#        z_min = z_max - 0.25
#        x_max = 0.667
#        x_min = 0.247
#        y_max = 0.28
#        y_min = -0.28

        z_max = 0.245
        z_min = 0.045
        x_max = 0.684
        x_min = 0.244
        y_max = 0.275
        y_min = -0.285

        roll_min = -math.pi / 2.0
        roll_max = math.pi / 2.0
        pitch_min = -math.pi / 2.0
        pitch_max = math.pi / 2.0
        yaw_min = -math.pi / 2.0
        yaw_max = math.pi / 2.0
        # action 을 normalize 하고 denomalize 하기 위한 value들

        self.min_max_action_position_range = {'z_max': z_max, 'z_min': z_min, 'x_max': x_max, 'x_min': x_min, 'y_max': y_max,
                                         'y_min': y_min}
        self.min_max_action_angle_range = {'roll_min': roll_min, 'roll_max': roll_max, 'pitch_min': pitch_min,
                                      'pitch_max': pitch_max, 'yaw_min': yaw_min,
                                      'yaw_max': yaw_max}

        self._PreImgQueueList = []  # actor process -> actor manager
        self._ImgQueueList = []  # actor process -> actor manager
        self._LatentQueueList = []  # actor process -> actor manager
        self._PolicyQueueList = []  # actor manager -> actor process
        self._RewardQueueList = []  # actor manager -> actor process
        self._DetectedQueueList = []  # actor manager -> actor process
        self._NbGraspSuccessQueueList = []  # actor manager -> actor process
        for i in range(self._num_actor):
            self._PreImgQueueList.append(Queue())
            self._ImgQueueList.append(Queue())
            self._LatentQueueList.append(Queue())
            self._PolicyQueueList.append(Queue())
            self._RewardQueueList.append(Queue())
            self._DetectedQueueList.append(Queue())
            self._NbGraspSuccessQueueList.append(Queue())

        args = (self._num_actor, self._weight_queue, gpuidx, self._bManagerStop, self._PreImgQueueList, self._ImgQueueList, self._LatentQueueList,
                self._PolicyQueueList, self._DetectedQueueList, self.args_main, self._ManagerLock, self._ManagerEndProcCnt)
        self._manager_process = Process(target=ManagerProcess, args=args)
        self._manager_process.start()

        while self._ManagerEndProcCnt.value != 1:
            sleep(1)

        self.StartDistributedActor()

    def __del__(self):
        self._bManagerStop.value = 1
        self._manager_process.join()
        self._weight_queue.close()

        self.KillAllProcess()

        for i in range(self._num_actor):
            self._PreImgQueueList[i].close()
            self._ImgQueueList[i].close()
            self._LatentQueueList[i].close()
            self._PolicyQueueList[i].close()
            self._RewardQueueList[i].close()
            self._DetectedQueueList[i].close()
            self._NbGraspSuccessQueueList[i].close()

    def KillAllProcess(self):
        self._bActorProcRun = 0
        for proc in self._procList:
            proc.join()

    def StartDistributedActor(self):
        for i in range(self._num_actor):
            proc = Process(target=ActorProcess, args=(i, self.args_main, self._bActorProcRun, self._ActorEndProcCnt, self._scripted_iter,
                                                      self._replay_buffer, self._PreImgQueueList[i], self._ImgQueueList[i], self._LatentQueueList[i],
                                                      self._PolicyQueueList[i], self._RewardQueueList[i],
                                                      self._NbGraspSuccessQueueList[i],  self._DetectedQueueList[i], self._rootpath, self._ConstructorLock,
                                                      self._graspcnt, self._ActorLock, self.min_max_action_position_range, self.min_max_action_angle_range))

            self._procList.append(proc)

        for p in self._procList:
            p.start()

        while self._ActorEndProcCnt.value != self._num_actor:
            sleep(1)

    def getGraspCount(self):
        self._ActorLock.acquire()
        try:
            graspcnt = self._graspcnt.value
        finally:
            self._ActorLock.release()

        return graspcnt

    def getLatent(self, actoridx):
        if actoridx >= self._num_actor:
            return None

        return self._LatentQueueList[actoridx].get()

    def UpdatePolicy(self, weights):

        self._ManagerLock.acquire()
        self._weight_queue.put(weights)
        self._ManagerLock.release()

    def get_mean_eval_returns(self):

        rewards = []
        for i in range(self._num_actor):

            self._ActorLock.acquire()
            try:
                if self._RewardQueueList[i].empty() is False:
                    self.rewards[i] = self._RewardQueueList[i].get()
            finally:
                self._ActorLock.release()

            rewards.append(self.rewards[i])

        return rewards

    def get_mean_grasp_success(self):

        nb_success = []
        for i in range(self._num_actor):

            self._ActorLock.acquire()
            try:
                if self._NbGraspSuccessQueueList[i].empty() is False:
                    self.nb_success[i] = self._NbGraspSuccessQueueList[i].get()
            finally:
                self._ActorLock.release()

            nb_success.append(self.nb_success[i])

        return nb_success