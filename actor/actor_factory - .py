import os
from multiprocessing import Value, Process, Lock, Queue, Manager
from time import sleep
import numpy as np
import math
import time
import cv2
import copy

# : Actor Critic의 hidden 크기설정
hidden_sizes = (256, 256)


def normalize_action(x, max_val, min_val):
    divisor = max_val - min_val
    dividend = x - min_val
    zero_to_one = dividend / divisor
    minus_one_to_one = (zero_to_one * 2.0) - 1.0
    return minus_one_to_one


def denormalize_action(minus_one_to_one, max_val, min_val):
    zero_to_one = (minus_one_to_one + 1.0) / 2.0
    x = zero_to_one * (max_val - min_val) + min_val
    return x


def resize_image(img, hei, wid):

    img = cv2.resize(img, (hei, wid))
#    img = tf.clip_by_value(img, 0, 255) / 127.5 - 1

    return img


def bgr2rgb(img):
    img = img[:, :, 0:3]  # 4채널에서 alpha 채널 제거
    img = img[..., ::-1]  # bgr to rgb
    return img


# = normalizing
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


# = normalizing
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


def ActorProcess(args, bProcRun, endProcCnt, scripted_iter, replaybuffer, imgqueue, latentqueue, policyqueue,
                 rewardqueue, qvalqueue, rootpath, constructLock, graspcnt, min_max_action_position_range, min_max_action_angle_range):
    from actor.scriptedPolicy import scriptedPolicy
    from simulator.UREnv import URGymEnv
    from utils.utils import WriteData, WriteImg, random_crop, ImgNormalize

    _scripted_iter = scripted_iter
    _replaybuffer = replaybuffer
    _databuffer = []
    _pid = os.getpid()

    # simulator param
#    print('\x1b[1;30m' + "-->>sys: Actor process simulator create (%d)" % _pid + '\x1b[0;m')
    endProcCnt.value += 1
    current_count = 0
    DoInit = False
    constructLock.acquire()
    environment = URGymEnv(renders=args.use_renderer, objBacthCount=args.nb_obj_in_tray)
    constructLock.release()

    print('\x1b[1;30m' + "-->>sys: Actor process run start (%d)" % _pid + '\x1b[0;m')

    scriptPolicy = scriptedPolicy(environment)

    epoch = -1
    nb_grasp = 0
    exploration_type_ratio = 0.7
    # 1 : script, 0 : network
    rewardqueue.put(args.min_reward)

    while not bProcRun.value == 1:
        sleep(0)
        environment.safeReset()
        epoch += 1

        for cycle in range(args.nb_epoch_cycles):
            sleep(0)

            done = False
            collision = False
            collisionDetected = False
            scriptEnd = False
            remove_object = False
            obj_id = None
            control = None
            action_info = None
            state = None
            is_first_step_in_grasp = True
            reward = args.min_reward
            scriptPolicy.reset()
            images = []                     # : 이미지 임시 저장을 위한 변수
            actions = []                    # : 액션 임시 저장을 위한 변수
            rewards = []                    ## 안쓰고있음
            internal_states = []            # : int state 임시 저장을 위한 변수
            is_success = False
            previous_action = None

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
#                print('\x1b[1;32m' + '-->>sys : {} step_in_rollout : {}/{}'.format(_pid, step_in_rollout,
#                                                                                args.nb_rollout_steps) + '\x1b[0;m')

                control = []                # : action을 뜻함
                action_list = []            # action을 파일로 쓰기 위한 최종 자료형
                # 새로운 grasp를 시작해야 하는 조건 체크
                if done == True:
                    break

                if collision == True:
                    environment.safeReset()
                    break

                if environment.checkObjectSafe() is False:
                    environment.safeReset()
                    break
                # 새로운 grasp를 시작해야 하는 조건 체크

                # grasp 의 첫번째 step은 임의의 포지션으로 이동하는 action
                # 첫번째 step은 로직상 특수하게 간주됨
                if is_first_step_in_grasp == True:              # : 첫번째 스텝 진행시
                    environment.openClossGripper(bOpenClose=False)  ## 추가 - 물체 던지기 방지용
                    environment.WaitUntilObjectSafe()               ## 추가 - 물체 던지기 방지용
                    first_action, action_info, blockUid, uniqueUid, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()

                    # 가끔씩 아래와 같은 경우가 발생하는데 확인해 볼것
                    # 아래와 같은 경우에는 새로운 grasp을 수행
                    if blockUid < 0:
                        environment.safeReset()
                        break

                    obj_id = blockUid
                    # 임의의 포지션으로 이동하는 action
                    collisionDetected, reward = environment.MoveRobotExactPosition(first_action, obj_id, False)
                    previous_action = copy.deepcopy(first_action)
                    is_first_step_in_grasp = False
                    # 이렇게 첫번째 step은 design된 action이기 때문에 RL학습에는 사용되지 않음 따라서 저장하지 않고 continue
                    continue

                # is_first_step_in_grasp 이 False일 경우 여기에서부터 시작
                # action을 수행하기 이전의 사전 state
                # 특정 물체만 highlight 할 경우 obj_unique_id 가 필요함
                img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img = environment.getExtendedObservation(obj_id, True)
                img = bgr2rgb(img)        # 4채널에서 alpha 채널 제거 # bgr to rgb  ## 수정 - 함수화
                ext_vis_img = bgr2rgb(ext_vis_img)
                int_vis_img = bgr2rgb(int_vis_img)
                ## img, _ = random_crop(img, None)                                  ## 수정 이미 랜덤크롭한 이미지들

                internal_state = environment.get_internal_state()

                # action을 생성
                if exploration_type == 'script':
                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, None])
                    control, action_info, blockUid, uniqueUid, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()
                    #<-control, action_info, _, _, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()              ## 확인필요
                    [ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent] = latentqueue.get()
                    state = np.concatenate((ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent), axis=1)

                elif exploration_type == 'network':

                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, False])
                    [ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent] = latentqueue.get()
                    state = np.concatenate((ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent), axis=1)
                    [netowrk_policy, q] = policyqueue.get()
                    netowrk_policy = netowrk_policy.astype(np.float)

                    # = normalizing
                    [control, action_info] = make_conrol_actionInfo_from_netowrk_policy(copy.deepcopy(netowrk_policy), min_max_action_position_range, min_max_action_angle_range)

                # action을 수행
                if exploration_type == 'script':
                    collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    previous_action = copy.deepcopy(control)
                elif exploration_type == 'network':

                    effstate = environment._UR.getEndEffectorState()
                    # 아래로 내려가는 동작을 취하고 있을 경우에는 gripper를 우선 무조건 open하고 정해진 위치까지 이동
                    if effstate[0][2] - control[2] > 0:
                        control[6] = 0
                        collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    # 위로 올라가는 동작을 취하고 있을 경우에 그리고 gripper를 닫을 때는 우선 무조건 gripper 부터 닫고 정해진 위치까지 이동
                    if effstate[0][2] - control[2] < 0:
                        if control[6] == 1:
                            previous_action[6] = 1
                            collisionDetected, reward = environment.MoveRobotExactPosition(previous_action, obj_id, True)
                            sleep(0.5)

                    collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    previous_action = copy.deepcopy(control)

                # collision 체크하여 collision 했으면 새로운 grasp을 수행
                if collisionDetected:
                    collision = True
                    environment.safeReset()
                    break

                # grasp 의 종료에 관련된 상황 체크
                if exploration_type == 'script':
                    # script가 종료일 경우에만 reward를 받는다
                    if scriptEnd is False:
                        reward = args.min_reward
                    if scriptEnd:
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1
                        done = True
                        is_success = False                      # + for SFPN

                        terminate_episode = np.random.uniform()
                        # action vector의 마지막 dimesion이 1 일 경우에만 reward 1을 받을 수 있음
                        if terminate_episode > args.min_ratio_for_terminate_epsode:
                            action_info[2] = (np.random.uniform() / 2.0) + 0.5  #episode를 종료한다는 결정을 내림
                            if reward == 1.0:
                                remove_object = True
                                is_success = True               # + for SFPN

                            else:  ## 추가 - 물체 grasp 실패시에도 그리퍼는 무조건 열기 (던지기 방지)
                                environment.openClossGripper(bOpenClose=False)
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
                            is_success = True                   # + for SFPN
                        else:  ## 추가 - 물체 grasp 실패시에도 그리퍼는 무조건 열기 (던지기 방지)
                            environment.openClossGripper(bOpenClose=False)
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

                # + for SFPN                                                                                            ## 확인필요
                if exploration_type == 1:
                    control = control[0:6] + action_info

                # action을 수행하고 난 사후 상태
                new_img, new_ext_pos_img, new_ext_vis_img, new_int_pos_img, new_int_vis_img = environment.getExtendedObservation(obj_id, True)
                new_img = bgr2rgb(new_img)                      ## 수정 - 함수화
                new_ext_vis_img = bgr2rgb(new_ext_vis_img)
                new_int_vis_img = bgr2rgb(new_int_vis_img)

                new_internal_state = environment.get_internal_state()
                imgqueue.put([new_img, new_ext_pos_img, new_ext_vis_img, new_int_pos_img, new_int_vis_img, new_internal_state, None])
                [new_ext_pos_latent, new_ext_vis_latent, new_int_pos_latent, new_int_vis_latent] = latentqueue.get()
                new_state = np.concatenate((new_ext_pos_latent, new_ext_vis_latent, new_int_pos_latent, new_int_vis_latent), axis=1)

                # = normalizing
                [action_list, terminal] = make_action__for_save(action_list, copy.deepcopy(control), copy.deepcopy(action_info), done, collision, min_max_action_position_range,
                                          min_max_action_angle_range)

                # +
#                images.append(img.copy())
#                internal_states.append(internal_state.copy())
#                np_action = action_list[0:4] + action_info                                                              ## 확인필요
#                actions.append(np_action.copy())

                current_count += 1
                path = WriteData(rootpath, state.tolist(), new_state.tolist(), None, None,
                                 action_list, internal_state, new_internal_state,
                                 reward, terminal, current_count, '.png')

                path = path.replace('\\', '/')
                _replaybuffer.store(path)

            graspcnt.value += 1

        if args.eval is True and epoch % 5 == 0 and epoch > 10:
            print('start eval!!!')

            environment.safeReset()
            time.sleep(0.1)

            mean_eval_returns = []
            mean_eval_qs = []
            for eval_cycle in range(args.nb_eval_epoch_cycles):

                eval_returns = []
                eval_qs = []
                done = False
                is_first_step_in_grasp = True
                obj_id = None
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
                        environment.safeReset()
                        break
                    # 새로운 grasp를 시작해야 하는 조건 체크

                    if is_first_step_in_grasp == True:
                        first_action, action_ext, blockUid, uniqueUid, scriptEnd, outOfTray = scriptPolicy.getScriptedPolicy()

                        # 가끔씩 아래와 같은 경우가 발생하는데 확인해 볼것
                        # 아래와 같은 경우에는 새로운 grasp을 수행
                        if blockUid < 0:
                            environment.safeReset()
                            break

                        obj_id = blockUid
                        # 임의의 포지션으로 이동하는 action
                        collisionDetected, reward = environment.MoveRobotExactPosition(first_action, obj_id, False)
                        previous_action = copy.deepcopy(first_action)
                        is_first_step_in_grasp = False
                        # 이렇게 첫번째 step은 design된 action이기 때문에 RL학습에는 사용되지 않음 따라서 저장하지 않고 continue
                        continue

                    img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img = environment.getExtendedObservation(obj_id, True)
                    img = img[:, :, 0:3]
                    img = img[..., ::-1]
                    img, _ = random_crop(img, None)

                    internal_state = environment.get_internal_state()

                    imgqueue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, True])
                    [ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent] = latentqueue.get()
                    [netowrk_policy, q] = policyqueue.get()
                    netowrk_policy = netowrk_policy.astype(np.float)
                    eval_qs.append(q)

                    [control, action_info] = make_conrol_actionInfo_from_netowrk_policy(copy.deepcopy(netowrk_policy), min_max_action_position_range, min_max_action_angle_range)


                    effstate = environment._UR.getEndEffectorState()
                    # 아래로 내려가는 동작을 취하고 있을 경우에는 gripper를 우선 무조건 open하고 정해진 위치까지 이동
                    if effstate[0][2] - control[2] > 0:
                        control[6] = 0
                        collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    # 위로 올라가는 동작을 취하고 있을 경우에 그리고 gripper를 닫을 때는 우선 무조건 gripper 부터 닫고 정해진 위치까지 이동
                    if effstate[0][2] - control[2] < 0:
                        if control[6] == 1:
                            previous_action[6] = 1
                            collisionDetected, reward = environment.MoveRobotExactPosition(previous_action, obj_id, True)
                            sleep(0.5)


                    collisionDetected, reward = environment.MoveRobotExactPosition(control, obj_id, True)
                    previous_action = copy.deepcopy(control)

                    if collisionDetected:
                        reward = args.min_reward
                        eval_returns.append(reward)
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

            summary_reward = np.mean(mean_eval_returns)
            summary_q = np.mean(mean_eval_qs)
            if rewardqueue.empty() is False:
                temp = rewardqueue.get()

            if qvalqueue.empty() is False:
                temp = qvalqueue.get()

            rewardqueue.put(summary_reward)
            qvalqueue.put(summary_q)


    endProcCnt.value -= 1



def ManagerProcess(num_actor, weightsQ, gpuidx, bManagerStop, ImgQueueList, LatentQueueList, PolicyQueueList, args, dummy, EndProcCnt):
    # GAN network alloc

    from VAEs.external_pos.external_pos_model import external_pos
    from VAEs.external_vis.external_vis_model import external_vis
    from VAEs.internal_pos.internal_pos_model import internal_pos
    from VAEs.internal_vis.internal_vis_model import internal_vis
    import tensorflow as tf
    from utils.utils import random_crop, ImgNormalize, ImgInverNormalize
    from D4PG.d4pg import D4PG

    _pid = os.getpid()
    print("Actor Manager process create (%d)" % _pid)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:2'):
        ext_pos = external_pos(sess, 'C:/Users/USER/Desktop/Mujoco/IncorlGrasp/VAEs/external_pos/checkpoints')
        ext_vis = external_vis(sess, 'C:/Users/USER/Desktop/Mujoco/IncorlGrasp/VAEs/external_vis/checkpoints')
        int_pos = internal_pos(sess, 'C:/Users/USER/Desktop/Mujoco/IncorlGrasp/VAEs/internal_pos/checkpoints')
        int_vis = internal_vis(sess, 'C:/Users/USER/Desktop/Mujoco/IncorlGrasp/VAEs/internal_vis/checkpoints')

    #TODO : network allocation

    with tf.device('/gpu:1'):
        d4pg = D4PG('gpu1_', args, observation_shape=args.latent_size,
                    action_shape=args.ddpg_action_shape, internal_state_shape=args.ddpg_internal_state_shape,
                    gamma=args.ddpg_gamma, tau=args.ddpg_tau, normalize_observations=args.normalize_observations,
                    batch_size=args.ddpg_batch_size, critic_l2_reg=args.ddpg_q_l2_reg,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr, action_noise_stddev=args.action_noise_stddev)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    d4pg.initialize(sess)

    EndProcCnt.value += 1

    while not bManagerStop.value == 1:
        if not weightsQ.empty():
            sleep(0.001)
            weight = weightsQ.get()

            if len(d4pg.actor.vars) != len(weight):
                NameError('g_var != weight')

            assign_op = []
            #            for i in range(len(weight)):
            for var, target_var in zip(d4pg.actor.vars, weight):
                target_var = tf.convert_to_tensor(target_var)
                assign_op.append(tf.assign(var, target_var))

            sess.run(assign_op)

        is_eval = None

        for i in range(num_actor):
            is_eval = None
            if ImgQueueList[i].empty() is False:

                [img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, is_eval] = ImgQueueList[i].get()     # receive img from actor
                ext_pos_latent = ext_pos.latent(ext_pos_img)
                ext_vis_latent = ext_vis.latent(ext_vis_img)
                int_pos_latent = int_pos.latent(int_pos_img)
                int_vis_latent = int_vis.latent(int_vis_img)

                LatentQueueList[i].put([ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent])
                state = np.concatenate((ext_pos_latent, ext_vis_latent, int_pos_latent, int_vis_latent), axis=1)

                if i == 0 and args.show_recon_of_latent == True:
                    ext_pos_recon = ext_pos.recon_img(ext_pos_img).squeeze()
                    ext_vis_recon = ext_vis.recon_img(ext_vis_img).squeeze()
                    int_pos_recon = int_pos.recon_img(int_pos_img).squeeze()
                    int_vis_recon = int_vis.recon_img(int_vis_img).squeeze()

                    img_show = img.copy()
                    ext_pos_recon_show = ImgInverNormalize(ext_pos_recon, 255)
                    ext_vis_recon_show = ImgInverNormalize(ext_vis_recon, 255)
                    int_pos_recon_show = ImgInverNormalize(int_pos_recon, 255)
                    int_vis_recon_show = ImgInverNormalize(int_vis_recon, 255)

                    ext_vis_recon_show = ext_vis_recon_show[..., ::-1]
                    int_vis_recon_show = int_vis_recon_show[..., ::-1]
                    ext_vis_img = ext_vis_img[..., ::-1]
                    int_vis_img = int_vis_img[..., ::-1]

                    img_recon = np.zeros((128*2 + 10*3, 128*4 + 10*5, 3), dtype=np.uint8)

                    img_recon[10:128+10, 10:128+10, :] = int_vis_img
                    img_recon[148:148+128, 10:128+10, :] = int_vis_recon_show

                    img_recon[10:128+10, 148:148+128, :] = int_pos_img
                    img_recon[148:148+128, 148:148+128, :] = int_pos_recon_show

                    img_recon[10:128+10, 286:286+128, :] = ext_vis_img
                    img_recon[148:148+128, 286:286+128, :] = ext_vis_recon_show

                    img_recon[10:128+10, 424:424+128, :] = ext_pos_img
                    img_recon[148:148+128, 424:424+128, :] = ext_pos_recon_show

                    cv2.imshow('img_recon', img_recon)
                    cv2.imshow('img', img_show)
                    cv2.waitKey(1)

                if is_eval is not None:
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
    def __init__(self, num_actor, replay_buffer, scripted_iter, gpuidx, rootpath, arg_ddpg, dummy, bContinue = False):
        self.args_main = arg_ddpg
        self._bManagerStop = Value('i', 0)
        self._ManagerEndProcCnt = Value('i', 0)
        self.rewards = np.zeros(num_actor) + arg_ddpg.default_reward
        self.qs = np.zeros(num_actor)
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
        self._graspcnt = Value('i', 0)
        self._graspcnt.value = 0

        self._replay_buffer = replay_buffer

        manager = Manager()
        self.min_max_action_position_range = manager.dict()
        self.min_max_action_angle_range = manager.dict()

        # action 을 normalize 하고 denomalize 하기 위한 value들
        z_max = 0.245
        z_min = z_max - 0.25
        x_max = 0.667
        x_min = 0.247
        y_max = 0.28
        y_min = -0.28

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

        self._ImgQueueList = []  # actor process -> actor manager
        self._LatentQueueList = []  # actor process -> actor manager
        self._PolicyQueueList = []  # actor manager -> actor process
        self._RewardQueueList = []  # actor manager -> actor process
        self._QvalQueueList = []  # actor manager -> actor process
        for i in range(self._num_actor):
            self._ImgQueueList.append(Queue())
            self._LatentQueueList.append(Queue())
            self._PolicyQueueList.append(Queue())
            self._RewardQueueList.append(Queue())
            self._QvalQueueList.append(Queue())

        args = (num_actor, self._weight_queue, gpuidx, self._bManagerStop,
                self._ImgQueueList, self._LatentQueueList,
                self._PolicyQueueList, self.args_main, self._ManagerLock, self._ManagerEndProcCnt)
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
            self._ImgQueueList[i].close()
            self._LatentQueueList[i].close()
            self._PolicyQueueList[i].close()
            self._RewardQueueList[i].close()
            self._QvalQueueList[i].close()

    def KillAllProcess(self):
        self._bProcRun = 0
        for proc in self._procList:
            proc.join()

    def StartDistributedActor(self):
        for i in range(self._num_actor):
            proc = Process(target=ActorProcess, args=(self.args_main, self._bActorProcRun, self._ActorEndProcCnt, self._scripted_iter,
                                                      self._replay_buffer, self._ImgQueueList[i], self._LatentQueueList[i],
                                                      self._PolicyQueueList[i], self._RewardQueueList[i],
                                                      self._QvalQueueList[i], self._rootpath, self._ConstructorLock,
                                                      self._graspcnt, self.min_max_action_position_range, self.min_max_action_angle_range))
            self._procList.append(proc)

        for p in self._procList:
            p.start()

        while self._ActorEndProcCnt.value != self._num_actor:
            sleep(1)

    def UpdatePolicy(self, weights):
        self._ActorLock.acquire()
        while not self._weight_queue.empty():
            self._weight_queue.get()

        self._weight_queue.put(weights)
        self._ActorLock.release()

    def getGraspCount(self):
        return self._graspcnt.value

    def get_mean_eval_returns(self):

        rewards = []
        for i in range(self._num_actor):
            if self._RewardQueueList[i].empty() is False:
                self.rewards[i] = self._RewardQueueList[i].get()

            rewards.append(self.rewards[i])

        return rewards

    def get_mean_eval_qs(self):

        qs = []
        for i in range(self._num_actor):
            if self._QvalQueueList[i].empty() is False:
                self.qs[i] = self._QvalQueueList[i].get()

            qs.append(self.qs[i])

        return qs