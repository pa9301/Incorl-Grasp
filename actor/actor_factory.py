import copy
import math
import os
import time
from multiprocessing import Value, Process, Lock, Queue, Manager
from time import sleep

import cv2
import numpy as np


def normalize_action(x, maxim, minim):
    divisor = maxim - minim
    dividend = x - minim
    zero_to_one = dividend / divisor
    minus_one_to_one = zero_to_one * 2 - 1
    return minus_one_to_one


def denormalize_action(minus_one_to_one, maxim, minim):
    zero_to_one = (minus_one_to_one + 1.0) / 2.0
    x = zero_to_one * (maxim - minim) + minim
    return x


def make_action_for_save(action_list, control, action_info, done, collision, range_action_pos, range_action_angle):
    action_list.append(normalize_action(control[0], range_action_pos['x_max'], range_action_pos['x_min']))
    action_list.append(normalize_action(control[1], range_action_pos['y_max'], range_action_pos['y_min']))
    action_list.append(normalize_action(control[2], range_action_pos['z_max'], range_action_pos['z_min']))
    action_list.append(normalize_action(control[3], range_action_angle['roll_max'], range_action_angle['roll_min']))
    action_list.append(action_info[0])
    action_list.append(action_info[2])
    terminal = float(action_info[2] >= 0.5)
    if done or collision:
        terminal = 1.0

    return copy.deepcopy(action_list), terminal


def net_policy_to_control_action(net_policy, range_action_pos, range_action_angle):
    control = [
        denormalize_action(net_policy[0], range_action_pos['x_max'], range_action_pos['x_min']),
        denormalize_action(net_policy[1], range_action_pos['y_max'], range_action_pos['y_min']),
        denormalize_action(net_policy[2], range_action_pos['z_max'], range_action_pos['z_min']),
        denormalize_action(net_policy[3], range_action_angle['roll_max'], range_action_angle['roll_min']),
        math.pi / 2,
        0.0,
        float(net_policy[4] >= 0.5)
    ]

    action_info = np.zeros(3, dtype=np.float)
    action_info[0] = net_policy[4]
    action_info[1] = 1.0 - net_policy[4]
    action_info[2] = net_policy[5]

    return copy.deepcopy(control), copy.deepcopy(action_info)


def actor_process(args, b_proc_run, end_proc_cnt, scripted_iter, replay_buffer, pre_img_queue, img_queue, latent_queue,
                  policy_queue, reward_queue, nb_grasp_success_queue, detect_queue, root_path, construct_lock, graspcnt,
                  actor_lock, range_action_pos, range_action_angle):
    from actor.scripted_policy import ScriptedPolicy
    from simulator.UREnv import URGymEnv
    from utils.utils import write_data

    _scripted_iter = scripted_iter
    _replay_buffer = replay_buffer
    _data_buffer = []
    _pid = os.getpid()

    # simulator param
    print("Actor process simulator create (%d)" % _pid)
    end_proc_cnt.value += 1
    current_count = 0
    construct_lock.acquire()
    environment = URGymEnv(renders=args.use_renderer, obj_batch_count=args.nb_obj_in_tray)
    construct_lock.release()

    print("Actor process run start (%d)" % _pid)

    script_policy = ScriptedPolicy(environment)

    max_grasp_nb = args.max_grasp_nb / 3.5
    epoch = -1
    nb_grasp = 0
    exploration_type_ratio = 1.0
    # 1 : script, 0 : network
    reward_queue.put(args.min_reward)
    nb_grasp_success_queue.put(0.0)

    eval_grasp_info = np.zeros(args.nb_eval_epoch_cycles, dtype=np.float)

    while not b_proc_run.value == 1:
        sleep(0)
        environment.safe_reset()
        epoch += 1

        same_id_count = 0
        temp_id = 0

        for cycle in range(args.nb_epoch_cycles):
            sleep(0)

            done = False
            collision = False
            script_end = False
            remove_object = False
            obj_id = None
            unique_obj_id = None
            state = None
            is_first_step_in_grasp = True
            script_policy.reset()

            # action 을 scripted_policy 로 생성할지 q_network 으로 생성할지에 대한 샘플링
            exploration_type = np.random.binomial(1, exploration_type_ratio, 1)[0]
            if exploration_type == 1:
                exploration_type = 'script'
            elif exploration_type == 0:
                exploration_type = 'network'

            # grasp 1번 수행 ( 여기에는 regrasp 을 포함하고 있음. 따라서 물리적으로 1번이상 파지를 수행할 수 있음)
            # action vector 의 마지막 dimension 이 grasp 을 종료할지에 대한 value 임.
            # 따라서 이 값이 1이면 for 문이 break 되고, 아니면 args.nb_rollout_steps 까지 수행.

            action_info = np.zeros(3, dtype=np.float)

            for step_in_rollout in range(args.nb_rollout_steps):

                control = []
                # 새로운 grasp 를 시작해야 하는 조건 체크
                if done:
                    break

                if collision or not environment.check_object_safe():
                    environment.safe_reset()
                    break

                action_list = []  # action 을 파일로 쓰기 위한 최종 자료형

                # grasp 의 첫번째 step 은 임의의 포지션으로 이동하는 action
                # 첫번째 step 은 로직상 특수하게 간주됨
                if is_first_step_in_grasp:
                    first_action, action_info, block_uid, unique_uid, script_end, out_of_tray \
                        = script_policy.get_scripted_policy()

                    # 아래와 같은 경우에는 새로운 grasp 을 수행
                    if block_uid < 0:
                        environment.safe_reset()
                        break

                    obj_id = block_uid
                    unique_obj_id = unique_uid
                    if temp_id is obj_id:
                        if 2 < same_id_count:
                            environment.safe_reset()
                            print('\x1b[1;30m' + '-->>env _ environment.safeReset() for retry' + '\x1b[0;m')
                            temp_id = 0
                            same_id_count = 0
                            break
                        same_id_count += 1
                    else:
                        same_id_count = 0
                        temp_id = obj_id
                    # 임의의 포지션으로 이동하는 action
                    environment.move_robot_exact_position(first_action, obj_id, False)
                    is_first_step_in_grasp = False
                    # 이렇게 첫번째 step 은 design 된 action 이기 때문에 RL 학습에는 사용되지 않음.
                    # 따라서 저장하지 않고 continue
                    continue

                # is_first_step_in_grasp 이 False 일 경우 여기에서부터 시작
                # action 을 수행하기 이전의 사전 state
                # 특정 물체만 highlight 할 경우 obj_unique_id 가 필요함
                rgb_img_360, rgb_img_256, rgb_crop_for_gan360 \
                    = environment.get_observation(_pid)
                internal_state = environment.get_internal_state()
                # action 을 생성
                pre_img_queue.put([rgb_img_360, rgb_img_256, rgb_crop_for_gan360, internal_state, None])
                boxes, scores, classes, num, fake_real_360, arm_seg_256 = detect_queue.get()
                _, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _ \
                    = environment.get_extended_observation(_pid, boxes, scores, classes, fake_real_360, arm_seg_256,
                                                           unique_obj_id, True)

                if exploration_type == 'script':
                    img_queue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, None])
                    control, action_info, _, _, script_end, out_of_tray = script_policy.get_scripted_policy()
                    state = latent_queue.get()

                elif exploration_type == 'network':
                    img_queue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, False])
                    state = latent_queue.get()
                    network_policy, _ = policy_queue.get()
                    network_policy = network_policy.astype(np.float)
                    control, action_info = net_policy_to_control_action(
                        copy.deepcopy(network_policy), range_action_pos, range_action_angle
                    )

                # action 을 수행
                collision_detected, reward = environment.move_robot_exact_position(control, obj_id, True)

                # collision 체크하여 collision 했으면 새로운 grasp 을 수행
                if collision_detected:
                    collision = True

                # grasp 의 종료에 관련된 상황 체크
                if exploration_type == 'script':
                    # script 가 종료일 경우에만 reward 를 받는다
                    if script_end:
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1
                        done = True
                        terminate_episode = np.random.uniform()
                        # action vector 의 마지막 dimension 이 1 일 경우에만 reward 1을 받을 수 있음
                        if terminate_episode > args.min_ratio_for_terminate_epsode:
                            action_info[2] = (np.random.uniform() / 2.0) + 0.5  # episode 를 종료한다는 결정을 내림
                            if reward == 1.0:
                                remove_object = True
                        else:
                            reward = args.min_reward
                    else:
                        reward = args.min_reward

                elif exploration_type == 'network':
                    eff_pos = environment.UR.get_end_effector_state()[0]

                    # end effector 의 z 값이 일정 높이 이상일 경우만 grasp 성공이 될 수 있음
                    if eff_pos[2] < args.min_z_value_for_grasp_success:
                        reward = args.min_reward

                    if action_info[2] >= 0.5:  # q network 이 grasp 를 종료하고자 할 때
                        done = True
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1

                        # 만약 grasp 성공했으면 물체를 없앨것
                        if reward == 1.0:
                            remove_object = True
                            print('grasp success in exploration')
                    else:  # q network 이 grasp 를 종료하고자 하지 않을 때
                        reward = args.min_reward

                        # step_in_rollout 이 maximum step 일 때
                    if step_in_rollout == args.nb_rollout_steps - 1:
                        # grasp 횟수를 증가
                        nb_grasp = nb_grasp + 1
                        break

                # grasp 의 종료에 관련된 상황 체크

                # grasp 성공했으면 물체를 삭제
                if remove_object:
                    environment.remove_target_object(obj_id)

                # action 을 수행하고 난 사후 상태
                new_rgb_img_360, new_rgb_img_256, new_rgb_crop_for_gan360 \
                    = environment.get_observation(_pid)

                new_internal_state = environment.get_internal_state()
                pre_img_queue.put([
                    new_rgb_img_360,
                    new_rgb_img_256,
                    new_rgb_crop_for_gan360,
                    new_internal_state, None
                ])
                boxes, scores, classes, num, fake_real_360, arm_seg_256 = detect_queue.get()
                _, new_img, new_ext_pos_img, new_ext_vis_img, new_int_pos_img, new_int_vis_img, _, _ \
                    = environment.get_extended_observation(
                        _pid, boxes, scores, classes, fake_real_360, arm_seg_256, unique_obj_id, True
                    )

                img_queue.put([
                    new_img,
                    new_ext_pos_img,
                    new_ext_vis_img,
                    new_int_pos_img,
                    new_int_vis_img,
                    new_internal_state,
                    None
                ])
                new_state = latent_queue.get()

                action_list, terminal = make_action_for_save(action_list, copy.deepcopy(control),
                                                             copy.deepcopy(action_info), done, collision,
                                                             range_action_pos,
                                                             range_action_angle)

                current_count += 1
                path = write_data(root_path, state.tolist(), new_state.tolist(), action_list, internal_state,
                                  new_internal_state, reward, terminal, current_count)

                path = path.replace('\\', '/')
                _replay_buffer.store(path)

            graspcnt.value += 1
            ratio = (max_grasp_nb - graspcnt.value) / max_grasp_nb
            if ratio < 0.05:
                ratio = 0.05
            exploration_type_ratio = ratio

        if args.eval is True and epoch % 5 == 0 and epoch > 10:
            print('start eval!!!')

            environment.safe_reset()
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
                remove_object = False

                script_policy.reset()

                for step_in_rollout in range(args.max_step_for_test_episode):
                    # 새로운 grasp 를 시작해야 하는 조건 체크
                    if done:
                        break
                    if environment.check_object_safe() is False:
                        environment.safe_reset()
                        break

                    if is_first_step_in_grasp:
                        first_action, action_ext, block_uid, unique_uid, script_end, out_of_tray \
                            = script_policy.get_scripted_policy()

                        # 아래와 같은 경우에는 새로운 grasp 을 수행
                        if block_uid < 0:
                            environment.safe_reset()
                            break

                        obj_id = block_uid
                        unique_obj_id = unique_uid
                        if temp_id is obj_id:
                            if 2 < same_id_count:
                                environment.safe_reset()
                                print('\x1b[1;30m' + '-->>env _ environment.safeReset() for retry' + '\x1b[0;m')
                                temp_id = 0
                                same_id_count = 0
                                break
                            same_id_count += 1
                        else:
                            same_id_count = 0
                            temp_id = obj_id
                        # 임의의 포지션으로 이동하는 action
                        environment.move_robot_exact_position(first_action, obj_id, False)
                        is_first_step_in_grasp = False
                        # 이렇게 첫번째 step 은 design 된 action 이기 때문에 RL 학습에는 사용되지 않음.
                        # 따라서 저장하지 않고 continue
                        continue

                    # is_first_step_in_grasp 이 False 일 경우 여기에서부터 시작
                    # action 을 수행하기 이전의 사전 state
                    # 특정 물체만 highlight 할 경우 obj_unique_id 가 필요함
                    rgb_img_360, rgb_img_256, rgb_crop_for_gan360 \
                        = environment.get_observation(_pid)
                    internal_state = environment.get_internal_state()
                    pre_img_queue.put([rgb_img_360, rgb_img_256, rgb_crop_for_gan360, internal_state, None])
                    boxes, scores, classes, num, fake_real_360, arm_seg_256 = detect_queue.get()

                    _, img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, _, _ \
                        = environment.get_extended_observation(_pid, boxes, scores, classes, fake_real_360, arm_seg_256,
                                                               unique_obj_id, True)

                    img_queue.put([img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, True])

                    latent_queue.get()
                    network_policy, q = policy_queue.get()
                    network_policy = network_policy.astype(np.float)
                    eval_qs.append(q)

                    control, action_info = net_policy_to_control_action(
                        copy.deepcopy(network_policy), range_action_pos, range_action_angle
                    )

                    # action 을 수행
                    collision_detected, reward = environment.move_robot_exact_position(control, obj_id, True)

                    if collision_detected:
                        reward = args.min_reward
                        eval_returns.append(reward)
                        environment.safe_reset()
                        break

                    eff_pos = environment.UR.get_end_effector_state()[0]

                    # end effector 의 z 값이 일정 높이 이상일 경우만 grasp 성공이 될 수 있음
                    if eff_pos[2] < args.min_z_value_for_grasp_success:
                        reward = args.min_reward

                    if action_info[2] >= 0.5:  # q network 이 grasp 를 종료하고자 할 때
                        eval_returns.append(reward)
                        done = True

                        # 만약 grasp 성공했으면 물체를 없앨것
                        if reward == 1.0:
                            remove_object = True
                            eval_grasp_info[eval_cycle] = 1.0
                            print('grasp success in test!!!')
                    else:  # q network 이 grasp 를 종료하고자 하지 않을 때
                        reward = args.min_reward

                    eval_returns.append(reward)

                    if remove_object:
                        environment.remove_target_object(obj_id)
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
            actor_lock.acquire()
            try:
                reward_queue.put(summary_reward)
                nb_grasp_success_queue.put(summary_nb_grasp_success)
            finally:
                actor_lock.release()

    end_proc_cnt.value -= 1


def manager_process(num_actor, weights_queue, b_manager_stop, pre_img_queue_list, img_queue_list, latent_queue_list,
                    policy_queue_list, detected_queue_list, args, lock, end_proc_cnt):
    # GAN network alloc
    from VAEs.VAE_model import VAE
    from object_detection.detector_model import Detector
    from arm_segmentation.arm_segmentation_model import ArmSegmentation
    from GAN.gan_model import GAN
    from D4PG.d4pg import D4PG
    from D4PG.noise import NormalActionNoise
    import tensorflow as tf
    from utils.utils import img_invert_normalize, refine_segmented_image_by_connected_component

    _pid = os.getpid()
    print("Actor Manager process create (%d)" % _pid)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:2'):
        detector_model = Detector(24)

    with tf.device('/gpu:1'):
        arm_segmentation_model = ArmSegmentation(args.arm_segmentation_dir)

    with tf.device('/gpu:0'):
        gan_model = GAN(sess, args.GAN_dir)

    with tf.device('/gpu:2'):
        vae_model = VAE(sess, args.SRL_dir)

    action_noise = None
    param_noise = None

    nb_actions = args.ddpg_action_shape
    for current_noise_type in args.noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    with tf.device('/gpu:1'):
        d4pg = D4PG('gpu1_', args, observation_shape=args.latent_size, action_shape=args.ddpg_action_shape,
                    internal_state_shape=args.ddpg_internal_state_shape, param_noise=param_noise,
                    action_noise=action_noise, gamma=args.ddpg_gamma, tau=args.ddpg_tau,
                    normalize_observations=args.normalize_observations, batch_size=args.ddpg_batch_size,
                    critic_l2_reg=args.ddpg_q_l2_reg, actor_lr=args.actor_lr, critic_lr=args.critic_lr)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    d4pg.initialize(sess)

    end_proc_cnt.value += 1

    while not b_manager_stop.value == 1:
        sleep(0.01)
        weight = None
        lock.acquire()

        try:
            if not weights_queue.empty():
                weight = weights_queue.get()
        finally:
            lock.release()

        if weight is not None:
            with tf.device('/gpu:1'):
                start_time = time.time()
                d4pg.update_weight(weight)
                print("sess.run: %s seconds ---" % (time.time() - start_time))

        for i in range(num_actor):

            if not pre_img_queue_list[i].empty():
                # receive img from actor
                rgb_img_360, rgb_img_256, rgb_crop_for_gan360, _, _ = pre_img_queue_list[i].get()
                boxes, scores, classes, num = detector_model.detection(rgb_img_360)

                arm_seg_256 = arm_segmentation_model.seg_img(rgb_img_256)  # : 0~2, 1*256*256
                arm_seg_256 = np.squeeze(arm_seg_256)
                arm_seg_256 = refine_segmented_image_by_connected_component(arm_seg_256, 2, 100)

                fake_real_temp = gan_model.fake_real_img(rgb_crop_for_gan360)  # : 이미지 저장할 세그먼트
                fake_real_temp = np.squeeze(fake_real_temp)
                fake_real_360 = img_invert_normalize(fake_real_temp, 255)

                detected_queue_list[i].put([boxes, scores, classes, num, fake_real_360, arm_seg_256])

            if not img_queue_list[i].empty():

                img, ext_pos_img, ext_vis_img, int_pos_img, int_vis_img, internal_state, is_eval \
                    = img_queue_list[i].get()  # receive img from actor

                vae_img = np.concatenate((int_pos_img, int_vis_img, ext_pos_img, ext_vis_img), axis=2)
                latent = vae_model.latent(vae_img)
                latent_queue_list[i].put(latent)
                state = latent

                if i == 0 and args.show_recon_of_latent is True:
                    img_recon = vae_model.recon_img(vae_img).squeeze()

                    img_show = img.copy()
                    img_show = img_show[..., ::-1]
                    recon_show = img_invert_normalize(img_recon, 255)

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
                    action, q = d4pg.policy(state, internal_state, is_eval=is_eval, compute_q=True)
                    policy_queue_list[i].put([action, q])

        sleep(0.01)

    print('manager close')


# Manager process 를 생성하고, Actor 들을 생성한다.
# Manager 는 Session 을 가지고, Actor 들의 입력을 처리해주는 역할을 한다.
class ActorFactory:
    def __init__(self, num_actor, replay_buffer, scripted_iter, gpuidx, root_path, arg_ddpg, nb_grasp):
        self.args_main = arg_ddpg
        self._bManagerStop = Value('i', 0)
        self._ManagerEndProcCnt = Value('i', 0)
        self.rewards = np.zeros(num_actor) + arg_ddpg.default_reward
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
        self._root_path = root_path
        self._graspcnt = Value('i', nb_grasp)
        self._graspcnt.value = nb_grasp

        self._replay_buffer = replay_buffer

        manager = Manager()
        self.min_max_action_position_range = manager.dict()
        self.min_max_action_angle_range = manager.dict()

        # action 을 normalize 하고 denormalize 하기 위한 value 들.
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

        self.min_max_action_position_range = {'z_max': z_max, 'z_min': z_min,
                                              'x_max': x_max, 'x_min': x_min,
                                              'y_max': y_max, 'y_min': y_min}
        self.min_max_action_angle_range = {'roll_min': roll_min, 'roll_max': roll_max,
                                           'pitch_min': pitch_min, 'pitch_max': pitch_max,
                                           'yaw_min': yaw_min, 'yaw_max': yaw_max}

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

        args = (
            self._num_actor, self._weight_queue, self._bManagerStop,
            self._PreImgQueueList, self._ImgQueueList, self._LatentQueueList,
            self._PolicyQueueList, self._DetectedQueueList,
            self.args_main, self._ManagerLock, self._ManagerEndProcCnt)
        self._manager_process = Process(target=manager_process, args=args)
        self._manager_process.start()

        while self._ManagerEndProcCnt.value != 1:
            sleep(1)

        self.start_distributed_actor()

    def __del__(self):
        self._bManagerStop.value = 1
        self._manager_process.join()
        self._weight_queue.close()

        self.kill_all_process()

        for i in range(self._num_actor):
            self._PreImgQueueList[i].close()
            self._ImgQueueList[i].close()
            self._LatentQueueList[i].close()
            self._PolicyQueueList[i].close()
            self._RewardQueueList[i].close()
            self._DetectedQueueList[i].close()
            self._NbGraspSuccessQueueList[i].close()

    def kill_all_process(self):
        self._bActorProcRun = 0
        for proc in self._procList:
            proc.join()

    def start_distributed_actor(self):
        for i in range(self._num_actor):
            proc = Process(target=actor_process,
                           args=(self.args_main, self._bActorProcRun, self._ActorEndProcCnt, self._scripted_iter,
                                 self._replay_buffer, self._PreImgQueueList[i], self._ImgQueueList[i],
                                 self._LatentQueueList[i],
                                 self._PolicyQueueList[i], self._RewardQueueList[i],
                                 self._NbGraspSuccessQueueList[i], self._DetectedQueueList[i], self._root_path,
                                 self._ConstructorLock,
                                 self._graspcnt, self._ActorLock, self.min_max_action_position_range,
                                 self.min_max_action_angle_range))

            self._procList.append(proc)

        for p in self._procList:
            p.start()

        while self._ActorEndProcCnt.value != self._num_actor:
            sleep(1)

    def get_grasp_count(self):
        self._ActorLock.acquire()
        try:
            graspcnt = self._graspcnt.value
        finally:
            self._ActorLock.release()

        return graspcnt

    def update_policy(self, weights):

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
