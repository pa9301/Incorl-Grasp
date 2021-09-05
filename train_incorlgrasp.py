import argparse
import os
import sys
from time import sleep

sys.path.append(os.path.abspath("./"))

import tensorflow as tf
from actor.actor_factory import ActorFactory
from memory.memory_factory import MemoryFactory
from D4PG.d4pg import D4PG
import D4PG.saver_util as su
from D4PG.noise import *
import copy
import json

parser = argparse.ArgumentParser(description='')

#  test
parser.add_argument('--eval', dest='eval', type=bool, default=True, help='')
parser.add_argument('--nb_eval_epoch_cycles', dest='nb_eval_epoch_cycles', type=int, default=10, help='')

# main learning loop
parser.add_argument('--nb_epoch_cycles', dest='nb_epoch_cycles', type=int, default=10, help='')
parser.add_argument('--nb_rollout_steps', dest='nb_rollout_steps', type=int, default=10, help='')

parser.add_argument('--nb_grasp_for_start_learn', dest='nb_grasp_for_start_learn', type=int, default=30, help='')
parser.add_argument('--nb_train_steps', dest='nb_train_steps', type=int, default=2, help='')

# learning ddpg
parser.add_argument('--ddpg_batch_size', dest='ddpg_batch_size', type=int, default=64, help='')
parser.add_argument('--ddpg_action_shape', dest='ddpg_action_shape', type=int, default=6, help='')
parser.add_argument('--ddpg_internal_state_shape', dest='ddpg_internal_state_shape', type=int, default=2, help='')
parser.add_argument('--ddpg_gamma', dest='ddpg_gamma', type=float, default=0.99, help='weight for future value')
parser.add_argument('--ddpg_tau', dest='ddpg_tau', type=float, default=0.01, help='mixing parameter for target network')
parser.add_argument('--actor_lr', dest='actor_lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--critic_lr', dest='critic_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ddpg_q_l2_reg', dest='ddpg_q_l2_reg', type=float, default=0.0, help='regularization parameter')
parser.add_argument('--default_reward', dest='default_reward', type=float, default=-0.05, help='')

# learning d4pg
parser.add_argument('--V_MIN', dest='V_MIN', type=float, default=-0.05,
                    help='Lower bound of critic value output distribution')
parser.add_argument('--V_MAX', dest='V_MAX', type=float, default=1.0,
                    help='Upper bound of critic value output distribution '
                         '(V_min and V_max should be chosen based on the range of '
                         'normalized reward values in the chosen env)')
parser.add_argument('--NUM_ATOMS', dest='NUM_ATOMS', type=int, default=51,
                    help='Number of atoms in output layer of distributional critic')

# beta vae
parser.add_argument('--latent_size', dest='latent_size', type=int, default=256, help='')

# etc.
parser.add_argument('--max-grasp-nb', type=int, default='50000')
parser.add_argument('--noise-type', type=str, default='normal_0.05')
parser.add_argument('--nb-actors', dest='nb_actors', type=int, default=5)  # 몇개 액터
parser.add_argument('--offline-data-path', dest='offline_data_path', type=str, default='../../data_sim')  # 저장위치
parser.add_argument('--online-memory-size', dest='online_memory_size', type=int, default=10000)
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./d4pg/checkpoint',
                    help='models are saved here')
parser.add_argument('--SRL_dir', dest='SRL_dir', default='./VAEs/checkpoints', help='models are saved here')

parser.add_argument('--nb_obj_in_tray', dest='nb_obj_in_tray', type=int, default=20, help='')  # : 트레이 내 오브젝트 갯수 조절
parser.add_argument('--use_renderer', dest='use_renderer', type=bool, default=True, help='')
parser.add_argument('--min_reward', dest='min_reward', type=float, default=-0.05, help='')
parser.add_argument('--normalize_observations', dest='normalize_observations', type=bool, default=False, help='')
parser.add_argument('--show_recon_of_latent', dest='show_recon_of_latent', type=bool, default=False, help='')
parser.add_argument('--min_ratio_for_terminate_epsode', dest='min_ratio_for_terminate_epsode', type=float, default=0.3,
                    help='')
parser.add_argument('--min_z_value_for_grasp_success', dest='min_z_value_for_grasp_success', type=float, default=0.24,
                    help='')
parser.add_argument('--max_step_for_test_episode', dest='max_step_for_test_episode', type=int, default=15, help='')

parser.add_argument('--arm-segmentation-dir', dest='arm_segmentation_dir', type=str,
                    default='./arm_segmentation/checkpoints')
parser.add_argument('--GAN-dir', dest='GAN_dir', type=str, default='./GAN/checkpoints')

args = parser.parse_args()


def main():
    memory_factory = MemoryFactory(args.offline_data_path + '/file', args.online_memory_size)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    action_noise = None
    param_noise = None

    log_path = args.checkpoint_dir + "/log"
    log_file_path = args.checkpoint_dir + "/log/log.txt"
    nb_grasp = 0
    training_step = 0
    if os.path.exists(log_path):
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                log_dict = json.load(f)
                nb_grasp = log_dict['nb_grasp']
                training_step = log_dict['training_step']
        else:
            log_dict = {'nb_grasp': nb_grasp, 'training_step': training_step}
            with open(log_file_path, 'w') as f:
                f.write(json.dumps(log_dict))
    else:
        log_dict = {'nb_grasp': nb_grasp, 'training_step': training_step}
        os.mkdir(log_path)
        with open(log_file_path, 'w') as f:
            f.write(json.dumps(log_dict))

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

    with tf.device('/gpu:0'):
        d4pg = D4PG('gpu0_', args, observation_shape=args.latent_size, action_shape=args.ddpg_action_shape,
                    internal_state_shape=args.ddpg_internal_state_shape, param_noise=param_noise,
                    action_noise=action_noise, gamma=args.ddpg_gamma, tau=args.ddpg_tau,
                    normalize_observations=args.normalize_observations, batch_size=args.ddpg_batch_size,
                    critic_l2_reg=args.ddpg_q_l2_reg, actor_lr=args.actor_lr, critic_lr=args.critic_lr)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    d4pg.initialize(sess)

    saver = su.SaverUtil(sess, args.checkpoint_dir)
    saver.load_latest_checkpoint_or_init_if_none()

    # tensorboard
    tensorboard_reward = tf.placeholder(tf.float32, name='tensorboard_reward')
    tensorboard_loss = tf.placeholder(tf.float32, name='tensorboard_loss')
    # tensorboard_q = tf.placeholder(tf.float32, name='tensorboard_q')
    tensorboard_nb_success = tf.placeholder(tf.float32, name='tensorboard_nb_success')
    tensorboard_std_success = tf.placeholder(tf.float32, name='tensorboard_std_success')
    # reward_summary = tf.summary.scalar(name='reward', tensor=tensorboard_reward)
    # loss_summary = tf.summary.scalar(name='loss', tensor=tensorboard_loss)
    # q_summary = tf.summary.scalar(name='q', tensor=tensorboard_q)
    # nb_success_summary = tf.summary.scalar(name='nb_success', tensor=tensorboard_nb_success)
    # std_success_summary = tf.summary.scalar(name='std_success', tensor=tensorboard_std_success)

    merged = tf.summary.merge_all()
    writer_per_grasp = tf.summary.FileWriter(args.checkpoint_dir + '/log_per_grasp', sess.graph)
    writer_per_iteration = tf.summary.FileWriter(args.checkpoint_dir + '/log_per_iteration', sess.graph)

    sleep(15)
    actor_var = sess.run(d4pg.actor.vars)
    print('actor_var')
    actor_factory = ActorFactory(
        args.nb_actors, memory_factory.online_memory, -1, None, args.offline_data_path, args, nb_grasp
    )
    print('ActorFactory')
    sleep(10)
    actor_factory.update_policy(copy.deepcopy(actor_var))
    sleep(10)
    print('UpdatePolicy')
    nb_grasp = actor_factory.get_grasp_count()
    print('nb_grasp:', nb_grasp)

    while True:
        sleep(0.1)
        nb_grasp = actor_factory.get_grasp_count()
        losses = []

        if nb_grasp > args.nb_grasp_for_start_learn:
            for _ in range(args.nb_train_steps):
                training_data = memory_factory.get_data(args.ddpg_batch_size)
                loss = d4pg.train(training_data)
                d4pg.update_target_net()
                losses.append(loss)

                training_step = training_step + 1

                print('training_step :', training_step, 'nb_grasp :', nb_grasp)

        if training_step % (args.nb_train_steps * 200) == 0 and training_step > 0:
            # weight update
            actor_var = sess.run(d4pg.actor.vars)
            actor_factory.update_policy(copy.deepcopy(actor_var))

        if training_step % (args.nb_train_steps * 1000) == 0 and training_step > 0:
            saver.force_save(training_step)
            log_dict = {'nb_grasp': nb_grasp, 'training_step': training_step}
            with open(log_file_path, 'w') as f:
                f.write(json.dumps(log_dict))

        if training_step % (args.nb_train_steps * 25) == 0 and training_step > 0:
            summary_loss = np.mean(losses)

            mean_eval_returns = actor_factory.get_mean_eval_returns()
            summary_reward = np.mean(mean_eval_returns)

            nb_grasp_success = actor_factory.get_mean_grasp_success()
            mean_nb_grasp_success = np.mean(nb_grasp_success)
            std_nb_grasp_success = np.std(nb_grasp_success)

            feed_dict = {tensorboard_loss: summary_loss, tensorboard_reward: summary_reward,
                         tensorboard_nb_success: mean_nb_grasp_success, tensorboard_std_success: std_nb_grasp_success}

            summary = sess.run(merged, feed_dict=feed_dict)
            writer_per_grasp.add_summary(summary, nb_grasp)
            writer_per_iteration.add_summary(summary, training_step)


if __name__ == '__main__':
    main()
