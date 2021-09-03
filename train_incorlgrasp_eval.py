
import argparse
import os
import sys
from time import sleep
sys.path.append(os.path.abspath("./"))

import numpy as np
import tensorflow as tf
from actor.actor_factory import ActorFactory
from memory.MemoryFactory import MemoryFactory
from D4PG.d4pg import D4PG
import D4PG.saver_util as su
from D4PG.models import Actor, Critic
from D4PG.noise import *
import copy

parser = argparse.ArgumentParser(description='')

#  test
parser.add_argument('--eval', dest='eval', type=bool, default=True, help='')
parser.add_argument('--nb_eval_epoch_cycles', dest='nb_eval_epoch_cycles', type=int, default=50, help='')

# main learning loop
parser.add_argument('--nb_epochs', dest='nb_epochs', type=int, default=100000, help='# of epoch')
parser.add_argument('--nb_epoch_cycles', dest='nb_epoch_cycles', type=int, default=10, help='')
parser.add_argument('--nb_rollout_steps', dest='nb_rollout_steps', type=int, default=10, help='')

parser.add_argument('--nb_max_regrasp', dest='nb_max_regrasp', type=int, default=3, help='')
parser.add_argument('--nb_grasp_for_start_learn', dest='nb_grasp_for_start_learn', type=int, default=30, help='')
parser.add_argument('--nb_train_steps', dest='nb_train_steps', type=int, default=2, help='')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=2000, help='')

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
parser.add_argument('--V_MIN', dest='V_MIN', type=float, default=-0.05, help='Lower bound of critic value output distribution')
parser.add_argument('--V_MAX', dest='V_MAX', type=float, default=1.0, help='Upper bound of critic value output distribution (V_min and V_max should be chosen based on the range of normalised reward values in the chosen env)')
parser.add_argument('--NUM_ATOMS', dest='NUM_ATOMS', type=int, default=51, help='Number of atoms in output layer of distributional critic')


# beta vae
parser.add_argument('--ext_pos_size', dest='ext_pos_size', type=int, default=64, help='')
parser.add_argument('--ext_vis_size', dest='ext_vis_size', type=int, default=192, help='')
parser.add_argument('--int_pos_size', dest='int_pos_size', type=int, default=64, help='')
parser.add_argument('--int_vis_size', dest='int_vis_size', type=int, default=128, help='')
parser.add_argument('--latent_size', dest='latent_size', type=int, default=448, help='')

# etc.
parser.add_argument('--noise-type', type=str, default='normal_0.05')
parser.add_argument('--nb-actors', dest='nb_actors', type=int, default=1)                                                                                   # 몇개 액터
parser.add_argument('--offline-data-path', dest='offline_data_path', type=str, default='C:/Users/USER/Desktop/data')     # 저장위치
parser.add_argument('--online-memory-size', dest='online_memory_size', type=int, default=10000)
parser.add_argument('--action-noise-stddev', type=float, default='0.05')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./d4pg/checkpoint', help='models are saved here')
parser.add_argument('--image_height', dest='image_height', type=int, default=472, help='')
parser.add_argument('--image_width', dest='image_width', type=int, default=472, help='')


parser.add_argument('--nb_obj_in_tray', dest='nb_obj_in_tray', type=int, default=20, help='') # : 트레이 내 오브젝트 갯수 조절
parser.add_argument('--use_renderer', dest='use_renderer', type=bool, default=True, help='')
parser.add_argument('--min_reward', dest='min_reward', type=float, default=-0.05, help='')
parser.add_argument('--normalize_observations', dest='normalize_observations', type=bool, default=False, help='')
parser.add_argument('--show_recon_of_latent', dest='show_recon_of_latent', type=bool, default=False, help='')
parser.add_argument('--min_ratio_for_terminate_epsode', dest='min_ratio_for_terminate_epsode', type=float, default=0.3, help='')
parser.add_argument('--min_z_value_for_grasp_success', dest='min_z_value_for_grasp_success', type=float, default=0.24, help='')
parser.add_argument('--max_step_for_test_episode', dest='max_step_for_test_episode', type=int, default=15, help='')

args = parser.parse_args()

def main():

    memory_factory = MemoryFactory(args.offline_data_path + '/file', args.online_memory_size)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

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
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    model = None

    with tf.device('/gpu:0'):
        d4pg = D4PG('gpu0_', args, observation_shape=args.latent_size,
                        action_shape=args.ddpg_action_shape, internal_state_shape=args.ddpg_internal_state_shape,
                        param_noise=param_noise, action_noise=action_noise,
                         gamma=args.ddpg_gamma, tau=args.ddpg_tau, normalize_observations=args.normalize_observations,
                         batch_size=args.ddpg_batch_size, critic_l2_reg=args.ddpg_q_l2_reg,
                         actor_lr=args.actor_lr, critic_lr=args.critic_lr)


    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    d4pg.initialize(sess)

    saver = su.SaverUtil(sess, args.checkpoint_dir)
    saver.load_latest_ckpt_or_init_if_none()

    # tensorboard
    tensorboard_reward = tf.placeholder(tf.float32, name='tensorboard_reward')
    tensorboard_loss = tf.placeholder(tf.float32, name='tensorboard_loss')
    tensorboard_q = tf.placeholder(tf.float32, name='tensorboard_q')
    reward_summary = tf.summary.scalar(name='reward', tensor=tensorboard_reward)
    loss_summary = tf.summary.scalar(name='loss', tensor=tensorboard_loss)
    q_summary = tf.summary.scalar(name='q', tensor=tensorboard_q)
    writer_loss = tf.summary.FileWriter(args.checkpoint_dir + '/loss', sess.graph)
    writer_reward = tf.summary.FileWriter(args.checkpoint_dir + '/reward', sess.graph)
    writer_q = tf.summary.FileWriter(args.checkpoint_dir + '/q', sess.graph)


    sleep(10)
    actor_var = sess.run(d4pg.actor.vars)
    print('actor_var')
    actorfactory = ActorFactory(args.nb_actors, memory_factory.onlineMemory, -1, None, args.offline_data_path, args, None)
    print('ActorFactory')
    sleep(10)
    actorfactory.UpdatePolicy(copy.deepcopy(actor_var))
    sleep(10)
    print('UpdatePolicy')
    training_step = 0
    nb_grasp = actorfactory.getGraspCount()
    print('nb_grasp:', nb_grasp)

if __name__ == '__main__':
    main()