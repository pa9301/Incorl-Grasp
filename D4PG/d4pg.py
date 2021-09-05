import numpy as np
import tensorflow as tf

from D4PG.l2_projection import _l2_project
from D4PG.models import Actor, Critic


def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    return tf.group(*init_updates), tf.group(*soft_updates)


hidden_sizes = (256, 256)


class D4PG(object):

    def __init__(self, name, args, observation_shape, action_shape, internal_state_shape, param_noise=None,
                 action_noise=None, gamma=0.99, tau=0.001, normalize_observations=True, batch_size=128,
                 observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf), critic_l2_reg=0.,
                 actor_lr=1e-4, critic_lr=1e-3):
        # Inputs.
        self.args = args
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + (observation_shape,), name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + (observation_shape,), name='obs1')
        self.internal_state0 \
            = tf.placeholder(tf.float32, shape=(None,) + (internal_state_shape,), name='internal_state0')
        self.internal_state1 \
            = tf.placeholder(tf.float32, shape=(None,) + (internal_state_shape,), name='internal_state1')
        #        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.terminals = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + (action_shape,), name='actions')

        # Parameters.
        self.action_shape = action_shape
        self.internal_state_shape = internal_state_shape

        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_observations
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg

        # Atom values of target network with Bellman update applied
        self.target_atoms_ph = tf.placeholder(tf.float32, shape=(None, self.args.NUM_ATOMS))
        # Future Z-distribution - for critic training
        self.target_Z_ph = tf.placeholder(tf.float32, shape=(None, self.args.NUM_ATOMS))
        # Gradient of critic's value output wrt action input - for actor training
        self.action_grads_ph = tf.placeholder(tf.float32, shape=(None,) + (action_shape,))

        self.critic = Critic(
            self.obs0, self.internal_state0, self.actions, hidden_sizes=hidden_sizes,
            name=name + 'critic', v_min=args.V_MIN, v_max=args.V_MAX, num_atoms=args.NUM_ATOMS
        )
        self.actor = Actor(
            self.obs0, self.internal_state0, args.ddpg_action_shape, hidden_sizes=hidden_sizes,
            name=name + 'actor'
        )

        self.target_critic = Critic(
            self.obs1, self.internal_state1, self.actions, hidden_sizes=hidden_sizes,
            name=name + 'target_critic', v_min=args.V_MIN, v_max=args.V_MAX, num_atoms=args.NUM_ATOMS)
        self.target_actor = Actor(
            self.obs1, self.internal_state1, args.ddpg_action_shape, hidden_sizes=hidden_sizes,
            name=name + 'target_actor'
        )

        # ========== Setup actor optimizer. ==========
        self.optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.grads = tf.gradients(self.actor.output, self.actor.trainable_vars, -self.action_grads_ph)
        # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
        self.grads_scaled = list(map(
            lambda x: tf.divide(x, self.args.ddpg_batch_size),
            self.grads
        ))
        self.actor_op = self.optimizer.apply_gradients(zip(self.grads_scaled, self.actor.trainable_vars))

        # ========== Setup critic optimizer. ==========
        self.optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # Project the target distribution onto the bounds of the original network
        target_z_projected = _l2_project(self.target_atoms_ph, self.target_Z_ph, self.critic.z_atoms)

        self.critic_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.critic.output_logits, labels=tf.stop_gradient(target_z_projected)
        )
        self.critic_mean_loss = tf.reduce_mean(self.critic_loss)
        self.critic_total_loss = self.critic_mean_loss
        self.critic_op = self.optimizer.minimize(self.critic_total_loss, var_list=self.critic.trainable_vars)

        # ========== Setup target network updates. ==========
        actor_init_updates, actor_soft_updates = get_target_updates(
            self.actor.vars, self.target_actor.vars, self.tau
        )
        critic_init_updates, critic_soft_updates = get_target_updates(
            self.critic.vars, self.target_critic.vars, self.tau
        )
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def policy(self, obs, internal_state, is_eval=False, compute_q=True):

        feed_dict = {self.obs0: obs, self.internal_state0: internal_state}
        action = self.sess.run(self.actor.output, feed_dict=feed_dict)
        q = None
        if compute_q:
            feed_dict = {self.obs0: obs, self.internal_state0: internal_state, self.actions: action}
            q = self.sess.run(self.critic.q_val, feed_dict=feed_dict)

        action = action.flatten()
        if self.action_noise is not None and is_eval is False:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def train(self, training_data):
        # Get a batch.
        obs0 = training_data['state before']
        obs0 = np.squeeze(obs0)
        actions0 = training_data['action']
        actions0 = np.squeeze(actions0)
        is0 = training_data['internal state before']
        is0 = np.squeeze(is0)
        obs1 = training_data['state after']
        obs1 = np.squeeze(obs1)
        is1 = training_data['internal state after']
        is1 = np.squeeze(is1)
        rewards = training_data['reward']
        terminals = training_data['terminal']

        int_terminals = np.squeeze(terminals.astype(np.int))
        mask_terminals = np.ma.make_mask(int_terminals)
        actions1 = self.sess.run(self.target_actor.output, {self.obs1: obs1, self.internal_state1: is1})

        # Predict future Z distribution by passing next states and actions through value target network,
        # also get target network's Z-atom values
        target_z_dist, target_z_atoms = self.sess.run(
            [self.target_critic.output_probs, self.target_critic.z_atoms],
            {self.obs1: obs1, self.internal_state1: is1, self.actions: actions1})
        # Create batch of target network's Z-atoms
        target_z_atoms = np.repeat(np.expand_dims(target_z_atoms, axis=0), self.args.ddpg_batch_size, axis=0)
        # Value of terminal states is 0 by definition
        target_z_atoms[mask_terminals, :] = 0.0
        # Apply Bellman update to each atom
        target_z_atoms = np.repeat(rewards, target_z_atoms.shape[1], axis=1) + target_z_atoms * self.gamma
        # Train critic
        _, critic_total_loss, _ = self.sess.run(
            [self.critic_loss, self.critic_total_loss, self.critic_op],
            {
                self.obs0: obs0, self.internal_state0: is0, self.actions: actions0,
                self.target_Z_ph: target_z_dist, self.target_atoms_ph: target_z_atoms
            }
        )
        # Use critic TD errors to update sample priorities

        actor_actions = self.sess.run(
            self.actor.output,
            {self.obs0: obs0, self.internal_state0: is0}
        )
        # Compute gradients of critic's value output distribution wrt actions
        action_grads = self.sess.run(
            self.critic.action_grads,
            {self.obs0: obs0, self.internal_state0: is0, self.actions: actor_actions}
        )
        # Train actor
        self.sess.run(
            self.actor_op,
            {self.obs0: obs0, self.internal_state0: is0, self.action_grads_ph: action_grads[0]}
        )

        return critic_total_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()

    def update_weight(self, weight):
        if len(self.actor.vars) != len(weight):
            NameError('g_var != weight')

        assign_op = []
        for var, target_var in zip(self.actor.vars, weight):
            target_var = tf.convert_to_tensor(target_var)
            assign_op.append(tf.assign(var, target_var))

        self.sess.run(assign_op)
