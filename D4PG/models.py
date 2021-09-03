import tensorflow as tf
import tensorflow.contrib as tc

import tensorflow.contrib.slim as slim
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from D4PG.l2_projection import _l2_project

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


def fully_connected_ln_relu(input, nhidden, scope):
    with tf.variable_scope(scope):

        hidden = tf.layers.dense(input, nhidden)

        hidden_ln = tc.layers.layer_norm(hidden, center=True, scale=True)
        hidden_ln_relu = tf.nn.relu(hidden_ln)

    return hidden_ln_relu


class Actor(Model):
    def __init__(self, obs, internal_state, nb_actions, hidden_sizes=(64,64), activation=tf.nn.relu, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        with tf.variable_scope(self.name) as scope:

            x1 = fully_connected_ln_relu(obs, 128, 'fc1')
            i_s = fully_connected_ln_relu(internal_state, 64, 'is_fc1')
            x = tf.concat([x1, i_s], axis=1)
            x2 = fully_connected_ln_relu(x, 128, 'fc2')
            x3 = fully_connected_ln_relu(x2, 64, 'fc3')
            x4 = tf.layers.dense(x3, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            continous_action = x4[:, 0:4]
            continous_action = tf.nn.tanh(continous_action)
            discrete_action = x4[:, 4:6]
            discrete_action = tf.nn.sigmoid(discrete_action)
            self.output = tf.concat([continous_action, discrete_action], axis=1)


class Critic(Model):
    def __init__(self, obs, internal_state, action, hidden_sizes=(64,64), activation=tf.nn.relu, action_merge_layer=1, name='critic', v_min=0.05, v_max=1.0, num_atoms=51, layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.action_merge_layer = action_merge_layer
        self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        with tf.variable_scope(self.name) as scope:

            x1 = fully_connected_ln_relu(obs, 128, 'fc1')
            a_s = tf.concat([action, internal_state], axis=1)
            a_s = fully_connected_ln_relu(a_s, 128, 'a_s_fc1')
            a_s = fully_connected_ln_relu(a_s, 64, 'a_s_fc2')
            x = tf.concat([x1, a_s], axis=1)
            x2 = fully_connected_ln_relu(x, 128, 'fc2')
            x3 = fully_connected_ln_relu(x2, 64, 'fc3')

            self.output_logits = tf.layers.dense(x3, self.num_atoms,
                                       kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003),
                                       name='output_logits')

            self.output_probs = tf.nn.softmax(self.output_logits, name='output_probs')
            self.z_atoms = tf.lin_space(self.v_min, self.v_max, self.num_atoms)
            self.Q_val = tf.reduce_sum(self.z_atoms * self.output_probs)  # the Q value is the mean of the categorical output Z-distribution

            self.action_grads = tf.gradients(self.output_probs, action, self.z_atoms)
            a = 0

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
