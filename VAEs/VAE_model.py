# Dependency imports
import os

import numpy as np
import tensorflow as tf

from utils.utils import img_normalize

slim = tf.contrib.slim


class VAE(object):

    def __init__(self, sess, loc):
        # Create local graph and use it in the session

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            checkpoint_info_file = loc

            checkpoint = tf.train.get_checkpoint_state(checkpoint_info_file)
            if checkpoint and checkpoint.model_checkpoint_path:
                checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)

                most_recent_checkpoint = "%s" % checkpoint_name + '.meta'
                most_recent_checkpoint = os.path.join(checkpoint_info_file, most_recent_checkpoint)
                self.saver = tf.train.import_meta_graph(most_recent_checkpoint, clear_devices=True)

                self.saver.restore(self.sess, os.path.join(checkpoint_info_file, checkpoint_name))
                # Get activation function from saved collection
                # You may need to change this in case you name it differently
                self.image = tf.get_collection('image')[0]
                self.latent_vec = tf.get_collection('latent_vec')[0]
                self.recon_image = tf.get_collection('recon_image')[0]
                self.training_phase = tf.get_collection('training_phase')[0]

    def latent(self, image):
        # The 'x' corresponds to name of input placeholder
        image = img_normalize(image)
        image = np.reshape(image, (1, 128, 128, 12))

        return self.sess.run(self.latent_vec, feed_dict={self.image: image, self.training_phase: False})

    def recon_img(self, image):
        image = img_normalize(image)
        image = np.reshape(image, (1, 128, 128, 12))
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.recon_image, feed_dict={self.image: image, self.training_phase: False})
