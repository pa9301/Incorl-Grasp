# Dependency imports
import os

import numpy as np
import tensorflow as tf

from utils.utils import img_normalize

slim = tf.contrib.slim


class GAN(object):
    def __init__(self, sess, loc):
        # Create local graph and use it in the session

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            checkpoint_info_file = loc

            checkpoint = tf.train.get_checkpoint_state(checkpoint_info_file)
            if checkpoint and checkpoint.model_checkpoint_path:
                checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)

                most_recent_ckpt = "%s" % checkpoint_name + '.meta'
                most_recent_ckpt = os.path.join(checkpoint_info_file, most_recent_ckpt)
                self.saver = tf.train.import_meta_graph(most_recent_ckpt, clear_devices=True)

                self.saver.restore(self.sess, os.path.join(checkpoint_info_file, checkpoint_name))
                # Get activation function from saved collection
                # You may need to change this in case you name it differently
                self.image = tf.get_collection('simul_image')[0]
                self.fake_real_image = tf.get_collection('fake_real_image')[0]

    #                self.training_phase = tf.get_collection('training_phase')[0]

    def fake_real_img(self, image):
        image = img_normalize(image)
        image = np.reshape(image, (1, 360, 360, 3))
        return self.sess.run(self.fake_real_image, feed_dict={self.image: image})
