# Dependency imports
import numpy as np

import tensorflow as tf
import yaml
import os, sys
from utils.utils import ImgNormalize

slim = tf.contrib.slim

class GAN(object):
    def __init__(self, sess, loc):
        # Create local graph and use it in the session

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            ckpt_info_file = loc

            ckpt = tf.train.get_checkpoint_state(ckpt_info_file)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

                most_recent_ckpt = "%s" % ckpt_name + '.meta'
                most_recent_ckpt = os.path.join(ckpt_info_file, most_recent_ckpt)
                self.saver = tf.train.import_meta_graph(most_recent_ckpt, clear_devices=True)

                self.saver.restore(self.sess, os.path.join(ckpt_info_file, ckpt_name))
                # Get activation function from saved collection
                # You may need to change this in case you name it differently
                self.image = tf.get_collection('simul_image')[0]
                self.fakereal_image = tf.get_collection('fake_real_image')[0]
#                self.training_phase = tf.get_collection('training_phase')[0]

    def fakereal_img(self, image):
        image = ImgNormalize(image, 255) #.astype(np.float32)
        image = np.reshape(image, (1, 360, 360, 3))
        return self.sess.run(self.fakereal_image, feed_dict={self.image: image})

