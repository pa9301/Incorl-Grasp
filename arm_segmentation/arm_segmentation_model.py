# Dependency imports
import numpy as np

import tensorflow as tf
import os

slim = tf.contrib.slim


class ArmSegmentation(object):

    def __init__(self, loc):
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
                self.seg_image = tf.get_collection('segmented_image')[0]

    def seg_img(self, image):
        image = np.reshape(image, (1, 256, 256, 3))
        return self.sess.run(self.seg_image, feed_dict={self.image: image})
