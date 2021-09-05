#!/usr/bin/env python
import os
import sys

import tensorflow as tf
import yaml


class SaverUtil(object):

    def __init__(self, sess, checkpoint_dir):
        self.sess = sess
        var_list = [v for v in tf.all_variables()]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1000)
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def guarantee_initialized_variables(self):

        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        tf.initialize_variables(uninitialized_vars)
        return uninitialized_vars

    def load_latest_checkpoint_or_init_if_none(self):
        """loads latest checkpoint from dir. if there are non run init variables."""
        # if no latest checkpoint init vars and return
        checkpoint_info_file = "%s/checkpoint" % self.checkpoint_dir
        checkpoint_file_path = "%s/" % self.checkpoint_dir
        if os.path.isfile(checkpoint_info_file):
            # load latest checkpoint
            info = yaml.load(open(checkpoint_info_file, "r"))
            assert 'model_checkpoint_path' in info
            most_recent_checkpoint = checkpoint_file_path + "%s" % (info['model_checkpoint_path'])
            sys.stderr.write("loading checkpoint %s\n" % most_recent_checkpoint)
            self.saver.restore(self.sess, most_recent_checkpoint)
        else:
            # no latest checkpoint, init and force a save now
            sys.stderr.write("no latest checkpoint in %s, just initing vars...\n" % self.checkpoint_dir)
            self.guarantee_initialized_variables()
            iteration = 0
            self.force_save(iteration)

    def force_save(self, num_transition):
        """force a save now."""
        new_checkpoint = "%s\\ckpt.%d" % (self.checkpoint_dir, num_transition)
        self.saver.save(self.sess, new_checkpoint)
