#!/usr/bin/env python
import os, yaml, sys
import tensorflow as tf

class SaverUtil(object):
  def __init__(self, sess, ckpt_dir):
    self.sess = sess
    var_list = [v for v in tf.all_variables()]
    self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1000)
    self.ckpt_dir = ckpt_dir
    self.is_restart_incorlgrasp = False
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)

    ckpt_info_file = "%s\\checkpoint" % self.ckpt_dir
    if os.path.isfile(ckpt_info_file):
      self.is_restart_incorlgrasp = True
    else:
      self.is_restart_incorlgrasp = False

  def guarantee_initialized_variables(self, list_of_variables=None):

    uninitialized_vars = []
    for var in tf.all_variables():
      try:
        self.sess.run(var)
      except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)

    tf.initialize_variables(uninitialized_vars)
    return uninitialized_vars

  def load_latest_ckpt_or_init_if_none(self):
    """loads latests ckpt from dir. if there are non run init variables."""
    var_list = [v for v in tf.all_variables()]
    # if no latest checkpoint init vars and return
    ckpt_info_file = "%s/checkpoint" % self.ckpt_dir
    ckpt_file_path = "%s/" % self.ckpt_dir
    if os.path.isfile(ckpt_info_file):
      # load latest ckpt
      info = yaml.load(open(ckpt_info_file, "r"))
      assert 'model_checkpoint_path' in info
      most_recent_ckpt = ckpt_file_path + "%s" % (info['model_checkpoint_path'])
      sys.stderr.write("loading ckpt %s\n" % most_recent_ckpt)
      self.saver.restore(self.sess, most_recent_ckpt)
    else:
      # no latest ckpts, init and force a save now
      sys.stderr.write("no latest ckpt in %s, just initing vars...\n" % self.ckpt_dir)
      #      tf.global_variables_initializer()
      self.guarantee_initialized_variables()
      iter = 0
      self.force_save(iter)
      self.is_restart_incorlgrasp = False


  def force_save(self, num_transition):
    """force a save now."""
    new_ckpt = "%s\\ckpt.%d" % (self.ckpt_dir, num_transition)
#    sys.stderr.write("saving ckpt %s\n" % new_ckpt)
#    file_name = os.path.join(self.ckpt_dir, 'GAN')
    self.saver.save(self.sess, new_ckpt)
