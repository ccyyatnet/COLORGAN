import os
import numpy as np
import pprint

from model_wgan import WGAN

import tensorflow as tf

tf.app.flags.DEFINE_string("devices", "gpu:0", "Which gpu to be used")

##params for dataset and environment
tf.app.flags.DEFINE_string("dataset", "lsun_64", "The name of dataset [celebA, lsun_64]")
tf.app.flags.DEFINE_string("dir_tag", "wgan_RMS", "dir_tag for sample_dir and checkpoint_dir")
tf.app.flags.DEFINE_string("result_dir", "./result/", "Where to save the checkpoint and sample")
tf.app.flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

##training setting
tf.app.flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
tf.app.flags.DEFINE_integer("batch_size", 64, "The size of batch images")
tf.app.flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
tf.app.flags.DEFINE_boolean("b_loadcheckpoint", False, "b_loadcheckpoint")
tf.app.flags.DEFINE_integer("sample_times", 3, "sample_times")

tf.app.flags.DEFINE_boolean("improved_wgan", False, "improved wgan")
tf.app.flags.DEFINE_float("gradient_penalty_lambda", 10, "gradient penalty lambda")
tf.app.flags.DEFINE_float("clip_value", 0.01, "clip_value")
tf.app.flags.DEFINE_integer("K_for_Dtrain", 5, "K_for_Dtrain")
tf.app.flags.DEFINE_integer("K_for_Gtrain", 1, "K_for_Gtrain")
tf.app.flags.DEFINE_float("smooth", 1.0, "smooth")
tf.app.flags.DEFINE_integer("d_kernel_size", 5, "d_kernel_size")
tf.app.flags.DEFINE_float("l1_lambda",1, "l1_lambda")
tf.app.flags.DEFINE_boolean("multi_z", False, "True for concat multi layer noise in generator")
tf.app.flags.DEFINE_boolean("multi_condition", False, "True for concat multi layer condition in generator")

##test setting
tf.app.flags.DEFINE_integer("test_image_idx", -1, "test_image_idx")
tf.app.flags.DEFINE_boolean("test_random_z", True, "test random z")
tf.app.flags.DEFINE_integer("test_images", 64, "number_of_test_images")
tf.app.flags.DEFINE_integer("test_offset", 0, "test_offset(<1000)")

##params for preprocess and model setting
tf.app.flags.DEFINE_string("color_space", "YUV", "Color space [YUV,RGB]")
tf.app.flags.DEFINE_integer("image_size", 64, "The size of the output images to produce [64]")
tf.app.flags.DEFINE_integer("center_crop_size", 0, "The width of the images presented to the model, 0 for auto")
tf.app.flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
tf.app.flags.DEFINE_integer("z_dim", 100, "z_dim")

##params for optimizer
tf.app.flags.DEFINE_string("optimizer", "RMSProp", "optimizer [RMSProp, Adam]")
tf.app.flags.DEFINE_float("adam_beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate of for optimizer")
tf.app.flags.DEFINE_float("g_learning_rate", 0.0001, "Learning rate of for optimizer")


FLAGS = tf.app.flags.FLAGS

def main(_):

    pprint.PrettyPrinter().pprint(FLAGS.__flags)

    FLAGS.sample_dir = FLAGS.result_dir + 'samples/' + FLAGS.dataset + '_' + FLAGS.dir_tag
    FLAGS.checkpoint_dir = FLAGS.result_dir + 'checkpoint/' + FLAGS.dataset + '_' + FLAGS.dir_tag

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        with tf.device(FLAGS.devices):

            dcgan = WGAN(sess, config=FLAGS)

            if FLAGS.is_train:
                dcgan.train(FLAGS)
            else:
                if dcgan.load(FLAGS):
                    print " [*] Load SUCCESS"
                    if FLAGS.test_random_z:
                        print " [*] Test RANDOM Z"
                        dcgan.test_fix(FLAGS)
                    else:
                        print " [*] Test Z"
                        dcgan.test_z(FLAGS)
                else:
                    print " [!] Load failed..."

if __name__ == '__main__':
    tf.app.run()
