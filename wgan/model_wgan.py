import os
import sys
import time
import math
import cPickle
from scipy import misc
from glob import glob
#from matplotlib import pyplot as plt

from ops_new import *
from utils_new import *

class DataProvider(object):
    def __init__(self, config):
        if config.dataset == 'celebA':
            self.data = glob(os.path.join("./data/celebA/img_align_celeba/*.jpg"))
        elif config.dataset == 'lsun_64':
            with open('../lsun_64/bedroom_train_valid.lst', 'r') as lstfile:
                self.data = ['../lsun_64/bedroom_train/'+imgname for imgname in lstfile.read().split()]
        else:
            self.data = glob(os.path.join("./data/", config.dataset, "*.jpg"))
        self.len = len(self.data)
        print 'data len:', self.len
        self.batch_idxs = self.len // config.batch_size
        self.batch_idx = 0
        self.epoch_idx = 0
        
    def load_data(self, config, init = False):
        if init:
            np.random.shuffle(self.data)
            self.batch_idx = 0
            self.epoch_idx = 0
        batch_files = self.data[self.batch_idx*config.batch_size : (self.batch_idx+1)*config.batch_size]
        if config.color_space=="YUV":
            batch_images = np.array([(cvtRGB2YUV(misc.imread(batch_file))/127.5 - 1.) for batch_file in batch_files], dtype=np.float32)
        elif config.color_space=="RGB":
            batch_images = np.array([(misc.imread(batch_file)/127.5 - 1.) for batch_file in batch_files], dtype=np.float32)
        else:
            print "[!] Wrong color space!"
        self.batch_idx+=1
        if self.batch_idx>=self.batch_idxs:
            np.random.shuffle(self.data)
            self.batch_idx = 0
            self.epoch_idx+=1
        return batch_images

    def load_one_data(self, config, idx):
        if idx<0:
            idx = np.random.randint(0, self.len)
        one_data = misc.imread(self.data[idx])
        if config.color_space=="YUV":
            one_data = cvtRGB2YUV(one_data)
        return np.array(one_data, dtype = np.float32)/127.5 - 1.

class WGAN(object):
    def __init__(self, sess, config=None):
        self.sess = sess
        self.config = config
        self.build_model(config)

    def build_model(self, config):
        #data
        self.z = tf.placeholder(tf.float32, [config.batch_size, config.z_dim], name='z')
        self.images = tf.placeholder(tf.float32, [config.batch_size] + [config.image_size, config.image_size, 3], name='real_images')
        
        #generator
        if config.color_space=="YUV":
            self.images_Y, self.images_U, self.images_V = tf.split(self.images, 3, 3)
            self.generate_image_UV = self.generator_colorization(self.z, self.images_Y, config=config)
            self.generate_image = tf.concat([self.images_Y, self.generate_image_UV], 3)
        elif config.color_space=="RGB":
            self.generate_image = self.generator_colorization(self.z, config=config)

        #discriminator_wgan # on sigmoid = no prob
        self.logits_real = self.discriminator_wgan(self.images, config=config)
        self.logits_fake = self.discriminator_wgan(self.generate_image, reuse=True, config=config)

        #w-distance
        self.g_loss = -tf.reduce_mean(self.logits_fake)
        self.d_loss = -tf.reduce_mean(self.logits_real - self.logits_fake)

        #improved wgan
        if config.improved_wgan:
            alpha = tf.random_uniform(
                shape=[config.batch_size, 1],
                minval=0.,
                maxval=1.,
                dtype=tf.float32
                )
            differences = self.generate_image - self.images
            interpolates = self.images+(alpha*differences)
            gradients = tf.gradients(self.discriminator_wgan(interpolates, reuse=True, config=config), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            self.d_loss+=config.gradient_penalty_lambda*self.gradient_penalty

        self.total_loss = self.d_loss + self.g_loss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config=None):
        #def optimizer
        if config.optimizer=="Adam":
            d_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.adam_beta1).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.adam_beta1).minimize(self.g_loss, var_list=self.g_vars)
        elif config.optimizer=="RMSProp":
            d_optim = tf.train.RMSPropOptimizer(config.d_learning_rate, decay=0.9).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(config.g_learning_rate, decay=0.9).minimize(self.g_loss, var_list=self.g_vars)
        else:
            print "[!] Wrong optimizer!"
            return

        if not config.improved_wgan:
            #clip_d_vars_op = [tf.assign(var, tf.clip_by_value(var, -0.01, -0.01)) for var in self.d_vars]
            clip_ops = []
            for var in self.d_vars:
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, -config.clip_value, config.clip_value)))
            clip_d_vars_op = tf.group(*clip_ops)

        tf.global_variables_initializer().run()

        data = DataProvider(config)

        #prepare validation samples
        sample_images = data.load_data(config, init=True)
        sample_z = np.random.uniform(-1, 1, size=(config.sample_times, config.batch_size, config.z_dim))
        save_size = int(math.sqrt(config.batch_size))
        save_images(sample_images[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:05d}.png'.format(config.sample_dir, 0, 0), color_space=config.color_space)

        #load model
        if config.b_loadcheckpoint:
            if self.load(config):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                return

        counter = 1
        sample_interval = 200//(config.K_for_Dtrain+config.K_for_Gtrain)
        save_interval = 2000//(config.K_for_Dtrain+config.K_for_Gtrain)
        log_txt = open(config.result_dir+'checkpoint/'+config.dataset+'_'+config.dir_tag+'_log.txt', 'w')
        start_time = time.time()

        while data.epoch_idx<config.epoch:
            # Update D network for k_d times
            for k_d in xrange(0, config.K_for_Dtrain):
                batch_images = data.load_data(config)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, config.z_dim]).astype(np.float32)
                _, _g_loss, _d_loss, _loss = self.sess.run([d_optim, self.g_loss, self.d_loss, self.total_loss], 
                        feed_dict={self.z: batch_z, self.images: batch_images})
                #clip d_vars
                if not config.improved_wgan:
                    self.sess.run([clip_d_vars_op], feed_dict={})
                    '''
                    print "d_vars after clip:"
                    for var in self.d_vars:
                        print var.name
                        print var.eval()
                        '''
            # Update G network
            for k_g in xrange(0, config.K_for_Gtrain):
                batch_images = data.load_data(config)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, config.z_dim]).astype(np.float32)
                self.sess.run([g_optim], feed_dict={self.z: batch_z, self.images: batch_images})

            print("Epoch: [%2d] [%5d/%5d] time: %4.4f, total loss: %.8f, g_loss: %.8f,\n\t d_loss: %.8f" % (
                    data.epoch_idx, data.batch_idx, data.batch_idxs, time.time() - start_time, _loss, _g_loss, _d_loss))
            log_txt.write("{:d} {:d} {:d} {:.8f} {:.8f} {:.8f}\n".format(data.epoch_idx, data.batch_idx, data.batch_idxs, _loss, _g_loss, _d_loss))

            if (counter%sample_interval == 1):
                save_size = int(math.sqrt(config.batch_size))
                for sample_idx in range(config.sample_times):
                    _generate_image, _g_loss, _d_loss, _loss = self.sess.run([self.generate_image, self.g_loss, self.d_loss, self.total_loss], feed_dict={self.z: sample_z[sample_idx], self.images: sample_images})
                    save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/train_{:02d}_{:05d}_z{:d}.png'.format(config.sample_dir, data.epoch_idx, data.batch_idx, sample_idx+1), color_space=config.color_space)
                    print("[Sample] loss z%d: %.8f, g_loss: %.8f,\n\t d_loss: %.8f" % (sample_idx+1, _loss, _g_loss, _d_loss))
                    log_txt.write("0 0 -{:d} {:.8f} {:.8f} {:.8f}\n".format(sample_idx+1, _loss, _g_loss, _d_loss))

            if (counter%save_interval == 0):
                self.save(config, counter)

            log_txt.flush()
            counter += 1

        log_txt.close()

    def discriminator_wgan(self, image, y=None, reuse=False, config=None):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, 64, k_h=config.d_kernel_size, k_w=config.d_kernel_size, name='d_h0_conv'), name='d_bn0')
            h1 = lrelu(batch_norm(conv2d(h0, 128, k_h=config.d_kernel_size, k_w=config.d_kernel_size, name='d_h1_conv'), name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, 256, k_h=config.d_kernel_size, k_w=config.d_kernel_size, name='d_h2_conv'), name='d_bn2'))
            h3 = lrelu(batch_norm(conv2d(h2, 512, k_h=config.d_kernel_size, k_w=config.d_kernel_size, name='d_h3_conv'), name='d_bn3'))

            h4 = linear(tf.reshape(h3, [config.batch_size, -1]), 64, name = 'd_h4_lin')
            h5 = linear(h4, 1, 'd_h5_lin')
            #wgan without sigmoid
            return h5

    def generator_colorization(self, z, image_Y, config=None):
        with tf.variable_scope("generator") as scope:
            # project z
            h0 = linear(z, config.image_size * config.image_size, 'g_h0_lin', with_w=False)
            # reshape 
            h0 = tf.reshape(h0, [-1, config.image_size, config.image_size, 1])
            h0 = tf.nn.relu(batch_norm(h0, name = 'g_bn0'))
            # concat with Y
            h1 = tf.concat([image_Y, h0], 3)
            #print 'h0 shape after concat:', h0.get_shape()
            h1 = conv2d(h1, 128, k_h = 7, k_w = 7, d_h = 1, d_w = 1, name = 'g_h1_conv')
            h1 = tf.nn.relu(batch_norm(h1, name = 'g_bn1'))

            h2 = tf.concat([image_Y, h1], 3)
            h2 = conv2d(h2, 64, k_h = 5, k_w = 5, d_h = 1, d_w = 1, name = 'g_h2_conv')
            h2 = tf.nn.relu(batch_norm(h2, name = 'g_bn2'))
            
            h3 = tf.concat([image_Y, h2], 3)
            h3 = conv2d(h3, 64, k_h = 5, k_w = 5, d_h = 1, d_w = 1, name = 'g_h3_conv')
            h3 = tf.nn.relu(batch_norm(h3, name = 'g_bn3'))

            h4 = tf.concat([image_Y, h3], 3)
            h4 = conv2d(h4, 64, k_h = 5, k_w = 5,  d_h = 1, d_w = 1, name = 'g_h4_conv')
            h4 = tf.nn.relu(batch_norm(h4, name = 'g_bn4'))

            h5 = tf.concat([image_Y, h4], 3)
            h5 = conv2d(h5, 32, k_h = 5, k_w = 5,  d_h =1, d_w = 1, name = 'g_h5_conv')
            h5 = tf.nn.relu(batch_norm(h5, name = 'g_bn5'))
            
            h6 = tf.concat([image_Y, h5], 3)
            h6 = conv2d(h6, 2, k_h = 5, k_w = 5,  d_h = 1, d_w = 1, name = 'g_h6_conv')
            out = tf.nn.tanh(h6)

            print 'generator out shape:', out.get_shape()

            return out

    def save(self, config=None, step=0):
        model_name = "WGAN.model"
        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        this_checkpoint_dir = self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        print 'Saved checkpoint_dir:', this_checkpoint_dir

    def load(self, config=None):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (config.dataset, config.batch_size, config.image_size)
        checkpoint_dir = os.path.join(config.checkpoint_dir, model_dir)
        print 'checkpoint_dir:', checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  #get_checkpoint_state() returns CheckpointState Proto
        if ckpt and ckpt.model_checkpoint_path:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            print 'latest checkpoint:', latest_checkpoint
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            #ckpt_name = 'WGAN.model-77922'
            print 'loading checkpoint:', os.path.join(checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test_z(self, config=None):
        data = DataProvider(config)
        save_size = int(math.sqrt(config.batch_size))
        test_image_idx = config.test_image_idx
        if test_image_idx<0:
            test_image_idx = np.random.randint(0, data.len)
        print 'Test image idx:', test_image_idx
        save_dir = '{}/{:06d}'.format(config.sample_dir, test_image_idx)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        test_image = data.load_one_data(config, test_image_idx)
        save_image(test_image, '{}/test_{:06d}_origin.png'.format(save_dir, test_image_idx), color_space=config.color_space)
        test_image_batch = np.array([test_image for i in range(config.batch_size)])

    def test_fix(self, config=None):
        data = DataProvider(config)
        save_size = int(math.sqrt(config.batch_size))

        #get fixed image
        test_image_idxs = np.arange(config.batch_size)*1000+config.test_offset
        test_images = []
        for test_image_idx in test_image_idxs:
            test_image = data.load_one_data(config, test_image_idx)
            test_images.append(test_image)
        test_image_batch = np.array(test_images, dtype=np.float32)
        save_images(test_image_batch,[save_size, save_size] , '{}/test_fixed_origin_{:01d}.png'.format(config.sample_dir, config.test_offset), color_space=config.color_space)
        #scipy.misc.imsave('{}/test_fixed_origin_{:01d}.png'.format(config.sample_dir, config.test_offset), merge(test_image_batch[:save_size * save_size], [save_size, save_size]))

        #get fixed z
        with open("test_z_fixed.pkl",'r') as infile:
            test_z_batches = cPickle.load(infile)
        
        save_result_g_loss = []
        save_result_d_loss = []

        print "Testing fixed %d images..."%config.batch_size
        for test_round_idx in range(config.batch_size):
            print 'Round',test_round_idx, 
            _generate_image, _g_loss, _d_loss = self.sess.run([self.generate_image, self.g_loss, self.d_loss], feed_dict={self.z: test_z_batches[test_round_idx], self.images: test_image_batch})
            print "g_loss: %.8f, d_loss: %.8f" % (_g_loss, _d_loss)
            save_images(_generate_image[:save_size * save_size], [save_size, save_size], '{}/test_fixed_round_{:01d}{:02d}.png'.format(config.sample_dir, config.test_offset, test_round_idx), color_space=config.color_space)
            #scipy.misc.imsave('{}/test_fixed_round_{:01d}{:02d}.png'.format(config.sample_dir, config.test_offset, test_round_idx), merge(generate_image[:save_size * save_size], [save_size, save_size]))

            save_result_g_loss.append(_g_loss)
            save_result_d_loss.append(_d_loss)
        
        print 'Test done.'

        with open('{}/test_fixed_prob_{:01d}.pkl'.format(config.sample_dir, config.test_offset), 'w') as outfile:
            cPickle.dump((test_image_idxs, test_images, test_z_batches, save_result_g_loss, save_result_d_loss), outfile)
        print 'Save done.'