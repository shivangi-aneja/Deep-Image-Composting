
from deep_adversarial_network.utils.common_util import *
from deep_adversarial_network.logging.tf_logger import Logger
from deep_adversarial_network.logging.logger import rootLogger
from IPython import display
import os, itertools, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

class DeepGAN(object):

    def __init__(self, discriminator, generator, model_name, dataset, batch_size,
                 optim=None, lr=1e-4, mplib=False, epochs=10,tf_log_path=None):
        self.optim = optim or tf.train.AdamOptimizer
        self.discriminator = discriminator
        self.generator = generator
        self.lr = lr
        self.epochs = epochs
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.mplib = mplib
        self.sess = None
        # Tensorboard Logging
        self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def adversarial_train(self, data_loader,test_loader, model_path):

        # variables : input
        comp_img = tf.placeholder(tf.float32, shape=(None, 32,32,3))
        mask = tf.placeholder(tf.float32, shape=(None, 32,32,3))
        isTrain = tf.placeholder(dtype=tf.bool)

        # networks : generator
        G_z = self.generator.make_generator_network(mask, reuse=False, isTrain=isTrain)

        # networks : discriminator
        D_real, D_real_logits = self.discriminator.make_discriminator_network(comp_img,isTrain=isTrain)
        D_fake, D_fake_logits = self.discriminator.make_discriminator_network(G_z, reuse=True, isTrain=isTrain)

        # loss for each network
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        # trainable variables for each network
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        G_vars = [var for var in T_vars if var.name.startswith('generator')]

        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            D_optim = self.optim(self.lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
            G_optim = self.optim(self.lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

        # open session and initialize all variables
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # results save folder
        model = model_path+self.model_name
        if not os.path.isdir(model):
            os.mkdir(model)
        if not os.path.isdir(model + '_fixed_results'):
            os.mkdir(model + '_fixed_results')

        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')

        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            G_losses = []
            D_losses = []
            for epoch_iter, (comp_image, fg_img, alpha, bg_img) in enumerate(data_loader):

                loss_d_, _ = self.sess.run([D_loss, D_optim], {comp_img: comp_image, mask: alpha, isTrain: True})
                D_losses.append(loss_d_)

                # update generator
                loss_g_, _ = self.sess.run([G_loss, G_optim], {comp_img: comp_image, mask: alpha, isTrain: True})
                G_losses.append(loss_g_)

                # Log the training losses
                self.logger.log(d_error=loss_d_, g_error=loss_g_, epoch=epoch, n_batch=epoch_iter,
                                num_batches=self.batch_size)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' %
                  ((epoch + 1), self.epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            fixed_p = model_path + 'Fixed_results/' + model + str(epoch + 1) + '.png'

            # Evaluate the model after every epoch
            self.evaluate_test_data(num_epoch=(epoch + 1),test_loader=test_loader, G_z=G_z, comp_img=comp_img,
                                    mask=mask,isTrain=isTrain, show=False, save=True, path=fixed_p)


            if self.mplib:
                train_hist['D_losses'].append(np.mean(D_losses))
                train_hist['G_losses'].append(np.mean(G_losses))
                train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        rootLogger.info('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), self.epochs, total_ptime))
        rootLogger.info("Training finish!!!... Save Training Results")
        with open(model + 'train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)
        rootLogger.info("Model Saved")

        if self.mplib:
            self.show_train_hist(train_hist, save=True, path= model + 'train_hist.png')

        self.sess.close()

    def show_train_hist(self,hist, show=False, save=False, path='Train_hist.png'):
        x = range(len(hist['D_losses']))

        y1 = hist['D_losses']
        y2 = hist['G_losses']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

    def evaluate_test_data(self,test_loader, num_epoch,G_z,comp_img,mask,isTrain,show=False, save=False, path='result',tf_log_path=None):


        for iter,(comp_image, fg_img, alpha, bg_img) in enumerate(test_loader):
            test_images = self.sess.run(G_z, {comp_img: comp_image, mask: alpha, isTrain: False})

            self.logger.log_images(images=test_images, num_images=len(test_images), epoch=num_epoch, n_batch=iter,
                              num_batches=self.batch_size, normalize=True)
