
from deep_adversarial_network.utils.common_util import *
from deep_adversarial_network.logging.tf_logger import Logger
from deep_adversarial_network.logging.logger import rootLogger
from IPython import display
import os, itertools, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from deep_adversarial_network.metrics.metric_eval import (calc_mse_psnr)
from deep_adversarial_network.metrics.metric_eval import (d_accuracy)
from torchvision.utils import make_grid
import torch

class DeepGAN(object):

    def __init__(self, discriminator, generator, model_name, dataset, batch_size,
                 d_optim=None, g_optim=None, d_lr=1e-4, g_lr=1e-4, mplib=False, epochs=10,tf_log_path=None):
        """
        Initialize all the parameters
        :param discriminator: discriminator model
        :param generator: generator model
        :param model_name: model name
        :param dataset: dataset
        :param batch_size: batch size
        :param d_optim: Optimizer for discriminator
        :param g_optim: Optimizer for generator
        :param d_lr: Learning Rate for discriminator
        :param g_lr: Learning Rate for generator
        :param mplib: Plotting for Matplotlib
        :param epochs: Epochs
        :param tf_log_path: Folder for Tensorflow
        """
        self.d_optim = d_optim or tf.train.AdamOptimizer
        self.g_optim = g_optim or tf.train.AdamOptimizer
        self.discriminator = discriminator
        self.generator = generator
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.epochs = epochs
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.mplib = mplib
        self.sess = None
        # Tensorboard Logging
        self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def adversarial_train(self, train_loader,test_loader, model_path):
        """
        Function for adversarial training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        :return:
        """

        # Name to store the GAN model
        gan_model_name = model_path+self.model_name+"_model_ckpt/"+self.model_name+".ckpt"

        # variables : input
        comp_img = tf.placeholder(tf.float32, shape=(None, 300,400,3))
        gt_img = tf.placeholder(tf.float32, shape=(None, 300,400,3))
        #z = tf.placeholder(tf.float32, shape=(None, 32,32,3))
        isTrain = tf.placeholder(dtype=tf.bool)

        # networks : generator
        G_z = self.generator.make_generator_network(comp_img, reuse=False, isTrain=isTrain)

        # networks : discriminator
        D_real, D_real_logits = self.discriminator.make_discriminator_network(gt_img,isTrain=isTrain)
        D_fake, D_fake_logits = self.discriminator.make_discriminator_network(G_z, reuse=True, isTrain=isTrain)

        # loss for each network
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        D_loss = D_loss_real + D_loss_fake
        #D_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))
        #G_loss = -tf.reduce_mean(tf.log(D_fake))
        G_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(gt_img, G_z))
        G_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        G_loss =  G_loss2 + 0.1 * G_loss1

        # trainable variables for each network
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        G_vars = [var for var in T_vars if var.name.startswith('generator')]

        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            D_optim = self.d_optim(self.d_lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
            G_optim = self.g_optim(self.g_lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

        # open session and initialize all variables
        # Try to restore model weights from previously saved model
        self.sess = tf.InteractiveSession()
        try:
            # 'Saver' op to save and restore all the variables
            rootLogger.info("Loading Saved Model")
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            saver.restore(self.sess, gan_model_name)
            rootLogger.info("Saved Model successfully loaded")
        except:
            tf.global_variables_initializer().run()
            rootLogger.info("Model not found, Created a new one")

        # results save folder
        model = model_path+self.model_name
        if self.mplib:
            if not os.path.isdir(model):
                os.mkdir(model)
            if not os.path.isdir(model + '_fixed_results'):
                os.mkdir(model + '_fixed_results')

        # Make directory for Saving Models
        if not os.path.isdir(model_path+self.model_name+"_model_ckpt"):
            os.mkdir(model_path+self.model_name+"_model_ckpt")

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
            for epoch_iter, (comp_image, gt_image) in enumerate(train_loader):

                # update discriminator
                loss_d_, _ = self.sess.run([D_loss, D_optim], {comp_img: comp_image, gt_img: gt_image, isTrain: True})
                D_losses.append(loss_d_)

                # update generator
                loss_g_, _ = self.sess.run([G_loss, G_optim], {comp_img: comp_image, gt_img: gt_image, isTrain: True})
                G_losses.append(loss_g_)

            # Log the training losses
            self.logger.log(d_error=np.mean(D_losses), g_error=np.mean(G_losses), epoch=epoch+1, n_batch=0,
                            num_batches=1)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' %
                  ((epoch+1), self.epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            fixed_p = model_path + 'Fixed_results/' + model + str(epoch + 1) + '.png'

            # Evaluate the model after every epoch
            self.evaluate_test_data(num_epoch=(epoch+1), test_loader=test_loader, G_z=G_z, D_fake =D_fake,
                                    D_fake_logits = D_fake_logits, D_real = D_real, D_real_logits = D_real_logits,
                                    D_loss = D_loss, G_loss = G_loss, comp_img=comp_img, gt_img=gt_img,
                                    isTrain=isTrain, show=False, save=True, path=fixed_p)

            # Save the model after every 10 epochs
            if (epoch+1)%10 == 0:
                # 'Saver' op to save and restore all the variables
                saver = tf.train.Saver()
                # Save model weights to disk
                save_path = saver.save(self.sess, gan_model_name)
                rootLogger.info("Model saved in file: %s" % save_path)

            if self.mplib:
                train_hist['D_losses'].append(np.mean(D_losses))
                train_hist['G_losses'].append(np.mean(G_losses))
                train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)
        rootLogger.info('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), self.epochs, total_ptime))
        rootLogger.info("Training finish!!!...")

        if self.mplib:
            with open(model + 'train_hist.pkl', 'wb') as f:
                pickle.dump(train_hist, f)
            rootLogger.info("Training history Saved")
            self.show_train_hist(train_hist, save=True, path= model + 'train_hist.png')

        self.sess.close()

    def show_train_hist(self,hist, show=False, save=False, path='Train_hist.png'):
        """
        Matplotlib plotting
        :param hist:
        :param show:
        :param save:
        :param path:
        :return:
        """
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

    def evaluate_test_data(self,test_loader, num_epoch, G_z, D_fake, D_fake_logits, D_real, D_real_logits,
                           D_loss, G_loss, comp_img,gt_img,isTrain, show=False, save=False,
                           path='result',tf_log_path=None):
        """
        Function to evaluate the result on test data
        :param test_loader: Loader for test data
        :param num_epoch: Epoch Number
        :param G_z: Random Gaussian Noise vector
        :param comp_img: Composite Image Hol
        :param gt_img:
        :param isTrain:
        :param show:
        :param save:
        :param path:
        :param tf_log_path:
        :return:
        """
        mse_avg_total = 0.0
        psnr_avg_total = 0.0
        Disc_accuracy_total = 0.0
        val_G_losses = []
        val_D_losses = []

        num_iter = len(test_loader)
        for iter,(comp_image, gt_image) in enumerate(test_loader):
            test_images = self.sess.run(G_z, {comp_img: comp_image, gt_img: gt_image, isTrain: False})

            mse_avg_iter, psnr_avg_iter = calc_mse_psnr(test_images, gt_image)

            mse_avg_total += mse_avg_iter
            psnr_avg_total += psnr_avg_iter


            D_real_prob, _ = self.sess.run([D_real, D_real_logits], {gt_img: gt_image, isTrain: False})
            D_fake_prob, _ = self.sess.run([D_fake, D_fake_logits], {G_z: comp_image, isTrain: False})

            Disc_accuracy = d_accuracy(D_real_prob, D_fake_prob)
            Disc_accuracy_total += Disc_accuracy

            val_loss_d_, _ = self.sess.run(D_loss, {comp_img: comp_image, gt_img: gt_image, isTrain: True})
            val_D_losses.append(val_loss_d_)

            # update generator
            val_loss_g_, _ = self.sess.run(G_loss, {comp_img: comp_image, gt_img: gt_image, isTrain: True})
            val_G_losses.append(val_loss_g_)

            self.logger.log_images(mode='generated',images=test_images, num_images=len(test_images), epoch=num_epoch, n_batch=iter,
                                   num_batches=len(test_loader), normalize=True)

            self.logger.log_images(mode='ground_truth', images=np.array(gt_image), num_images=len(gt_image), epoch=num_epoch, n_batch=iter,
                              num_batches=len(test_loader), normalize=True)

        mse_avg_total /= num_iter
        psnr_avg_total /= num_iter
        Disc_accuracy_total /= num_iter

        rootLogger.info("Epoch %d  MSE = [%.4f]    PSNR = [%.4f]"%(num_epoch,mse_avg_total,psnr_avg_total))
        self.logger.log_scores(mse=mse_avg_total, psnr=psnr_avg_total, epoch=num_epoch)

        rootLogger.info("Epoch %d  loss_d= [%.3f], loss_g= [%.3f]"%(num_epoch, np.mean(val_D_losses), np.mean(val_G_losses)))
        self.logger.log(d_error=np.mean(val_D_losses), g_error=np.mean(val_G_losses), epoch=epoch + 1, n_batch=0,
                        num_batches=1)

        rootLogger.info("Epoch %d  Disc_Acc = [%.4f] "%(num_epoch, Disc_accuracy_total))
        self.logger.log_scores(mse=mse_avg_total, psnr=psnr_avg_total, epoch=num_epoch)
        #self.logger.log_scores(disc_acc=Disc_accuracy_total,epoch=num_epoch)


    # def get_noise(batch_size, n_noise):
    #     return np.random.normal(size=(batch_size, n_noise))
