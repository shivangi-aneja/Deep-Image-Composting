
from deep_adversarial_network.utils.common_util import *
from deep_adversarial_network.logging.tf_logger import Logger
from IPython import display

class DeepGAN(object):

    def __init__(self, discriminator, generator, noise_size, model_name, dataset, batch_size,
                 optim=None, lr=1e-4, epochs=10):
        self.optim = optim or tf.train.AdamOptimizer
        self.discriminator = discriminator
        self.generator = generator
        self.lr = lr
        self.epochs = epochs
        self.noise_size = noise_size
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size

    def adversarial_train(self, X, Z, data_loader, tf_log_path):

        G_sample = self.generator.make_generator_network(z=Z)
        D_real = self.discriminator.make_discriminator_network(x=X)
        D_fake = self.discriminator.make_discriminator_network(G_sample)

        # Losses
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

        # Optimizers
        D_opt = self.optim(self.lr).minimize(D_loss, var_list=self.discriminator.discriminator_variables)
        G_opt = self.optim(self.lr).minimize(G_loss, var_list=self.generator.generator_variables)

        # TRAIN
        # Testing
        num_test_samples = 16
        test_noise = noise(num_test_samples, self.noise_size)
        num_batches = len(data_loader)

        # Start interactive session
        session = tf.InteractiveSession()
        # Init Variables
        tf.global_variables_initializer().run()
        # Init Logger
        logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

        #Train
        # Iterate through epochs
        for epoch in range(self.epochs):
            for n_batch, (batch, _) in enumerate(data_loader):

                # 1. Train Discriminator
                X_batch = images_to_vectors(batch.permute(0, 2, 3, 1).numpy())
                feed_dict = {X: X_batch, Z: noise(self.batch_size, self.noise_size)}
                _, d_error, d_pred_real, d_pred_fake = session.run(
                    [D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict
                )

                # 2. Train Generator
                feed_dict = {Z: noise(self.batch_size, self.noise_size)}
                _, g_error = session.run(
                    [G_opt, G_loss], feed_dict=feed_dict
                )

                if n_batch % 100 == 0:
                    display.clear_output(True)
                    # Generate images from test noise
                    test_images = session.run(
                        G_sample, feed_dict={Z: test_noise}
                    )
                    test_images = vectors_to_images(test_images)
                    # Log Images
                    logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches, format='NHWC')
                    # Log Status
                    logger.display_status(
                        epoch, self.epochs, n_batch, num_batches,
                        d_error, g_error, d_pred_real, d_pred_fake
                    )
