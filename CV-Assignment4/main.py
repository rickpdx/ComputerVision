import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import plot_model

# Creates generator for use with the simple GAN


def create_generator():
    generator = models.Sequential()
    generator.add(layers.Dense(512, input_shape=[100]))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Dense(256))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Dense(128))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Dense(784))
    generator.add(layers.Reshape([28, 28, 1]))
    return generator

# Creates generator for use with the CNN-GAN


def create_conv_generator():
    generator = models.Sequential()
    generator.add(layers.Dense(7 * 7 * 128, input_shape=[100]))
    generator.add(layers.Reshape([7, 7, 128]))
    generator.add(layers.BatchNormalization(momentum=0.3))
    generator.add(layers.Conv2DTranspose(64, kernel_size=5,
                                         strides=2, padding='same'))
    generator.add(layers.BatchNormalization(momentum=0.3))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv2DTranspose(1, kernel_size=5,
                                         strides=2, padding='same'))
    generator.add(layers.Activation('tanh'))
    return generator

# Creates discriminator for use with the simple GAN


def create_discriminator():
    discriminator = models.Sequential()
    discriminator.add(layers.Dense(1, input_shape=[28, 28, 1]))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(256))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Dense(128))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Dense(64))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    return discriminator

# Creates discriminator for use in the CNN-GAN


def create_conv_discriminator():
    discriminator = models.Sequential()
    discriminator.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same',
                                    input_shape=[28, 28, 1]))
    discriminator.add(layers.LeakyReLU(0.3))
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Conv2D(128, kernel_size=5, strides=2,
                                    padding='same'))
    discriminator.add(layers.LeakyReLU(0.3))
    discriminator.add(layers.Dropout(0.5))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    return discriminator

# Create and compile the GAN


def create_gan(generator, discriminator):
    gan = models.Sequential([generator, discriminator])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.trainable = False
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# Generates random noise vectors sampled from a standard MVN distribution
# Uses generator and noise to great false images
# Returns false images, mean, and cov


def generate_fake_images(X, generator, batch_size):
    mean_x = np.zeros(100,)
    cov_x = np.eye(100, 100)
    noise = np.random.multivariate_normal(mean_x, cov_x, size=(100))
    gen_image = generator.predict_on_batch(noise)
    return gen_image, mean_x, cov_x

# Combines real and fake images and then permutates them and their labels
# Returns the full set and labels


def permutate_sets(X, gen_image, batch_size):
    perm = np.random.permutation(X.shape[0]+gen_image.shape[0])
    r_label = np.ones(shape=(batch_size, 1))
    f_label = np.zeros(shape=(batch_size, 1))
    labels = np.concatenate((r_label, f_label), axis=0)
    full_set = np.concatenate((X, gen_image), axis=0)
    label_list = []
    X_perm = []
    for p in perm:
        label_list.append(labels[p])
        X_perm.append(full_set[p])
    labels = np.array(label_list)
    full_set = np.array(X_perm)
    return full_set, labels

# Generates 5 fake images using the generator


def plot_gen_image(X, e, n, generator, name=''):
    X = X.reshape(100, -1)
    mean_x = np.zeros(100,)
    cov_x = np.eye(100, 100)
    noise = np.random.multivariate_normal(
        mean_x, cov_x, n)
    sample_img = generator.predict(noise)
    plt.figure()
    ax1 = plt.subplot(1, 5, 1)
    ax2 = plt.subplot(1, 5, 2)
    ax3 = plt.subplot(1, 5, 3)
    ax4 = plt.subplot(1, 5, 4)
    ax5 = plt.subplot(1, 5, 5)
    ax1.imshow(sample_img[0:1].reshape(28, 28), cmap='gray')
    ax2.imshow(sample_img[1:2].reshape(28, 28), cmap='gray')
    ax3.imshow(sample_img[2:3].reshape(28, 28), cmap='gray')
    ax4.imshow(sample_img[3:4].reshape(28, 28), cmap='gray')
    ax5.imshow(sample_img[4:5].reshape(28, 28), cmap='gray')
    plt.tight_layout()
    fig_str = name + 'epoch_' + str(e) + '.jpg'
    res_dir = 'figures/'
    plt.savefig(res_dir + fig_str)
    plt.close()
    plt.clf()


# Convert data to numpy arrays and normalize data
def preprocess(src):
    src = src.reshape(-1, 28, 28, 1)
    src = src/255
    return (src*2 - 1)


def main():
    # Settings for threading
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(6)

    # Datasets
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    # Preprocess datasets
    x_train = preprocess(x_train)

    res_dir = 'results/'

    # GAN
    print('------GAN------')
    generator = create_generator()
    generator.summary()

    discriminator = create_discriminator()
    discriminator.summary()

    gan = create_gan(generator, discriminator)
    gan.summary()

    # CNN GAN
    print('------CNN GAN------')
    conv_generator = create_conv_generator()
    conv_generator.summary()

    conv_discriminator = create_conv_discriminator()
    conv_discriminator.summary()

    cnn_gan = create_gan(conv_generator, conv_discriminator)
    cnn_gan.summary()

    batch_size = 200
    epochs = 100

    # Loop for simple GAN training
    gan_loss_list = []
    disc_loss_list = []
    for e in range(1, epochs + 1):
        print('---EPOCH #', e, '---')
        for b in range(len(x_train)//batch_size):
            X = x_train[b * batch_size: (b+1) * batch_size]
            gen_input = X.reshape(100, -1)

            # Get fake images from generator
            gen_image, mean_x, cov_x = generate_fake_images(
                gen_input, generator, batch_size)

            # Concatenate real and fake images, then permutate the indices
            full_set, labels = permutate_sets(X, gen_image, batch_size)

            # Train the discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(
                full_set, labels, reset_metrics=False)

            # Train the generator
            discriminator.trainable = False
            noise = np.random.multivariate_normal(mean_x, cov_x, batch_size)
            r_label = np.ones(shape=(batch_size, 1))
            gan_loss = gan.train_on_batch(noise, r_label, reset_metrics=False)

        # Generate figures for the report
        gan_loss_list.append(gan_loss)
        disc_loss_list.append(d_loss)
        print('Discriminator Loss:', d_loss)
        print('GAN Loss:', gan_loss)
        if e % 10 == 0 or e == 1 or e == epochs:
            samples = 5
            plot_gen_image(x_train, e, samples, generator, name='reg_')

    # Plot Losses and save them to a CVS
    gan_col = np.reshape(gan_loss_list, (-1, 1))
    dan_col = np.reshape(disc_loss_list, (-1, 1))
    losses = np.concatenate((gan_col, dan_col), axis=1)
    plt.figure()
    plt.plot(gan_loss_list)
    plt.plot(disc_loss_list)
    plt.ylabel('Loss Value')
    plt.xlabel('Epoch')
    plt.title('Discriminator and GAN Loss')
    plt.savefig(res_dir + 'plot.jpg')
    plt.close()
    np.savetxt('losses.csv', losses, delimiter=",")

    # Loop for CNN-GAN training
    gan_loss_list = []
    disc_loss_list = []
    for e in range(1, epochs+1):
        print('---EPOCH #', e, '---')
        for b in range(len(x_train)//batch_size):
            X = x_train[b * batch_size: (b+1) * batch_size]
            gen_input = X.reshape(100, -1)

            # Get fake images from generator
            gen_image, mean_x, cov_x = generate_fake_images(
                gen_input, conv_generator, batch_size)

            # Concatenate real and fake images, then permutate the indices
            full_set, labels = permutate_sets(X, gen_image, batch_size)

            # Train the discriminator
            conv_discriminator.trainable = True
            d_loss = conv_discriminator.train_on_batch(
                full_set, labels)

            # Train the generator
            conv_discriminator.trainable = False
            noise = np.random.multivariate_normal(mean_x, cov_x, batch_size)
            r_label = np.ones(shape=(batch_size, 1))
            gan_loss = cnn_gan.train_on_batch(noise, r_label)

        # Generate figures for the report
        gan_loss_list.append(gan_loss)
        disc_loss_list.append(d_loss)
        print('Discriminator Loss:', d_loss)
        print('GAN Loss:', gan_loss)
        if e % 10 == 0 or e == 1 or e == epochs:
            samples = 5
            plot_gen_image(x_train, e, samples, conv_generator, name='cnn_')

    # Plot Losses and save them to a CVS
    gan_col = np.reshape(gan_loss_list, (-1, 1))
    dan_col = np.reshape(disc_loss_list, (-1, 1))
    losses2 = np.concatenate((gan_col, dan_col), axis=1)
    plt.figure()
    plt.plot(gan_loss_list)
    plt.plot(disc_loss_list)
    plt.ylabel('Loss Value')
    plt.xlabel('Epoch')
    plt.title('Discriminator and CNN-GAN Loss')
    plt.savefig(res_dir+'cnn_plot.jpg')
    plt.close()
    np.savetxt('cnn_losses.csv', losses2, delimiter=",")


if __name__ == "__main__":
    main()
