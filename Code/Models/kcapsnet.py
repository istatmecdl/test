
# coding: utf-8

# In[ ]:

"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Author: Matteo Alberti
    
"""

import numpy as np
import keras

keras.backend.set_image_data_format('channels_first')


from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from Layers.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import Model, Sequential




def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((train_set_x, train_set_y), (test_set_x, test_set_y))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (train_set_x, train_set_y), (test_set_x, test_set_y) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.95 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

def test(model, data):
    test_set_x, test_set_y = data
    y_pred, x_recon = model.predict([test_set_x, test_set_y], batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(test_set_y, 1))/test_set_y.shape[0])

    import matplotlib.pyplot as plt
    from Misc.utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([test_set_x[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()
    
def build_caps(width, height, depth, classes, summary, weightsPath=None):

        model = CapsNet(input_shape=(depth, height, width),
                        #n_class=len(np.unique(np.argmax(train_set_y, 1))),
                        n_class = classes,
						num_routing=3)

        if summary==True:
                model.summary()

        #model = to_multi_gpu(model, 2)

        #if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
                model.load_wights(weightsPath)
        
        return model
    
class CapsuleNet:

    @staticmethod
    def build_caps(width, height, depth, classes, summary, weightsPath=None):

        model = CapsNet(input_shape=(depth, height, width),
                        #n_class=len(np.unique(np.argmax(train_set_y, 1))),
                        n_class = classes,
						num_routing=3)

        if summary==True:
                model.summary()

        #model = to_multi_gpu(model, 2)

        #if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
                model.load_wights(weightsPath)
        
        return model
    
    def train(model, data, args):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param data: a tuple containing training and testing data, like `((train_set_x, train_set_y), (test_set_x, test_set_y))`
        :param args: arguments
        :return: The trained model
        """
        from keras import callbacks
        # unpacking the data
        (train_set_x, train_set_y), (test_set_x, test_set_y) = data
    
        # callbacks
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                   batch_size=args.batch_size, histogram_freq=args.debug)
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.95 ** epoch))
    
        # compile the model
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'out_caps': 'accuracy'})
        
    def test(model, data_test):
        test_set_x, test_set_y = data_test
        y_pred, x_recon = model.predict([test_set_x, test_set_y], batch_size=100)
        print('-'*50)
        print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(test_set_y, 1))/test_set_y.shape[0])
    
        import matplotlib.pyplot as plt
        from Misc.utils import combine_images
        from PIL import Image
    
        img = combine_images(np.concatenate([test_set_x[:50],x_recon[:50]]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
        print()
        print('Reconstructed images are saved to ./real_and_recon.png')
        print('-'*50)
        plt.imshow(plt.imread("real_and_recon.png", ))
        plt.show()

    """
    # Training without data augmentation:
    model.fit([train_set_x, train_set_y], [train_set_y, train_set_x], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[test_set_x, test_set_y], [test_set_y, test_set_x]], callbacks=[log, tb, checkpoint, lr_decay])
    """
'''
    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           horizontal_flip=True)  # shift up to 2 pixel for Cifar10
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(train_set_x, train_set_y, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(train_set_y.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[test_set_x, test_set_y], [test_set_y, test_set_x]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from Misc.utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model
'''




'''
def load_cifar10():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    (train_set_x, train_set_y), (test_set_x, test_set_y) = cifar10.load_data()

    train_set_x = train_set_x.reshape(-1, 32, 32, 3).astype('float32') / 255.
    test_set_x = test_set_x.reshape(-1, 32, 32, 3).astype('float32') / 255.
    train_set_y = to_categorical(train_set_y.astype('float32'))
    test_set_y = to_categorical(test_set_y.astype('float32'))
    return (train_set_x, train_set_y), (test_set_x, test_set_y)
'''
'''
if __name__ == "__main__":
    import numpy as np
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (train_set_x, train_set_y), (test_set_x, test_set_y) = load_cifar10()

    # define model
'''


'''

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((train_set_x, train_set_y), (test_set_x, test_set_y)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=model, data=(test_set_x, test_set_y))

'''