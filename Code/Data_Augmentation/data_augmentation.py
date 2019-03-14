from keras.preprocessing.image import ImageDataGenerator
import timeit
class Data_Augmentation:
    def data_augmentation():
        print ('\nReal Time Data Augmentation ON\n')
        
        load_start_time = timeit.default_timer()
        # This will do preprocessing and realtime data augmentation:
        '''
        datagen = ImageDataGenerator(zoom_range=0.2,
                                    #shear_range=0.1, #displaces each point in fixed direction
                                    horizontal_flip=True,  # randomly flip images
                                    vertical_flip=True,  # randomly flip images
                                    rotation_range=30)  # randomly rotate images in the range (degrees, 0 to 180)
                                    #width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                                    #height_shift_range=0.2)  # randomly shift images vertically (fraction of total height)

                                    #featurewise_center=False,  # set input mean to 0 over the dataset
                                    #samplewise_center=False,  # set each sample mean to 0
                                    #featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    #samplewise_std_normalization=False,  # divide each input by its std
                                    #zca_whitening=False,  # apply ZCA whitening
        
        print('zoom_range=0.2')
        #print('width_shift_range=0.2')
        #print('height_shift_range=0.2')
        print('horizontal_flip=True')
        print('vertical_flip=True')
        #print('shear_range=0.1')
        print('rotation_range=30')
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        
        print ('\nLoading time data augmentation: %.2f minutes\n' % ((timeit.default_timer() - load_start_time) / 60.))
        '''
        
        datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip = True, vertical_flip = True, rotation_range = 180)                                   

        return datagen      