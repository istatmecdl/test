'''
Created on 17/07/2018
Modified on 17/07/2018

@author: Francesco Pugliese, Eleonora Bernasconi
'''

# Other imports
import numpy as np
import os
from Preprocessing.preprocessing import load_kaggle_data, load_kaggle_segmented_data, load_trash_data, load_EuroSat, load_Corine
import sys 
import pdb

# Models imports
from Models.kcifar10net import Cifar10Net
from Models.kwideresnet import WideResNet
from Models.klenet import LeNet
from Models.kcapsnet import CapsuleNet
from Models.ksatellitenet import SatelliteNet

# Keras imports
from keras.datasets import cifar
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam

# TensorFlow imports
from tensorflow.examples.tutorials.mnist import input_data

# Define all the type of loss functions 
class DataPreparation(object):

    @staticmethod
    def get_training_algorithm(parameters):
        if parameters.training_algorithm.lower() == "adam":
            opt = 'adam'
            #opt = Adam(lr=parameters.learning_rate)                                                    # Adam Training Algorithm
        elif parameters.training_algorithm.lower() == "sgd_1":
            opt = SGD(lr=parameters.learning_rate)                                                     # Stochastic Gradient Descent Training Algorithm
        elif parameters.training_algorithm.lower() == "sgd_2":
            opt = SGD(lr=parameters.learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)       # Stochastic Gradient Descent Training Algorithm
        elif parameters.training_algorithm.lower() == "nadam":
            opt = Nadam(lr=parameters.learning_rate)
            
        return opt

    @staticmethod
    def topology_selector(parameters):
        if parameters.neural_model.lower() == "lenet":
            topology = LeNet
        elif parameters.neural_model.lower() == "cifar10net": 
            topology = Cifar10Net
        elif parameters.neural_model.lower() == "wideresnet": 
            topology = WideResNet
        elif parameters.neural_model.lower() == "capsnet":
            topology = CapsuleNet
        elif parameters.neural_model.lower() == "satellitenet":
            topology = SatelliteNet
        else:
            print("Model does not exist.")
            sys.exit()
        
        return topology

    @staticmethod
    def reshape_normalize(parameters, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y):

        # Pre-treat the datasets for Models
        if parameters.mnist_benchmark == True:
            train_set_y = np_utils.to_categorical(train_set_y, parameters.output_size)
            val_set_y = np_utils.to_categorical(val_set_y, parameters.output_size)
            test_set_y = np_utils.to_categorical(test_set_y, parameters.output_size)
            
            # Normalization
            train_set_x = train_set_x.astype('float32')
            val_set_x = val_set_x.astype('float32')
            test_set_x = test_set_x.astype('float32')
            train_set_x /= parameters.normalization_ratio_2
            val_set_x /= parameters.normalization_ratio_2
            test_set_x /= parameters.normalization_ratio_2
            
        elif parameters.cifar10_benchmark == True: 
            # Convert class vectors to binary class matrices.
            train_set_y = np_utils.to_categorical(train_set_y, parameters.output_size)
            val_set_y = np_utils.to_categorical(val_set_y, parameters.output_size)
            test_set_y = np_utils.to_categorical(test_set_y, parameters.output_size)

            # Normalization
            train_set_x = train_set_x.astype('float32')
            val_set_x = val_set_x.astype('float32')
            test_set_x = test_set_x.astype('float32')
            train_set_x /= parameters.normalization_ratio_2
            val_set_x /= parameters.normalization_ratio_2
            test_set_x /= parameters.normalization_ratio_2
        
        elif parameters.eurosat_data == True:
        
            # Convert class vectors to binary class matrices.
            train_set_y = np_utils.to_categorical(train_set_y, parameters.output_size)
            val_set_y = np_utils.to_categorical(val_set_y, parameters.output_size)
            test_set_y = np_utils.to_categorical(test_set_y, parameters.output_size)

            # Normalization
            train_set_x = train_set_x.astype('float32')
            val_set_x = val_set_x.astype('float32')
            test_set_x = test_set_x.astype('float32')
            
            train_set_x /= parameters.normalization_ratio_2
            val_set_x /= parameters.normalization_ratio_2
            test_set_x /= parameters.normalization_ratio_2
          
        elif parameters.trash_data == True:
            train_set_y = np_utils.to_categorical(train_set_y, num_classes=5)
            val_set_y = np_utils.to_categorical(val_set_y, parameters.output_size)
            test_set_y = np_utils.to_categorical(test_set_y, num_classes=5)
            
            # Normalization
            train_set_x = train_set_x.astype('float32')
            val_set_x = val_set_x.astype('float32')
            test_set_x = test_set_x.astype('float32')
            train_set_x /= parameters.normalization_ratio_2
            val_set_x /= parameters.normalization_ratio_2
            test_set_x /= parameters.normalization_ratio_2
                
        elif parameters.kaggle == True:
            val_set_x = []
            val_set_y = []
            # Reshape and add an axis to each data set in order to pass it to the conv layer. Transform labels into integer categorical arrays too.     
            if parameters.output_size > 1:
                train_set_x, train_set_y = [train_set_x.reshape((train_set_x.shape[0],parameters.input_size[0],parameters.input_size[1]))[:,np.newaxis,:,:], np_utils.to_categorical(train_set_y.astype("int"), parameters.output_size)]
                if parameters.mode == 0: 
                    test_set_x, test_set_y = [test_set_x.reshape((test_set_x.shape[0],parameters.input_size[0],parameters.input_size[1]))[:,np.newaxis,:,:], np_utils.to_categorical(test_set_y.astype("int"), parameters.output_size)]
            else: 
                train_set_x, train_set_y = [train_set_x.reshape((train_set_x.shape[0],parameters.input_size[0],parameters.input_size[1]))[:,np.newaxis,:,:], train_set_y.astype("int")]
                if parameters.mode == 0: 
                    test_set_x, test_set_y = [test_set_x.reshape((test_set_x.shape[0],parameters.input_size[0],parameters.input_size[1]))[:,np.newaxis,:,:], test_set_y.astype("int")]
    
        return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]
    
    @staticmethod
    def nodule_plane_selection(num_projections, mode):
        if num_projections == 0 and mode == 1:
            train_set_x = train_set_xy_x
            train_set_y = train_set_xy_y
            train_set_patients_ids = train_set_xy_patients_ids
        elif num_projections == 1 and mode == 1:
            train_set_x = train_set_xz_x
            train_set_y = train_set_xz_y
            train_set_patients_ids = train_set_xz_patients_ids
        elif num_projections == 2 and mode == 1:
            train_set_x = train_set_yz_x
            train_set_y = train_set_yz_y
            train_set_patients_ids = train_set_yz_patients_ids    
    
        return [train_set_x, train_set_y, train_set_patients_ids]
    
    @staticmethod
    def set_train_valid_test(data_type, parameters):
        val_set_x =  []
        val_set_y =  []
        
        if data_type == "mnist":                                                   # Mnist Dataset
            parameters.mode = 0
            print ('\nBenchmarking of the Model on MNIST Dataset...')
            mnist = input_data.read_data_sets(parameters.mnist_dataset_path, one_hot=False)
            
            train_set_x = np.zeros((50000, 1, 28, 28), dtype='uint8')
            train_set_y = np.zeros((50000,), dtype='uint8')
            
            train_set_x = np.vstack([img.reshape(-1,1,28,28) for img in mnist.train.images])
            train_set_y = mnist.train.labels

            test_set_x = np.vstack([img.reshape(-1,1,28,28) for img in mnist.test.images])
            test_set_y = mnist.test.labels
            
        elif data_type == "cifar10":                                               # Cifa10 Dataset
            # Modify original parameter for this dataset
            parameters.mode = 0
            
            print ('\nBenchmarking of the Model on Cifar10 Dataset...')

            train_set_x = np.zeros((50000, 3, 32, 32), dtype='uint8')
            train_set_y = np.zeros((50000,), dtype='uint8')

            # Merges all the data batches
            # Loads Cifar10 train set
            for i in range(1, 6):
                train_path = os.path.join(os.path.split(__file__)[0], parameters.cifar10_dataset_path, 'data_batch_' + str(i))
                data, labels = cifar.load_batch(train_path)
                train_set_x[(i - 1) * 10000: i * 10000, :, :, :] = data
                train_set_y[(i - 1) * 10000: i * 10000] = labels
                
            # Loads Cifar10 test set
            test_path = os.path.join(os.path.split(__file__)[0], parameters.cifar10_dataset_path, 'test_batch')
            test_set_x, test_set_y = cifar.load_batch(test_path)

            train_set_y = np.reshape(train_set_y, (len(train_set_y), 1))            # convert train row vector into a col vector
            test_set_y = np.reshape(test_set_y, (len(test_set_y), 1))               # convert test row vector into a col vector
            
        elif data_type == "trash":                                                  # on TrashNet data
            print ('\nLoading Trash Net Dataset...')
            (train_set_x, test_set_x, train_set_y, test_set_y)=load_trash_data()
            
            parameters.epochs_number = parameters.trash_epochs_number
            parameters.use_valid_set = parameters.trash_valid_set
            parameters.output_size = parameters.trash_output_size
            parameters.input_channels = parameters.trash_input_channels
            parameters.batch_size = parameters.trash_batch_size
            parameters.input_size = [parameters.trash_input_size, parameters.trash_input_size]
            train_set_x = train_set_x.reshape(-1, 3, 256, 256).astype('float32') / parameters.normalization_ratio_2
            test_set_x = test_set_x.reshape(-1, 3, 256, 256).astype('float32') / parameters.normalization_ratio_2
            parameters.mode = 0

        elif data_type == "eurosat":                                                # on EuroSAT data
            print ('\nLoading Eurosat Dataset...')
            parameters.output_size = parameters.eurosat_output_size
            parameters.input_channels = parameters.eurosat_input_channels
            parameters.input_size = [parameters.eurosat_input_size, parameters.eurosat_input_size]
            parameters.mode = 0
            #train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = load_EuroSat(validation_split = parameters.valid_set_perc, test_split = parameters.test_set_perc, shuffle = True, limit = parameters.limit, datapath = parameters.eurosat_dataset_path, input_size = parameters.eurosat_input_size, rescale = parameters.rescale, test_size = parameters.test_set_perc, parameters = parameters)
            train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = load_Corine(validation_split = parameters.valid_set_perc, test_split = parameters.test_set_perc, shuffle = True, limit = parameters.limit, datapath = parameters.eurosat_dataset_path, input_size = parameters.eurosat_input_size, rescale = parameters.rescale, test_size = parameters.test_set_perc, parameters = parameters)
            #pdb.set_trace()
            
        else:                                                                       # on kaggle data:
            # spread datasets onto all the input training sets and testing sets
            if parameters.mode == 0: 
                datasets = load_kaggle_data(datapath = parameters.kaggle_dataset_path, preprocimgspath = parameters.preprocessed_images_path, normalize_x = parameters.normalize_x, normalize_y = parameters.normalize_y, input_channels = parameters.input_channels, input_size = parameters.input_size, rescale = parameters.rescale, truePath = parameters.truePath, falsePath = parameters.falsePath, testPath = parameters.testPath)
                train_set_x, train_set_y, train_set_patients_ids = datasets[0]
                test_set_x, test_set_y, test_set_patients_ids = datasets[1]
            elif parameters.mode == 1: 
                datasets = load_kaggle_segmented_data(datapath = parameters.kaggle_dataset_path, preprocimgspath = parameters.preprocessed_images_path, normalize_x = parameters.normalize_x, normalize_y = parameters.normalize_y, input_channels = parameters.input_channels, input_size = parameters.input_size, rescale = parameters.rescale, truePath = parameters.truePath, falsePath = parameters.falsePath, testPath = parameters.testPath, max_patients_train = parameters.max_elements_train, max_nodules_train = parameters.max_segments_per_element_train, max_patients_test = parameters.max_segments_test, max_nodules_test = parameters.max_segments_per_element_test) 
                train_set_xy_x, train_set_xy_y, train_set_xy_patients_ids = datasets[0] 
                train_set_xz_x, train_set_xz_y, train_set_xz_patients_ids = datasets[1]
                train_set_yz_x, train_set_yz_y, train_set_yz_patients_ids = datasets[2]
                test_set_xy_x, test_set_xy_y, test_set_xy_patients_ids = datasets[3]
                test_set_xz_x, test_set_xz_y, test_set_xz_patients_ids = datasets[4]
                test_set_yz_x, test_set_yz_y, test_set_yz_patients_ids = datasets[5]

        if parameters.mode == 0:
            return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]
            
        elif parameters.mode == 1:     
            return [train_set_x, train_set_y, test_set_x, test_set_y, 
                train_set_xy_x, train_set_xy_y, train_set_xy_patients_ids, train_set_xz_x, train_set_xz_y, train_set_xz_patients_ids, 
                train_set_yz_x, train_set_yz_y, train_set_yz_patients_ids, test_set_xy_x, test_set_xy_y, test_set_xy_patients_ids, 
                test_set_xz_x, test_set_xz_y, test_set_xz_patients_ids, test_set_yz_x, test_set_yz_y, test_set_yz_patients_ids]