'''
Created on 21/03/2018
Modiefied on 17/07/2018

@author: Francesco Pugliese, Eleonora Bernasconi
'''

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# Keras imports
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Program imports
from Preprocessing.data_preparation import DataPreparation
from Settings.settings import SetParameters
from Initialization.initialization import Initialization
from Data_Augmentation.data_augmentation import Data_Augmentation
from Models.model_preparation import ModelPreparation
from View.view import Viewclass
from Misc.utils import set_cpu_or_gpu

# Other imports
import numpy as np
import os
import timeit
import platform

class Training():
    def func_train():
        # Operating System
        OS = platform.system()                                                               # returns 'Windows', 'Linux', etc

        default_config_file = "enginecv.ini"                                                 # Default Configuration File

        ## CONFIGURATION
        ## COMMAND LINE ARGUMENT PARSER

        parser = Initialization.parser()                                                     # Define parser

        (arg1) = parser.parse_args()
        config_file = arg1.conf
        if config_file is None: 
            config_file = default_config_file                                                # Default Configuration File

        ## CONFIGURATION FILE PARSER
        # Read the Configuration File
        set_parameters = SetParameters("../Conf", config_file, OS) 
        parameters = set_parameters.read_config_file()

        # Set the CPU or GPU
        set_cpu_or_gpu(parameters)

        ## INITIALIZATION    
        init = Initialization(parameters) 
        valid_set_split, test_set_split, global_start_time, deepnetworks, default_callbacks, parameters = init.read_initialization()

        ## DATA PREPARATION
        if parameters.mnist_benchmark == True:                                               # Mnist Benchmark data    
            data_type = "mnist"
        elif parameters.cifar10_benchmark == True:                                           # Cifar10 Benchmark data    
            data_type = "cifar10"
        elif parameters.trash_data == True:                                                  # TrashNet data
            data_type = "trash"
        elif parameters.eurosat_data == True:                                                # EuroSAT data
            data_type = "eurosat"
        elif parameters.corine_data == True:                                                 # EuroSAT data
            data_type = "corine"
        elif parameters.kaggle_data == True:                                                 # Kaggle Lung Cancer Detection data
            data_type = "kaggle"                                                            
            
        if parameters.mode == 0: 
            train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = DataPreparation.set_train_valid_test(data_type, parameters)
            
        '''
        elif parameters.mode == 1: 
            train_set_x, train_set_y, test_set_x, test_set_y, 
            train_set_xy_x, train_set_xy_y, train_set_xy_patients_ids, train_set_xz_x, train_set_xz_y, train_set_xz_patients_ids, train_set_yz_x, train_set_yz_y, train_set_yz_patients_ids, 
            test_set_xy_x, test_set_xy_y, test_set_xy_patients_ids, test_set_xz_x, test_set_xz_y, test_set_xz_patients_ids, test_set_yz_x, test_set_yz_y, test_set_yz_patients_ids = DataPreparation.set_train_valid_test(data_type, parameters)
        '''
        for p in range(parameters.num_projections): 

            if parameters.mode == 1: 
                train_set_x, train_set_y, train_set_patients_ids = DataPreparation.nodule_plane_selection(p, parameters.mode)                                       # Nodules plane selection  
            
            n_train_batches = len(train_set_x) // parameters.batch_size                                                                                             # compute number of minibatches for training, validation and testing
            n_test_batches = len(test_set_x) // parameters.batch_size                                                                                               # compute number of minibatches for training, validation and testing

            normal_train_set_y = train_set_y
            if parameters.mode == 0:
                normal_test_set_y = test_set_y

            Viewclass.print_sizes(parameters.mode, parameters.batch_size, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y)

            train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y  = DataPreparation.reshape_normalize(parameters, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y)     # Reshape and Normalize  

            if parameters.save_best_model == True:
                checkPoint=ModelCheckpoint(parameters.models_path+'/'+parameters.model_file, save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            
            mp = ModelPreparation()
            deepnetwork = mp.model_selector(parameters, False, True)                       # Builds models for training
            
            if parameters.save_best_model == True:
                default_callbacks = default_callbacks+[checkPoint]
           
            # COMMON CODE
            
            start_time = timeit.default_timer()

            print ('\nTraining of the neural network N.%i ...\n' % (p+1))
            
            earlyStopping=EarlyStopping(monitor='loss', patience=1, verbose=0, mode='min') 

            if parameters.early_stopping == True:
                default_callbacks = default_callbacks+[earlyStopping]
            
            if parameters.save_best_model == True:
                default_callbacks = default_callbacks+[checkPoint]

            pdb.set_trace()
            
            if parameters.dataAugmentation == False:
                # Training with NO Data Augmentation
                if parameters.neural_model == "capsnet": 
                    mem_train_set_x = train_set_x
                    mem_train_set_y = train_set_y
                    mem_test_set_x = test_set_x
                    mem_test_set_y = test_set_y
                    
                    train_set_x = [mem_train_set_x, mem_train_set_y]
                    train_set_y = [mem_train_set_y, mem_train_set_x]
                    test_set_x = [mem_test_set_x, mem_test_set_y]
                    test_set_y = [mem_test_set_y, mem_test_set_x]

                history = deepnetwork.fit(train_set_x, train_set_y, batch_size=parameters.batch_size, epochs=parameters.epochs_number, validation_data = (val_set_x, val_set_y), shuffle=True, callbacks=default_callbacks, verbose = 2)

            else:
                # Training with Data Augmentation
                datagen = Data_Augmentation.data_augmentation()
                
                if parameters.saveAugmentedData == True: 
                    if os.path.exists(parameters.preprocessed_images_path):    # Removes the folder to avoid overlapping
                        os.remove(parameters.preprocessed_images_path) 
                    os.makedirs(parameters.preprocessed_images_path)
                    history = deepnetwork.fit_generator(datagen.flow(train_set_x, train_set_y, batch_size=parameters.batch_size, save_to_dir=parameters.preprocessed_images_path, save_prefix='eurosat_augmented', save_format='jpg'), steps_per_epoch=train_set_x.shape[0]/parameters.batch_size, epochs=parameters.epochs_number, validation_data = (val_set_x, val_set_y), callbacks = default_callbacks, verbose = 2)
                else: 
                    history = deepnetwork.fit_generator(datagen.flow(train_set_x, train_set_y, batch_size=parameters.batch_size), steps_per_epoch=train_set_x.shape[0]/parameters.batch_size, epochs=parameters.epochs_number, validation_data = (val_set_x, val_set_y), callbacks = default_callbacks, verbose = 2)

            # Save the Model after the last epoch, not saving the best model 
            if parameters.save_best_model == False:                                    
                deepnetwork.save_weights(parameters.models_path+'/'+parameters.model_file)

        if parameters.neural_model != "capsnet":
                deepnetworks.append(deepnetwork)
                deepnetworks = np.asarray(deepnetworks)
                
        end_time = timeit.default_timer()
        print ('\nTraining time: %.2f minutes' % ((end_time - start_time) / 60.))

        if parameters.plot_history == True: 
            Viewclass.plot_hist(history = history, parameters = parameters)

        # Emits a sound warning the processing is finished
        # winsound.Beep(sound_freq, sound_dur)    
        return history
        
history = Training.func_train()
 