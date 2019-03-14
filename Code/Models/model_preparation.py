'''
Created on 27/07/2018
Modified on 27/07/2018

@author: Francesco Pugliese, Eleonora Bernasconi
'''

# Other imports
from Preprocessing.data_preparation import DataPreparation
import os
import sys

#import argparse
from Metrics.metrics import Metrics

# Define all the type of loss functions 
class ModelPreparation(object):

    def __init__(self):
        return None
    
    def model_selector(self, parameters, test, summary):
        if parameters.cifar10_benchmark == True:                                                # Cifar10 Model Building    
            deepnetwork = self.build_cifar10_deepnetwork(parameters, test, summary)                   
        elif parameters.eurosat_data == True:                                                   # EuroSat Model Building
            deepnetwork = self.build_eurosat_deepnetwork(parameters, test, summary)
        elif parameters.mnist_benchmark == True:                                                # Mnist Model Building    
            deepnetwork = self.build_mnist_deepnetwork(parameters, test, summary)    
        elif parameters.trash_data == True:                                                     # TrashNet Model Building    
            deepnetwork = self.build_trashnet_deepnetwork(parameters, test, summary)
        elif parameters.kaggle_data == True:                                                    # Kaggle Lung Cancer Model Building
            deepnetwork = self.build_kaggle_deepnetwork(parameters, test, summary)
        
        return deepnetwork
        
    def build_cifar10_deepnetwork(self, parameters, test, summary):
        summary = True
        if test == True: 
            summary = False
        opt = DataPreparation.get_training_algorithm(parameters)                                            # Training Algorithm Selection                                                                                       
        topology = DataPreparation.topology_selector(parameters)                                            # Topology Selection
        '''
        if parameters.neural_model.lower() == "capsnet":
            model=CapsuleNet.build_caps(parameters.cifar10_input_size, parameters.cifar10_input_size, parameters.cifar10_input_channels, parameters.cifar10_output_size, True, weightsPath=None)
            topology.train(model, data, args)
            
        else:
        '''
        
        deepnetwork = topology.build(width=parameters.input_size[0], height=parameters.input_size[1], depth=parameters.input_channels, classes=parameters.output_size, summary=summary)
        if test == True: 
            if os.path.isfile(os.path.join("./",parameters.models_path)+"/"+parameters.model_file):
                deepnetwork.load_weights(os.path.join("./",parameters.models_path)+"/"+parameters.model_file)   
            else:
                print('\nPre-trained model not found in : %s.' % (parameters.models_path))
                sys.exit("")
        deepnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return deepnetwork
    
    def build_mnist_deepnetwork(self, parameters, test, summary):
        summary = True
        if test == True: 
            summary = False
        opt = DataPreparation.get_training_algorithm(parameters)                                            # Training Algorithm Selection                                                                                
        topology = DataPreparation.topology_selector(parameters)                                            # Topology Selection
        '''
        if parameters.neural_model.lower() == "capsnet":
            from keras import callbacks
            model=topology.build_caps(parameters.mnist_input_size, parameters.mnist_input_size, parameters.mnist_input_channels, parameters.mnist_output_size, True, weightsPath=None)
            #CapsuleNet.train(model=model, data=((train_set_x, train_set_y), (test_set_x, test_set_y)), args=args)
            log = callbacks.CSVLogger(args.save_dir + '/log.csv')
            tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', batch_size=args.batch_size, histogram_freq=args.debug)
            checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', save_best_only=True, save_weights_only=True, verbose=1)
            lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.95 ** epoch))
            model.compile(optimizer=parameters.opt, loss=[margin_loss, 'mse'], loss_weights=[1., args.lam_recon], metrics={'out_caps': 'accuracy'})

        else:
        '''
        deepnetwork = topology.build(width=parameters.input_size[0], height=parameters.input_size[1], depth=parameters.input_channels, classes=parameters.output_size, summary=summary)    
        if test == True: 
            if os.path.isfile(os.path.join("./",parameters.models_path)+"/"+parameters.model_file):
                deepnetwork.load_weights(os.path.join("./",parameters.models_path)+"/"+parameters.model_file)   
            else:
                print('\nPre-trained model not found in : %s.' % (parameters.models_path))
                sys.exit("")
        deepnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
        return deepnetwork

    def build_eurosat_deepnetwork(self, parameters, test, summary):
        summary = True
        if test == True: 
            summary = False
        opt = DataPreparation.get_training_algorithm(parameters)                                            # Training Algorithm Selection                                                                                  

        topology = DataPreparation.topology_selector(parameters)                                            # Topology Selection
        '''
        if parameters.neural_model.lower() == "capsnet":
            model=topology.build_caps(parameters.cifar10_input_size, parameters.cifar10_input_size, parameters.cifar10_input_channels, parameters.cifar10_output_size, True, weightsPath=None)
            topology.train(model, data, args)
        else:
        '''
        deepnetwork = topology.build(width=parameters.input_size[0], height=parameters.input_size[1], depth=parameters.input_channels, classes=parameters.output_size, summary=summary)
        if test == True: 
            if os.path.isfile(os.path.join("./",parameters.models_path)+"/"+parameters.model_file):
                deepnetwork.load_weights(os.path.join("./",parameters.models_path)+"/"+parameters.model_file)   
            else:
                print('\nPre-trained model not found in : %s.' % (parameters.models_path))
                sys.exit("")
        deepnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return deepnetwork

    def build_trashnet_deepnetwork(self, parameters, test, summary):
        summary = True
        if test == True: 
            summary = False
        opt = DataPreparation.get_training_algorithm(parameters)                                            # Training Algorithm Selection                                                                                 
        topology = DataPreparation.topology_selector(parameters)                                            # Topology Selection
        '''  
        if parameters.neural_model.lower() == "capsnet":
            
            model=topology.build_caps(parameters.trash_input_size, parameters.trash_input_size, parameters.trash_input_channels, parameters.trash_output_size, True, weightsPath=None)
            #CapsuleNet.train(model=model, data=((train_set_x, train_set_y), (test_set_x, test_set_y)), args=args)
            model.compile(optimizer=opt, loss=[margin_loss, 'mse'], loss_weights=[1., args.lam_recon], metrics={'out_caps': 'accuracy'})
         
        else:
        '''
        deepnetwork = topology.build(width=parameters.input_size[0], height=parameters.input_size[1], depth=parameters.input_channels, classes=parameters.output_size, summary=summary)    
        if test == True: 
            if os.path.isfile(os.path.join("./",parameters.models_path)+"/"+parameters.model_file):
                deepnetwork.load_weights(os.path.join("./",parameters.models_path)+"/"+parameters.model_file)   
            else:
                print('\nPre-trained model not found in : %s.' % (parameters.models_path))
                sys.exit("")
        deepnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return deepnetwork
            
    def build_kaggle_deepnetwork(self, parameters, test, summary):
        summary = True
        if test == True: 
            summary = False
        opt = DataPreparation.get_training_algorithm(parameters)                                            # Training Algorithm Selection                                                                        
        topology = DataPreparation.topology_selector(parameters)                                            # Topology Selection

        parameters.model_summary = True                                                                     # Don't show summary for other nodule sections
        if parameters.num_projections > 0:
            parameters.model_summary = False
    
        deepnetwork = topology.build(width=parameters.input_size[0], height=parameters.input_size[1], depth=parameters.input_channels, classes=parameters.output_size, summary=summary)    
        if test == True: 
            if os.path.isfile(os.path.join("./",parameters.models_path)+"/"+parameters.model_file):
                deepnetwork.load_weights(os.path.join("./",parameters.models_path)+"/"+parameters.model_file)   
            else:
                print('\nPre-trained model not found in : %s.' % (parameters.models_path))
                sys.exit("")
        deepnetwork.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', Metrics.fmeasure])
        
        if parameters.num_projections == 0 and parameters.mode == 1:
            print ('\nSegmented Images Mode - Nodule Section on plane XY\n')
        elif parameters.num_projections == 1 and parameters.mode == 1:
            print ('\nSegmented Images Mode - Nodule Section on plane XZ\n')
        elif parameters.num_projections == 2 and parameters.mode == 1:
            print ('\nSegmented Images Mode - Nodule Section on plane YZ\n')
    
        return deepnetwork