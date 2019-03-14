'''
Created on 21/03/2018
Modiefied on 17/07/2018

@author: Francesco Pugliese, Matteo Alberti, Eleonora Bernasconi
'''

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# File imports
from Preprocessing.data_preparation import DataPreparation
from Settings.settings import SetParameters
from Initialization.initialization import Initialization
from Models.model_preparation import ModelPreparation
from View.view import Viewclass

# Other imports
import os
import platform

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

# Set CPU or GPU type
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
if parameters.gpu == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = parameters.gpu_id

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
non_cat_test_set_y = test_set_y
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y  = DataPreparation.reshape_normalize(parameters, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y)     # Reshape and Normalize  

# MODEL EVALUATION        

if parameters.cifar10_benchmark or parameters.mnist_benchmark == True or parameters.trash_data == True or parameters.eurosat_data == True: 
    '''
    if parameters.neural_model == "capsnet":
        data_test = test_set_x, test_set_y
        CapsuleNet.test(model, data_test)
    '''
    print ('\nTesting the Trained Neural Network...\n')
    mp = ModelPreparation()
    deepnetwork = mp.model_selector(parameters, True)               # Builds models for test

    [loss, accuracy] = deepnetwork.evaluate(test_set_x, test_set_y, verbose = 1)

    classes1 = deepnetwork.predict_classes(test_set_x, verbose=1)
    classes2 = deepnetwork.predict_proba(test_set_x, verbose=0)

    print ("\nAccuracy  : %.2f%%" % (accuracy * 100))
    print ("Loss      : %f" % loss)
    print ()

    if parameters.print_confusion_matrix ==True:                                                                                                                        # Plot Confusion Matrix
        Viewclass.confmat(parameters = parameters, deepnetwork = deepnetwork, test_set_x = test_set_x, test_set_y = test_set_y)
    if parameters.plot_history == True:
        Viewclass.plot_hist_test(parameters = parameters)

'''
else: 
    for p in range(parameters.num_projections): 
        if p == 0 and parameters.mode == 1:
           print ('\nTesting Nodule Section on plane XY\n')
           test_set_x = test_set_xy_x
           test_set_y = test_set_xy_y
           test_set_patients_ids = test_set_xy_patients_ids
        elif p == 1 and parameters.mode == 1:
           print ('\nTesting Nodule Section on plane XZ\n')
           test_set_x = test_set_xz_x
           test_set_y = test_set_xz_y
           test_set_patients_ids = test_set_xz_patients_ids
        elif p == 2 and parameters.mode == 1:
           print ('\nTesting Nodule Section on plane YZ\n')
           test_set_x = test_set_yz_x
           test_set_y = test_set_yz_y
           test_set_patients_ids = test_set_yz_patients_ids

        if parameters.mode == 1:
            print ('\nTest set values size (X): %i x %i' % (test_set_x.shape[0], test_set_x.shape[1]))
            print ('Test set target vector size (Y): %i x 1' % test_set_y.shape[0])
            print ('Sum of test set values (X): %.2f' % test_set_x.sum());
            print ('Sum of test set target (Y): %i' % test_set_y.sum());

            normal_test_set_y = test_set_y

            # Reshape and add an axis to each data set in order to pass it to the conv layer. Transform labels into integer categorical arrays too.     
            if parameters.output_size > 1:
                test_set_x, test_set_y = [test_set_x.reshape((test_set_x.shape[0], parameters.input_size[0], parameters.input_size[1]))[:,np.newaxis,:,:], np.utils.to_categorical(test_set_y.astype("int"), parameters.output_size)]
            else: 
                test_set_x, test_set_y = [test_set_x.reshape((test_set_x.shape[0],parameters.input_size[0], parameters.input_size[1]))[:,np.newaxis,:,:], test_set_y.astype("int")]

        [prediction_classes, prediction_probs] = predictClasses(deepnetworks[p], test_set_x, batch_size = parameters.batch_size, verbose = 0)
        prediction_output = np.vstack((prediction_classes, normal_test_set_y.astype("int")))

        # qui mettere un controllo nel caso si usi la binary cross entropy 
        chosen_prediction_probs = np.reshape(prediction_probs[:,1], (len(prediction_probs[:,1]), 1)) 
        if parameters.print_confusion_matrix == True: 
            print ('Testing the Last Trained Network...\n')
            printConfusionMatrix(prediction_output)

        if parameters.save_best_model == True:
            topology = DataPreparation.topology_selector(parameters)
            opt = DataPreparation.get_training_algorithm(parameters)
            bestDeepNetwork = topology.build(width=parameters.input_size[0], height=parameters.input_size[1], depth=parameters.input_channels, classes=parameters.output_size, summary = False) 
            bestDeepNetwork.load_weights(os.path.join("./",parameters.models_path)+"/best_kaggle_model_"+str(p)+".cnn")       
            bestDeepNetwork.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            [bestPredictionClasses, bestPrediction_probs] = predictClasses(bestDeepNetwork, test_set_x, batch_size = parameters.batch_size, verbose = 0)
            bestPredictionOutput = np.vstack((bestPredictionClasses, normal_test_set_y.astype("int")))
    
            # qui mettere un controllo nel caso si usi la binary cross entropy 
            chosen_prediction_probs = np.reshape(bestPrediction_probs[:,1], (len(bestPrediction_probs[:,1]), 1)) 
            if parameters.print_confusion_matrix == True: 
                print ('\n\nTesting the Best Trained Network...\n')
                printConfusionMatrix(bestPredictionOutput)
'''

