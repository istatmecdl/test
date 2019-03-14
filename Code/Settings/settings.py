'''
Created on 23/03/2018


@author: Francesco Pugliese, Eleonora Bernasconi
'''

import configparser

class SetParameters:
    
    def __init__(self, conf_file_path, conf_file_name, OS):
        # Class Initialization (constructor) 
        self.conf_file_path = conf_file_path
        self.conf_file_name = conf_file_name
        
        # SYSTEM
        self.gpu = False
        self.gpu_id = '0'
        
        # PREPROCESSING
        self.print_output = True
        self.double_normalization = False
        self.normalization_ratio_1 = 255.0  
        self.normalization_ratio_2 = 255.0

        # Kaggle Lung Cancer competition
        self.mode = 0                                                    # 0: Read, train and evaluate on Whole images, 1: Read, train and evaluate on Segmented images
        self.num_projections = 3                                         # for mode = 1: number of projections per nodule for Kaggle competitions
        self.max_elements_train = -1                                     # for both modes: number of max patients, -1 means all the patients
        self.max_segments_per_element_train = -1                         # for mode = 1: max number of nodules per patient to extract, -1 means all the nodules
        self.max_elements_test = -1                                      # for both modes: number of max patients, -1 means all the patients
        self.max_segments_per_element_test = -1                          # for mode = 1: max number of nodules per patient to extract, -1 means all the nodules

        # General Input Preprocessing
        self.truePath = "1"
        self.falsePath = "0"
        self.testPath = "NO"
        self.externalValidset = False                                    # True: use an external provided validation set
        self.validTruePath = "1"
        self.validFalsePath = "0"
        self.dataAugmentation = False
        self.saveAugmentedData = False

        # Dataset
        self.language = 'en'
        if self.mode==0:                 # Original Images
            self.kaggle_dataset_path = '../../../Kaggle_Data/img2'
            self.kaggle_validset_path = '../../../Kaggle_Data/img2_VALID'
        else:                       # Segmented Images
            self.kaggle_dataset_path_segmented = '../../../Kaggle_Data/Segmented_new'
            self.kaggle_validset_path_segmented = '../../../Kaggle_Data/Segmented_VALID'
            
        self.cifar10_dataset_path = '/home/dli2017/Cifar10_Data/cifar-10-batches-py'
        self.mnist_dataset_path = '/home/dli2017/MNIST_data'
        self.valid_set_perc = 3                                          # Validation set percentage with respect to the Training Set
        self.test_set_perc = 3                                           # Test set percentage with respect to the entire Data Set
        self.normalize_x = False                                         # Normalize input between 0 and 1, time-consuming for bigger datasets
        self.normalize_y = False                                         # Normalize input between 0 and 1, time-consuming for bigger datasets
        self.limit = None
        
        # LOCAL DATA
        #trash_data
        self.trash_data=True
        self.trash_epochs_number = 100
        self.trash_valid_set = True
        self.trash_output_size = 5
        self.trash_input_channels = 3
        self.trash_batch_size = 64
        self.trash_input_size = 256

        #eurosat_data
        self.eurosat_data=True
        self.eurosat_dataset_path = ""
        self.eurosat_input_size = 64
        self.eurosat_input_channels = 3
        self.eurosat_output_size = 10
        self.classification_type = "EuroSAT"
  
        # OUTPUT
        self.output_path = '../Output'
        self.preprocessed_images_path = '../Output/PreprocessedImages'
        self.print_confusion_matrix = True
        self.write_submit_file = True
        self.submission_path = '../Output/Metrics'
        self.submission_txtfile_path = '../Output/Csv'
        self.plot_history = True
        self.plot_classify = True
        self.plot_maps = False
        self.save_pieplot = True
        self.charts_path = '../Output/Charts' 
        self.pieplot_file = 'classification_pieplot.jpg' 
        self.plot_ground_truth = True
        self.plots_show = True
        self.ground_truth_path = '../Input/Ground_Truth'
        self.ground_truth_file = 'StatisticheGlobaliLucasFinal.csv'
        self.ground_truth_pieplot_file = 'ground_truth_pieplot.jpg'
        self.pause_time = 2
        self.save_tiles = True
        self.tiles_path = '../Output/Tiles'
        self.plot_times = True
        self.plot_times_files = 'times_barplot.jpg'

        # MODEL 
        self.neural_model = 'lenet'
        self.models_path = "../SavedModels"
        self.model_file = "model.cnn"
        self.summary = True

        # TRAINING
        self.epochs_number = 100
        self.learning_rate = 0.0001                                      # best learning rate = 0.0015 at moment, 0.001 on mcover
        self.early_stopping = True
        self.save_best_model = True
        self.batch_size = 32
        self.rescale = True                                              # True: adapt the data size to required input size, False: adapt input layer to the data size                                                                                                                                                                                                    
        self.training_algorithm = "sgd"
        self.training = False
        
        # TOPOLOGY
        self.ins = 40
        self.input_size = [self.ins, self.ins]
        self.input_channels = 1                                          # Number of Input Channels
        self.output_size = 2                                             # Number of Output Classes

        # BENCHMARKING
        # Cifar10 benchmarking
        self.cifar10_benchmark = True
        self.cifar10_dataset_path = ''
        self.cifar10_epochs_number = 100
        self.cifar10_use_valid_set = True
        self.cifar10_output_size = 10
        self.cifar10_input_channels = 3
        self.cifar10_batch_size = 32
        self.cifar10_input_size = 32

        # Mnist benchmarking
        self.mnist_benchmark=True
        self.mnist_dataset_path = ''
        self.mnist_epochs_number = 100
        self.mnist_use_valid_set = True
        self.mnist_output_size = 10
        self.mnist_input_channels = 1
        self.mnist_batch_size = 28
        self.mnist_input_size = 28

		# Others
        self.OS = OS
        
        # GLOBAL CONSTANTS
        # Alert Sound
        self.sound_freq = 1000                                           # Set Frequency in Hertz
        self.sound_dur = 3000                                            # Set Duration in ms, 1000 ms == 1 second
        
        # Classification
        self.interface = True
        self.input_image = 'Test_examples/Eurosat_example_01.jpg'
        self.save_tiles = False
        self.stride = 64
        self.rotate_tiles = False
        self.random_rotations = False
        self.quantization = False
        self.n_samples = 3        
        
    def read_config_file(self):
        # Read the Configuration File
        config = configparser.ConfigParser()
        config.read(self.conf_file_path+'/'+self.conf_file_name)
        config.sections()

        # System
        self.gpu = config.getboolean('System', 'gpu')
        self.gpu_id = config.get('System','gpu_id')

        # PREPROCESSING
        self.double_normalization = config.getboolean('Preprocessing', 'double_normalization')
        if self.double_normalization == True: 
            self.normalization_ratio_1 = 255.0  
            self.normalization_ratio_2 = 255.0
        else: 
            self.normalization_ratio_1 = 255.0  
            self.normalization_ratio_2 = 1.0

        self.print_output = config.getboolean('Preprocessing', 'print_output')

        # Kaggle Lung Cancer competition
        self.mode = config.getint('Preprocessing', 'mode')
        self.num_projections = config.getint('Preprocessing', 'num_projections')
        self.max_elements_train = config.getint('Preprocessing', 'max_elements_train')
        self.max_segments_per_element_train = config.getint('Preprocessing', 'max_segments_per_element_train')
        self.max_elements_test = config.getint('Preprocessing', 'max_elements_test')
        self.max_segments_per_element_test = config.getint('Preprocessing', 'max_segments_per_element_test')

        # General Input Preprocessing
        self.truePath = config.get('Preprocessing', 'truePath')
        self.falsePath = config.get('Preprocessing', 'falsePath')
        self.testPath = config.get('Preprocessing', 'testPath')
        self.normalize_x = config.getboolean('Preprocessing', 'externalValidset')
        self.validTruePath = config.get('Preprocessing', 'validTruePath')
        self.validFalsePath = config.get('Preprocessing', 'validFalsePath')
        self.dataAugmentation = config.getboolean('Preprocessing', 'dataAugmentation')
        self.saveAugmentedData = config.getboolean('Preprocessing', 'saveAugmentedData')

        # Dataset7
        self.language = config.get('Dataset', 'language')
        if self.mode==0:                 # Original Images
            self.kaggle_dataset_path = config.get('Dataset', 'kaggle_dataset_path')
            self.kaggle_validset_path = config.get('Dataset', 'kaggle_validset_path')
        else:                       # Segmented Images
            self.kaggle_dataset_path_segmented = config.get('Dataset', 'kaggle_dataset_path')
            self.kaggle_validset_path_segmented = config.get('Dataset', 'kaggle_validset_path')
            
        self.valid_set_perc = config.getint('Dataset', 'valid_set_perc')
        self.test_set_perc = config.getint('Dataset', 'test_set_perc')
        self.normalize_x = config.getboolean('Dataset', 'normalize_x')
        self.normalize_y = config.getboolean('Dataset', 'normalize_y')
        
        self.limit = config.get('Dataset', 'limit')
        try: 
            self.limit = int(self.limit)
        except ValueError: 
            self.limit = None
        
        # LOCAL DATA
        #trash_data
        self.trash_data = config.getboolean('Local_data', 'trash_data')
        self.trash_epochs_number = config.getint('Local_data', 'trash_epochs_number')
        self.trash_valid_set = config.getboolean('Local_data', 'trash_valid_set')
        self.trash_output_size = config.getint('Local_data', 'trash_output_size')
        self.trash_input_channels = config.getint('Local_data', 'trash_input_channels')
        self.trash_batch_size = config.getint('Local_data', 'trash_batch_size')
        self.trash_input_size = config.getint('Local_data', 'trash_input_size')
        self.classification_type = config.get('Local_data', 'classification_type')

        #eurosat_data
        self.eurosat_data = config.getboolean('Local_data', 'eurosat_data')
        self.eurosat_dataset_path = config.get('Local_data', 'eurosat_dataset_path')
        self.eurosat_input_size = config.getint('Local_data', 'eurosat_input_size')
        self.eurosat_input_channels = config.getint('Local_data', 'eurosat_input_channels')
        self.eurosat_output_size = config.getint('Local_data', 'eurosat_output_size')
        
        # OUTPUT
        output_path = config.get('Output', 'output_path')
        self.preprocessed_images_path = config.get('Output', 'preprocessed_images_path')
        self.print_confusion_matrix = config.getboolean('Output', 'print_confusion_matrix')
        self.plot_history = config.getboolean('Output', 'plot_history')
        self.write_submit_file = config.getboolean('Output', 'write_submit_file')
        self.submission_path = config.get('Output', 'submission_path')
        self.submission_txtfile_path = config.get('Output', 'submission_txtfile_path')
        self.plot_classify = config.getboolean('Output', 'plot_classify')
        self.plot_maps = config.getboolean('Output', 'plot_maps')
        self.save_pieplot = config.getboolean('Output', 'save_pieplot')
        self.charts_path = config.get('Output', 'charts_path') 
        self.pieplot_file = config.get('Output', 'pieplot_file') 
        self.plot_ground_truth = config.getboolean('Output', 'plot_ground_truth')
        self.plots_show = config.getboolean('Output', 'plots_show')
        self.ground_truth_path = config.get('Output', 'ground_truth_path')
        self.ground_truth_file = config.get('Output', 'ground_truth_file')
        self.ground_truth_pieplot_file = config.get('Output', 'ground_truth_pieplot_file')
        self.pause_time = config.getint('Output', 'pause_time')
        self.save_tiles = config.getboolean('Output', 'save_tiles')
        self.tiles_path = config.get('Output', 'tiles_path')
        self.plot_times = config.getboolean('Output', 'plot_times')
        self.plot_times_files = config.get('Output', 'plot_times_files')

        # MODEL 
        self.neural_model = config.get('Model', 'neural_model')
        self.models_path = config.get('Model', 'models_path')
        self.model_file = config.get('Model', 'model_file')
        self.summary = config.getboolean('Model', 'summary')

        # TRAINING
        self.epochs_number = config.getint('Training', 'epochs_number')
        self.learning_rate = config.getfloat('Training', 'learning_rate')
        self.early_stopping = config.getboolean('Training', 'early_stopping')
        self.save_best_model = config.getboolean('Training', 'save_best_model')
        self.batch_size = config.getint('Training', 'batch_size')
        self.rescale = config.getboolean('Training', 'rescale')
        self.training_algorithm = config.get('Training', 'training_algorithm')
        self.training = config.getboolean('Training', 'training')

        # TOPOLOGY
        self.ins = config.getint('Topology', 'input_size')
        self.input_size = [self.ins, self.ins]                           # The Model's Required Input Size
        self.input_channels = config.getint('Topology', 'input_channels')
        self.output_size = config.getint('Topology', 'output_size')

        # BENCHMARKING
        # Cifar10 benchmarking
        self.cifar10_benchmark = config.getboolean('Benchmarking', 'cifar10_benchmark')
        self.cifar10_dataset_path = config.get('Benchmarking', 'cifar10_dataset_path')
        self.cifar10_epochs_number = config.getint('Benchmarking', 'cifar10_epochs_number')
        self.cifar10_use_valid_set = config.getboolean('Benchmarking', 'cifar10_use_valid_set')
        self.cifar10_output_size = config.getint('Benchmarking', 'cifar10_output_size')
        self.cifar10_input_channels = config.getint('Benchmarking', 'cifar10_input_channels')
        self.cifar10_batch_size = config.getint('Benchmarking', 'cifar10_batch_size')
        self.cifar10_input_size = config.getint('Benchmarking', 'cifar10_input_size')

        # Mnist benchmarking
        self.mnist_benchmark = config.getboolean('Benchmarking', 'mnist_benchmark')
        self.mnist_epochs_number = config.getint('Benchmarking', 'mnist_epochs_number')
        self.mnist_use_valid_set = config.getboolean('Benchmarking', 'mnist_use_valid_set')
        self.mnist_output_size = config.getint('Benchmarking', 'mnist_output_size')
        self.mnist_input_channels = config.getint('Benchmarking', 'mnist_input_channels')
        self.mnist_batch_size = config.getint('Benchmarking', 'mnist_batch_size')
        self.mnist_input_size = config.getint('Benchmarking', 'mnist_input_size')
		
        # Classification
        self.interface = config.getboolean('Classification', 'interface')
        self.input_image = config.get('Classification', 'input_image')
        self.save_tiles = config.getboolean('Classification', 'save_tiles')
        self.stride = config.getint('Classification', 'stride')
        self.rotate_tiles = config.getboolean('Classification', 'rotate_tiles')
        self.random_rotations = config.getboolean('Classification', 'random_rotations')
        self.quantization = config.getboolean('Classification', 'quantization')
        self.n_samples = config.getint('Classification', 'n_samples')

        return self		
