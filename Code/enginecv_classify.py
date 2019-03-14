'''
Created on 21/03/2018
Modiefied on 26/02/2019

@author: Francesco Pugliese, Eleonora Bernasconi
'''
import pdb
import numpy as np
import seaborn as sns
from matplotlib import colors
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
# Program imports
from Settings.settings import SetParameters
from Preprocessing.preprocessing import load_EuroSat, load_EuroSat_classify
from Postprocessing.postprocessing import Postprocessing
from Initialization.initialization import Initialization
from Models.model_preparation import ModelPreparation
from Preprocessing.data_preparation import DataPreparation
from Postprocessing.postprocessing import Postprocessing
from View.view import Viewclass, Time
from Misc.utils import rotateImage, set_cpu_or_gpu

# Keras imports
from keras.preprocessing.image import array_to_img, img_to_array

# Other imports
import platform
import timeit
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import cv2

class Classify ():
    def clas(File):
        # Operating System
        OS = platform.system()                                                               # returns 'Windows', 'Linux', etc
                
        pie = None
        bar = None
        
        default_config_file = "enginecv.ini"                                                 # Default Configuration File

        n_samples = 1

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

        if parameters.interface == False:                                                    # Read the input file name in case there is not interface
            File = parameters.input_image
            
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
        elif parameters.kaggle_data == True:                                                 # Kaggle Lung Cancer Detection data
            data_type = "kaggle"                                                            

        # CLASSIFY
        jpg = 'Test_examples/Classification'
        
        if parameters.classification_type == "EuroSAT":  
            n_classes = 10
        elif parameters.classification_type == "Lucas":         
            n_classes = 5
            
        parameters.output_size = parameters.eurosat_output_size
        parameters.input_channels = parameters.eurosat_input_channels
        parameters.input_size = [parameters.eurosat_input_size, parameters.eurosat_input_size]

        if parameters.training == False:                                                    # Prende un file con indirizzo dato: File
            train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = load_EuroSat_classify(validation_split = 0, test_split = 0, shuffle = False, limit = None, file_name = File, input_size = parameters.eurosat_input_size, rescale = False, test_size = 0, parameters = parameters)
        else:                                                                               # Prende i file dalla cartella jpg
            train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = load_EuroSat(validation_split = 0, test_split = 0, shuffle = False, limit = None, datapath = jpg, input_size = parameters.eurosat_input_size, rescale = False, test_size = 0, parameters = parameters)
        
        #non_cat_test_set_y = test_set_y
        train_set_x, train_set_y, val_set_x, val_set_y, image, test_set_y  = DataPreparation.reshape_normalize(parameters, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y)     # Reshape and Normalize  
                
        if parameters.quantization == True: 
            n_quantizations = 0
            n_possible_samples = 3
            quantization_array = np.zeros([n_classes+1, parameters.n_samples+1], dtype='float')
        else: 
            parameters.n_samples = 1
            
        times = []
            
        for s in range(0, parameters.n_samples): 
            if parameters.quantization == True: 
                if s == 0:
                    parameters.stride = 64
                    parameters.rotate_tiles = False
                    parameters.random_rotations = False
                    n_quantizations = n_quantizations + 1
                elif s == 2: 
                    parameters.stride = 32
                    parameters.rotate_tiles = False
                    parameters.random_rotations = False
                    n_quantizations = n_quantizations + 1
                elif s == 1: 
                    parameters.stride = 64
                    parameters.rotate_tiles = True
                    parameters.random_rotations = True
                    n_quantizations = n_quantizations + 1
                print("Number of sample: ", n_quantizations)
            
                if parameters.n_samples > n_quantizations and s == n_possible_samples:
                    break
            
            # Set prefix of plots files
            if parameters.rotate_tiles == False: 
                rotation_type = 'no_rotation'
            else: 
                if parameters.random_rotations == False: 
                    rotation_type = '180'
                else: 
                    rotation_type = 'random'
                                
            prefix = parameters.classification_type.lower()+'_'+'stride_'+str(parameters.stride)+'_'+'rotation_'+rotation_type+'_'
            
            # LAND COVER EXTRACTION, computing classes array sizes
            vertical_n_images = 1 + (parameters.input_size[0] // parameters.stride) * (image.shape[2] // parameters.input_size[0] - 1)
            horizontal_n_images = 1 + (parameters.input_size[1] // parameters.stride) * (image.shape[3] // parameters.input_size[1] - 1)
           
            coords_list = []
            classes_list = []

            mp = ModelPreparation()
            deepnetwork = mp.model_selector(parameters, True, parameters.summary)                # Builds models for test
            classes_array = np.zeros([vertical_n_images, horizontal_n_images], dtype='int')
            
            if parameters.save_tiles == True: 
                if not os.path.exists(parameters.tiles_path):
                    os.makedirs(parameters.tiles_path)
            
            for i in range(vertical_n_images): 
                for k in range(horizontal_n_images): 
                    coords = [0 + i * parameters.stride, parameters.input_size[0] + i * parameters.stride, 0 + k * parameters.stride, parameters.input_size[1] + k * parameters.stride]

                    image_tile = image[:,:,coords[0]:coords[1],coords[2]:coords[3]][0:1]
                                    
                    # Test save the first image
                    image_tile_to_save = np.asarray(array_to_img(image_tile[0,:,:,:] * parameters.normalization_ratio_1 * parameters.normalization_ratio_2))
                    cv2.imwrite("../Output/tile_tmp.jpg", image_tile_to_save) 
                    
                    image_tile = cv2.imread("../Output/tile_tmp.jpg")
                    
                    if parameters.rotate_tiles:
                        angle = 180
                        
                        if parameters.random_rotations == True:
                            angle_idx = np.random.randint(4)
                            if angle_idx == 0:
                                angle = 0
                            elif angle_idx == 1: 
                                angle = 90
                            elif angle_idx == 2: 
                                angle = 180
                            elif angle_idx == 3: 
                                angle = 270
                        
                        image_tile = rotateImage(image_tile, angle)
                        
                    image_tile = img_to_array(image_tile)

                    data = []
                    data.append(image_tile)
                    data_set = np.array(data, dtype="float") / parameters.normalization_ratio_1 / parameters.normalization_ratio_2

                    preproc_time = timeit.default_timer() - global_start_time

                    classes = deepnetwork.predict_classes(data_set, batch_size=parameters.batch_size, verbose=0)

                    gpu_time = timeit.default_timer() - global_start_time - preproc_time

                    #probs = deepnetwork.predict_proba(data_set, batch_size=parameters.batch_size, verbose=0)
                    
                    if parameters.save_tiles == True: 
                        if not os.path.exists(os.path.join(parameters.tiles_path, Postprocessing.labels_to_eurosat_classes_converter(classes[0])+"_"+classes[0].__str__())):
                            os.makedirs(os.path.join(parameters.tiles_path, Postprocessing.labels_to_eurosat_classes_converter(classes[0])+"_"+classes[0].__str__()))

                        cv2.imwrite(parameters.tiles_path + '/' + Postprocessing.labels_to_eurosat_classes_converter(classes[0])+"_"+classes[0].__str__()+"/"+"tile"+"_"+(i+1).__str__()+"_"+(k+1).__str__()+".jpg", image_tile_to_save) 
                    
                    classes_list.append(classes)
                    coords_list.append(coords) 
                    
                    classes_array[i,k] = classes

            if os.path.exists(os.path.join(parameters.output_path, 'tile_tmp.jpg')):
                os.remove("../Output/tile_tmp.jpg") 
        
            x = []
            y = []
            x_len = int((len(coords_list))/vertical_n_images)
            y_len = int((len(coords_list))/horizontal_n_images)       
        
            for k in range(0, parameters.input_size[0]*x_len, parameters.input_size[0]):
                x.append(k)
            for j in range(parameters.input_size[1]*(y_len-1), 0, -parameters.input_size[1]):
                y.append(j) 
            y.append(0)
            y = y[::-1]
            
            csv = []
                
            if parameters.classification_type == "EuroSAT":  
                classes_array_list, countings = Postprocessing.eurosat_labels_counting(coords_list, classes_list, parameters.language)       # Counts the Eurosat Labels
                [fracs, labels, explode, colors] = Postprocessing.eurosat_statistics_compute(countings, coords_list, parameters.language)
            elif parameters.classification_type == "Lucas":         
                classes_array_list, countings = Postprocessing.lucas_labels_counting(coords_list, classes_list, parameters.language)       # Counts the Lucas Labels
                [fracs, labels, explode, colors] = Postprocessing.lucas_statistics_compute(countings, coords_list, parameters.language)
            
            if parameters.quantization == True: 
                quantization_array[:-1,s] = countings
                           
            print('\n')
            
            np.savetxt("../Output/Csv/classes_array_list.txt", classes_array_list, delimiter=",",newline='\r\n',fmt='%s')
            np.savetxt("../Output/Csv/x.csv", x, delimiter=",")        
            np.savetxt("../Output/Csv/y.csv", y, delimiter=",")
            np.savetxt("../Output/Csv/coords_list.csv", coords_list, delimiter=",")
                   
            # Maps
            if parameters.plot_maps == True:
                fig, ax = plt.subplots(figsize=(64*x_len, 64*(y_len-1)))
                im = ax.imshow(classes_array)
            
                # We want to show all ticks...
                ax.set_xticks(np.arange(len(x)))
                ax.set_yticks(np.arange(len(y)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(x)
                ax.set_yticklabels(y)
                
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                for i in range(len(y)):
                    for j in range(len(x)):
                        text = ax.text(j, i, classes_array[i, j], ha="center", va="center", color="white")

                fig.tight_layout()
                plt.savefig(parameters.submission_path  + '/Heatmap.jpg')
                plt.show()
                
            
            ## PLOT
            if parameters.plot_classify == True:
                sorted_indices = np.argsort(labels).astype('int')
                labels = np.asarray(labels)[sorted_indices]
                fracs = np.asarray(fracs)[sorted_indices]
                pie = Viewclass.plot_pie(colors = colors, parameters = parameters, labels = labels, fracs = fracs, title = Postprocessing.get_plot_classes_title(parameters)[0], save_pieplot = parameters.save_pieplot, charts_path = parameters.charts_path, pieplot_file = prefix+parameters.pieplot_file, show = parameters.plots_show, pause_time = parameters.pause_time)

                #bar = Viewclass.plot_bar_chart(colors = colors,  parameters = parameters, labels = labels,fracs = fracs)
                bar = None    
            else: 
                pie = None
                bar = None
                
            post_proc_time = timeit.default_timer() - global_start_time - preproc_time - gpu_time

            times.append([preproc_time, gpu_time, post_proc_time, preproc_time+post_proc_time])
            
        if parameters.plot_ground_truth == True:
            data = np.asarray(pd.read_csv(parameters.ground_truth_path+'/'+parameters.ground_truth_file, sep=';', usecols = [5, 6])) 
            labels_gt = [x.lower().capitalize() for x in data[:,0]]
            fracs_gt = [float(x.replace('%', '').replace(',','.')) for x in data[:,1]]

            sorted_indices_gt = np.argsort(labels_gt).astype('int')
            labels_gt = np.asarray(labels_gt)[sorted_indices_gt]
            fracs_gt = np.asarray(fracs_gt)[sorted_indices_gt]

            pie_gt = Viewclass.plot_pie(colors = colors, parameters = parameters, labels = labels_gt, fracs = fracs_gt, title = Postprocessing.get_plot_classes_title(parameters)[1], save_pieplot = parameters.save_pieplot, charts_path = parameters.charts_path, pieplot_file = parameters.ground_truth_pieplot_file, show = parameters.plots_show, pause_time = parameters.pause_time)

            bar_gt = None    
        else: 
            pie_gt = None
            bar_gt = None
        
        if parameters.quantization == True: 
            # computing marginals
            for n_class in range(n_classes):
                quantization_array[n_class,-1] = quantization_array[n_class,:].sum() / quantization_array.sum()
            for n_sample in range(parameters.n_samples):
                quantization_array[-1,n_sample] = quantization_array[:,n_sample].sum() / quantization_array.sum()

            quantization_array[-1,-1] = quantization_array[:,:].sum() / quantization_array.sum()
 
            ## PLOT
            if parameters.plot_classify == True:
                fracs = quantization_array[:-1,-1]*100      # extracts last column in percentage
                fracs = fracs[fracs!=0]                     # removes zeros
                fracs = np.asarray(fracs)[sorted_indices]
                pie = Viewclass.plot_pie(colors = colors, parameters = parameters, labels = labels, fracs = fracs, title = Postprocessing.get_plot_classes_title(parameters)[2], save_pieplot = parameters.save_pieplot, charts_path = parameters.charts_path, pieplot_file = parameters.classification_type.lower()+'_'+"quantification_"+parameters.pieplot_file, show = parameters.plots_show, pause_time = parameters.pause_time)

                bar = None    
            else: 
                pie = None
                bar = None
        
        if parameters.plot_times == True:
            fracs = np.asarray(times[0])
            labels_times = ["Preprocessing Time", "Gpu Time", "Postprocessing Time", "Global Time"]
            title_times = "Times \n\n"+"Operating System: "+OS
            bar = Viewclass.plot_bar_chart(colors = colors, parameters = parameters, labels = labels_times, fracs = fracs, title = title_times, save_pieplot = parameters.save_pieplot, charts_path = parameters.charts_path, pieplot_file = parameters.classification_type.lower()+'_'+"quantification_"+parameters.pieplot_file, show = parameters.plots_show, pause_time = parameters.pause_time)
            
        return (pie, bar, parameters) 

global_start_time = timeit.default_timer()

File = '../Input/ExportLecceCropTest4.jp2'        
pie, bar, parameters = Classify.clas(File)

global_end_time = timeit.default_timer()

global_time_in_sec = global_end_time - global_start_time
print ('\n\n')
t = Time()
t.print_computation_time_in_days(global_time_in_sec, 0)                  # displays computation time in days, hours, etc
print ('\n\n')
