'''
Created on 22/03/2017

@author: Eleonora Bernasconi, Matteo Alberti, Francesco Pugliese
'''

# import matplot
import matplotlib
matplotlib.use("Agg")

# import keras 
from keras.preprocessing.image import img_to_array, array_to_img

# import os
import os
from os import listdir
from os.path import isfile, isdir, join

#import sk
from sklearn.model_selection import train_test_split

#other imports
import timeit
from skimage import io
import scipy.misc
import numpy as np
from imutils import paths
import argparse
import random
import cv2
import sys
import platform
import pdb

def load_EuroSat(validation_split, test_split, shuffle, limit, datapath, input_size, rescale, test_size, parameters):
    
    # load data
    load_start_time = timeit.default_timer()
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(datapath)))
    # loop over the input images
    data = []
    labels = []
    count = 0
    for imagePath in imagePaths:
        
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        if rescale == True: 
            image = cv2.resize(image, (input_size, input_size))
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        if label == "AnnualCrop":
            label = 0
        elif label == "Forest":
            label = 1
        elif label == "HerbaceousVegetation":
            label = 2
        elif label == "Highway":
            label = 3
        elif label == "Industrial":
            label = 4
        elif label == "Pasture":
            label = 5
        elif label == "PermanentCrop":
            label = 6
        elif label == "Residential":
            label = 7
        elif label == "River":
            label = 8
        else :
            label = 9
        labels.append(label)
    
        if limit is not None:
            count += 1
            
            if count>limit:
                break

    data_set = np.array(data, dtype="float") / parameters.normalization_ratio_1
    labels = np.array(labels)

    # split the data into a training set and a validation set
    indices = np.arange(data_set.shape[0])
    if shuffle == True: 
        np.random.shuffle(indices)
    data_set = data_set[indices]
    labels = labels[indices]
    
    num_validation_samples = int(round(validation_split * data_set.shape[0]))
    num_test_samples = int(round(test_split * data_set.shape[0]))
    train_set_x = data_set[:-(num_test_samples+num_validation_samples)]
    train_set_y = labels[:-(num_test_samples+num_validation_samples)]
    val_set_x = data_set[-(num_test_samples+num_validation_samples):-num_test_samples]
    val_set_y = labels[-(num_test_samples+num_validation_samples):-num_test_samples]
    test_set_x = data_set[-num_test_samples:]
    test_set_y = labels[-num_test_samples:]
    
    print ('\n\nLoading time: %.2f minutes\n' % ((timeit.default_timer() - load_start_time) / 60.))
   
    return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]

def load_Corine(validation_split, test_split, shuffle, limit, datapath, input_size, rescale, test_size, parameters):
    
    # load data
    load_start_time = timeit.default_timer()
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(datapath)))
    # loop over the input images
    data = []
    labels = []
    count = 0
    for imagePath in imagePaths:
        
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        if rescale == True: 
            image = cv2.resize(image, (input_size, input_size))
        
        #print(imagePath)
        
        image = img_to_array(image)

        if image.shape == (3,32,32):
            data.append(image)
            # extract the class label from the image path and update the labels list
            label = imagePath.split(os.path.sep)[-2]
            if label == "0_Sea_and_ocean":
                label = 0
            elif label == "1_Discontinuous_urban_fabric":
                label = 1
            elif label == "2_Complex_cultivation_patterns":
                label = 2
            elif label == "3_Olive_grooves":
                label = 3
            elif label == "4_Non-irrigated_arable_land":
                label = 4
            elif label == "5_Pastures":
                label = 5
            elif label == "6_Continuous_urban_fabric":
                label = 6
            elif label == "7_Salt_marshes":
                label = 7
            elif label == "8_Land_principally_occupied_by_agriculture__with_significant_areas_of_natural_vegetation":
                label = 8
            elif label == "9_Sclerophyllous_vegetation":
                label = 9
            elif label == "10_Coniferous_forest":
                label = 10
            elif label == "11_Mixed_forest":
                label = 11
            elif label == "12_Annual_crops_associated_with_permanent_crops":
                label = 12
            elif label == "13_Mineral_extraction_sites":
                label = 13
            elif label == "14_Fruit_trees_and_berry_plantations":
                label = 14
            elif label == "15_Vineyards":
                label = 15
            elif label == "16_Agro-forestry_areas":
                label = 16
            elif label == "17_Industrial_or_commercial_units":
                label = 17
            elif label == "18_Sport_and_leisure_facilities":
                label = 18
            elif label == "19_Coastal_lagoons":
                label = 19
            elif label == "20_Airports":
                label = 20
            elif label == "21_Broad-leaved_forest":
                label = 21
            elif label == "22_Dump_sites":
                label = 22
            elif label == "23_Natural_grassland":
                label = 23
            labels.append(label)
        
            if limit is not None:
                count += 1
                
                if count>limit:
                    break

    data_set = np.array(data, dtype="float") / parameters.normalization_ratio_1
    labels = np.array(labels)

    # split the data into a training set and a validation set
    indices = np.arange(data_set.shape[0])
    if shuffle == True: 
        np.random.shuffle(indices)
    data_set = data_set[indices]
    labels = labels[indices]
    
    num_validation_samples = int(round(validation_split * data_set.shape[0]))
    num_test_samples = int(round(test_split * data_set.shape[0]))
    train_set_x = data_set[:-(num_test_samples+num_validation_samples)]
    train_set_y = labels[:-(num_test_samples+num_validation_samples)]
    val_set_x = data_set[-(num_test_samples+num_validation_samples):-num_test_samples]
    val_set_y = labels[-(num_test_samples+num_validation_samples):-num_test_samples]
    test_set_x = data_set[-num_test_samples:]
    test_set_y = labels[-num_test_samples:]
    
    print ('\n\nLoading time: %.2f minutes\n' % ((timeit.default_timer() - load_start_time) / 60.))
    
    return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]

def load_EuroSat_classify(validation_split, test_split, shuffle, limit, file_name, input_size, rescale, test_size, parameters):
    
    # load data
    load_start_time = timeit.default_timer()
    # loop over the input images
    data = []
    labels = []
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(file_name)
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if rescale == True: 
        image = cv2.resize(image, (input_size, input_size))

    image = img_to_array(image) # reshape
    
    data.append(image)

    data_set = np.array(data, dtype="float") / parameters.normalization_ratio_1
    labels = np.array(labels)

    # split the data into a training set and a validation set
    indices = np.arange(data_set.shape[0])
    if shuffle == True: 
        np.random.shuffle(indices)
    data_set = data_set[indices]
    
    num_validation_samples = int(round(validation_split * data_set.shape[0]))
    num_test_samples = int(round(test_split * data_set.shape[0]))
    train_set_x = data_set[:-(num_test_samples+num_validation_samples)]
    train_set_y = labels[:-(num_test_samples+num_validation_samples)]
    val_set_x = data_set[-(num_test_samples+num_validation_samples):-num_test_samples]
    val_set_y = labels[-(num_test_samples+num_validation_samples):-num_test_samples]
    test_set_x = data_set[-num_test_samples:]
    test_set_y = labels[-num_test_samples:]
    
    print ('\n\nLoading time: %.2f minutes\n' % ((timeit.default_timer() - load_start_time) / 60.))
   
    return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]
    
def load_kaggle_data(datapath='', preprocimgspath='', normalize_x = False, normalize_y = False, input_channels = 3, input_size = [512,512], rescale = True, truePath="", falsePath="", testPath=""):
    
    onechannel = False
	
    if input_channels == 1: 
        onechannel = True
	
    # Set files
    print('\nLoading Kaggle Datasets from the files...')
	
    noncancerpatientslist = np.asarray([f for f in listdir(datapath+"/"+falsePath) if isdir(join(datapath+"/"+falsePath, f))])
    cancerpatientslist = np.asarray([f for f in listdir(datapath+"/"+truePath) if isdir(join(datapath+"/"+truePath, f))])
    testpatientslist = np.asarray([f for f in listdir(datapath+"/"+testPath) if isdir(join(datapath+"/"+testPath, f))])

    train_set = []
	# Reads all the non-cancer patients
    print ("\n\nReading the Non-cancer patients images", end='', flush=True)
    #for i in range(10):
    for i in range(noncancerpatientslist.shape[0]):
        print('.', end='', flush=True)
        imgfileslist = np.sort(np.asarray([f for f in listdir(datapath+"/"+falsePath+"/"+noncancerpatientslist[i]) if isfile(join(datapath+"/"+falsePath+"/"+noncancerpatientslist[i], f))]))
        image = scipy.misc.imread(datapath+"/"+falsePath+"/"+noncancerpatientslist[i]+"/"+imgfileslist[1], flatten = onechannel, mode='RGB')
        if rescale == True: 
            image = scipy.misc.imresize(image, (input_size[0], input_size[1]))
        train_set.append([np.asarray(image).flatten(),0,noncancerpatientslist[i]])
 		
    # Reads all the cancer patients
    print ("\n\nReading the Cancer patients images", end='', flush=True)
    #for i in range(10):
    for i in range(cancerpatientslist.shape[0]):
        print('.', end='', flush=True)
        imgfileslist = np.sort(np.asarray([f for f in listdir(datapath+"/"+truePath+"/"+cancerpatientslist[i]) if isfile(join(datapath+"/"+truePath+"/"+cancerpatientslist[i], f))]))
        image = scipy.misc.imread(datapath+"/"+truePath+"/"+cancerpatientslist[i]+"/"+imgfileslist[1], flatten = onechannel, mode='RGB')
        if rescale == True: 
            image = scipy.misc.imresize(image, (input_size[0], input_size[1]))
        train_set.append([np.asarray(image).flatten(),1,cancerpatientslist[i]])
        if i==0:
            scipy.misc.imsave(preprocimgspath+"/"+'lung_x_ray_train_set_example.jpg', image)  # for test
		
    # Converts training set into an array	
    train_set = np.asarray(train_set)

    test_set = []
    print ("\n\nReading the Test patients images", end='', flush=True)
    #for i in range(10):
    for i in range(testpatientslist.shape[0]):
        print('.', end='', flush=True)
        imgfileslist = np.sort(np.asarray([f for f in listdir(datapath+"/"+testPath+"/"+testpatientslist[i]) if isfile(join(datapath+"/"+testPath+"/"+testpatientslist[i], f))]))

        image = scipy.misc.imread(datapath+"/"+testPath+"/"+testpatientslist[i]+"/"+imgfileslist[1], flatten = onechannel, mode='RGB')
        if rescale == True: 
            image = scipy.misc.imresize(image, (input_size[0], input_size[1]))
        test_set.append([np.asarray(image).flatten(),0,testpatientslist[i]])
        if i==0:
            scipy.misc.imsave(preprocimgspath+"/"+'lung_x_ray_test_set_example.jpg', image)  # for test
		
    # Converts test set into an array	
    test_set = np.asarray(test_set)

    # General Shuffle
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    # Extracts attributes and labels from training set 
    train_set_x = train_set[:,0]
    train_set_y = train_set[:,1].astype("int")   																				# extracts the target			
    train_set_patients_ids = train_set[:,2]
	
    # Extracts attributes and labels from test set 
    test_set_x = test_set[:,0]
    test_set_y = test_set[:,1].astype("int") 																					    			# NOTE! there are no targets provided in test set !			
    test_set_patients_ids = test_set[:,2]
	
    input_size = train_set_x[0].shape
	
    # Converts attributes arrays
    train_set_x = np.vstack(train_set_x)
    test_set_x = np.vstack(test_set_x)
	
    if normalize_x == True:
        train_set_x = train_set_x / train_set_x.max()								
        test_set_x = test_set_x / test_set_x.max()								
        
    if normalize_y == True:
        train_set_y = train_set_y / train_set_y.max()
        test_set_y = test_set_y / test_set_y.max()
	
    rval = [(train_set_x, train_set_y, train_set_patients_ids), (test_set_x, test_set_y, test_set_patients_ids)]
	
    return rval

def load_trash_data():
    global train_set_x, test_set_x, train_set_y, test_set_y
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    OS = platform.system()    
# Read the Configuration File
    if OS == "Windows":
        ap.add_argument("-d", "--dataset", default="C:\\Users\\Alber\\Desktop\\trash", required=False, help="path to input dataset")
        ap.add_argument("-m", "--model", default="import_data", required=False, help="path to output model")
        ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
    elif OS == "Linux":
        ap.add_argument("-d", "--dataset", default="/home/dli2017/TrashNet_Data/trash", required=False, help="path to input dataset")
        ap.add_argument("-m", "--model", default="import_data", required=False, help="path to output model")
        ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())


#    EPOCHS = 150
#    INIT_LR = 1e-3
#    BS = 32
 
# initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []
 
# grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    random.seed(42)
    random.shuffle(imagePaths)
# loop over the input images
    for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (256, 256))
        image = img_to_array(image)
        data.append(image)
 
    # extract the class label from the image path and update the
    # labels list
        label = imagePath.split(os.path.sep)[-2]
        if label == "cardboard":
            label = 0
        elif label == "plastic":
            label = 1
        elif label == "paper":
            label = 2
        elif label == "metal":
            label = 3
        elif label == "glass":
            label = 4
    
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
    '''
    data = np.array(data, dtype="float") / parameters.normalization_ratio_1
    labels = np.array(labels)
    '''
    data=np.array(data)
    labels= np.array(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
    (train_set_x, test_set_x, train_set_y, test_set_y) = train_test_split(data, labels, test_size=0.25, random_state=42)
 
# convert the labels from integers to vectors
    '''
    mem_train_set_x = train_set_x
    mem_train_set_y = train_set_y
    mem_test_set_x = test_set_x
    mem_test_set_y = test_set_y
                    
    train_set_x = np.asarray(mem_train_set_x)
    train_set_y = np.asarray(mem_train_set_y)
    test_set_x = np.asarray(mem_test_set_x)
    test_set_y = np.asarray(mem_test_set_y)
    
    train_set_x = np.vstack(train_set_x)
    test_set_x = np.vstack(test_set_x)
    '''
    rval = (train_set_x, test_set_x, train_set_y, test_set_y)

    return rval

def load_kaggle_segmented_data(datapath='', preprocimgspath='', normalize_x = False, normalize_y = False, input_channels = 3, input_size = [64,64], rescale = True, truePath="", falsePath="", testPath="", max_patients_train = 20, max_nodules_train = 150, max_patients_test = 20, max_nodules_test = 150):
    
    onechannel = False
    
    if input_channels == 1: 
        onechannel = True
    
    # Set files
    print('\nLoading Kaggle Datasets from the files...')
    
    noncancerpatientslist = np.asarray([f for f in listdir(datapath+"/"+falsePath) if isdir(join(datapath+"/"+falsePath, f))])
    cancerpatientslist = np.asarray([f for f in listdir(datapath+"/"+truePath) if isdir(join(datapath+"/"+truePath, f))])
    testpatientslist = np.asarray([f for f in listdir(datapath+"/"+testPath) if isdir(join(datapath+"/"+testPath, f))])
    
    train_set_xy = []
    train_set_xz = []
    train_set_yz = []
    # Reads all the non-cancer patients
    print ("\n\nReading the Non-cancer patients images", end='', flush=True)

    max_current_patients_train = noncancerpatientslist.shape[0]
    if max_current_patients_train == 0:
        print('\n\nNo patients in Non-cancer folder !')
        sys.exit(0)
    if max_current_patients_train > max_patients_train and max_patients_train > 0: 
        max_current_patients_train = max_patients_train                          # if the chosen max number of patients exceeds the folder's number of patients then choose this 
	
    for i in range(max_current_patients_train):
        print('.', end='', flush=True)
        imgfileslist = np.sort(np.asarray([f for f in listdir(datapath+"/"+falsePath+"/"+noncancerpatientslist[i]) if isfile(join(datapath+"/"+falsePath+"/"+noncancerpatientslist[i], f))]))
        
        max_nodules_number_train = imgfileslist.shape[0]
        if max_nodules_number_train > max_nodules_train and max_nodules_train > 0: 
            max_nodules_number_train = max_nodules_train

        for k in range(max_nodules_number_train):
            filename = datapath+"/"+falsePath+"/"+noncancerpatientslist[i]+"/"+imgfileslist[k]
            image = scipy.misc.imread(filename, flatten = onechannel, mode='RGB')
            if rescale == True: 
                image = scipy.misc.imresize(image, (input_size[0], input_size[1]))
            if (filename.find("_XY")>=0):
                train_set_xy.append([np.asarray(image).flatten(),0,noncancerpatientslist[i]])           # 0 is the positive label
            elif (filename.find("_XZ")>=0): 
                train_set_xz.append([np.asarray(image).flatten(),0,noncancerpatientslist[i]])
            elif (filename.find("_YZ")>=0): 
                train_set_yz.append([np.asarray(image).flatten(),0,noncancerpatientslist[i]])
            if i==0 and k<3:
                scipy.misc.imsave(preprocimgspath+"/"+imgfileslist[k]+'_non_cancer.jpg', image)  # for test

    # Reads all the cancer patients
    print ("\n\nReading the Cancer patients images", end='', flush=True)

    max_current_patients_train = cancerpatientslist.shape[0]
    if max_current_patients_train == 0:
        print('\n\nNo patients in Cancer folder !')
        sys.exit(0)
    if max_current_patients_train > max_patients_train and max_patients_train > 0: 
        max_current_patients_train = max_patients_train                          # if the chosen max number of patients exceeds the folder's number of patients then choose this 

    for i in range(max_current_patients_train):
        print('.', end='', flush=True)
        imgfileslist = np.sort(np.asarray([f for f in listdir(datapath+"/"+truePath+"/"+cancerpatientslist[i]) if isfile(join(datapath+"/"+truePath+"/"+cancerpatientslist[i], f))]))

        max_nodules_number_train = imgfileslist.shape[0]
        if max_nodules_number_train > max_nodules_train and max_nodules_train > 0: 
            max_nodules_number_train = max_nodules_train
            
        for k in range(max_nodules_number_train):
            filename = datapath+"/"+truePath+"/"+cancerpatientslist[i]+"/"+imgfileslist[k]
            image = scipy.misc.imread(filename, flatten = onechannel, mode='RGB')
            if rescale == True: 
                image = scipy.misc.imresize(image, (input_size[0], input_size[1]))
            if (filename.find("_XY")>=0):
                train_set_xy.append([np.asarray(image).flatten(),1,cancerpatientslist[i]])		       # 1 is the positive label
            elif (filename.find("_XZ")>=0): 
                train_set_xz.append([np.asarray(image).flatten(),1,cancerpatientslist[i]])
            elif (filename.find("_YZ")>=0): 
                train_set_yz.append([np.asarray(image).flatten(),1,cancerpatientslist[i]])
            if i==0 and k<3:
                scipy.misc.imsave(preprocimgspath+"/"+imgfileslist[k]+'_cancer.jpg', image)  # for test

	# Converts training set into an array   
    train_set_xy = np.asarray(train_set_xy)
    train_set_xz = np.asarray(train_set_xz)
    train_set_yz = np.asarray(train_set_yz)

    test_set_xy = []
    test_set_xz = []
    test_set_yz = []
    # Reads all the test patients
    print ("\n\nReading the Test patients images", end='', flush=True)

    max_current_patients_test = testpatientslist.shape[0]
    if max_current_patients_test == 0:
        print('\n\nNo patients in Test folder !')
        sys.exit(0)
    if max_current_patients_test > max_patients_test and max_patients_test > 0: 
        max_current_patients_test = max_patients_test                          # if the chosen max number of patients exceeds the folder's number of patients then choose this 
 
    for i in range(max_current_patients_test):
        print('.', end='', flush=True)
        imgfileslist = np.sort(np.asarray([f for f in listdir(datapath+"/"+testPath+"/"+testpatientslist[i]) if isfile(join(datapath+"/"+testPath+"/"+testpatientslist[i], f))]))
    
        max_nodules_number_test = imgfileslist.shape[0]
        if max_nodules_number_test > max_nodules_test and max_nodules_test > 0: 
            max_nodules_number_test = max_nodules_test

        for k in range(max_nodules_number_test):
            filename = datapath+"/"+testPath+"/"+testpatientslist[i]+"/"+imgfileslist[k]
            image = scipy.misc.imread(filename, flatten = onechannel, mode='RGB')
            if rescale == True: 
                image = scipy.misc.imresize(image, (input_size[0], input_size[1]))
            if (filename.find("_XY")>=0):
                test_set_xy.append([np.asarray(image).flatten(),0,testpatientslist[i]])		       # 0 is the fake label, it does not exist for the test set
            elif (filename.find("_XZ")>=0): 
                test_set_xz.append([np.asarray(image).flatten(),0,testpatientslist[i]])
            elif (filename.find("_YZ")>=0): 
                test_set_yz.append([np.asarray(image).flatten(),0,testpatientslist[i]])
            if i==0 and k<3:
                scipy.misc.imsave(preprocimgspath+"/"+imgfileslist[k]+'_test.jpg', image)  # for test

    # Converts test set into an array   
    test_set_xy = np.asarray(test_set_xy)
    test_set_xz = np.asarray(test_set_xz)
    test_set_yz = np.asarray(test_set_yz)
	
    # General Shuffle
    np.random.shuffle(train_set_xy)
    np.random.shuffle(train_set_xz)
    np.random.shuffle(train_set_yz)
    np.random.shuffle(test_set_xy)
    np.random.shuffle(test_set_xz)
    np.random.shuffle(test_set_yz)

    # Extracts attributes and labels from training set 
    train_set_xy_x = train_set_xy[:,0]
    train_set_xy_y = train_set_xy[:,1].astype("int")                                                                                  # extracts the target xy           
    train_set_xy_patients_ids = train_set_xy[:,2]
    
    train_set_xz_x = train_set_xz[:,0]
    train_set_xz_y = train_set_xz[:,1].astype("int")                                                                                  # extracts the target xz          
    train_set_xz_patients_ids = train_set_xz[:,2]

    train_set_yz_x = train_set_yz[:,0]
    train_set_yz_y = train_set_yz[:,1].astype("int")                                                                                  # extracts the target yz          
    train_set_yz_patients_ids = train_set_yz[:,2]

    # Extracts attributes and labels from test set 
    test_set_xy_x = test_set_xy[:,0]
    test_set_xy_y = test_set_xy[:,1].astype("int")                                                                                    # NOTE! there are no targets provided in test set !         
    test_set_xy_patients_ids = test_set_xy[:,2]
    
    test_set_xz_x = test_set_xz[:,0]
    test_set_xz_y = test_set_xz[:,1].astype("int")                                                                                    # NOTE! there are no targets provided in test set !         
    test_set_xz_patients_ids = test_set_xz[:,2]
   
    test_set_yz_x = test_set_yz[:,0]
    test_set_yz_y = test_set_yz[:,1].astype("int")                                                                                    # NOTE! there are no targets provided in test set !         
    test_set_yz_patients_ids = test_set_yz[:,2]

    # Converts attributes arrays
    train_set_xy_x = np.vstack(train_set_xy_x)
    train_set_xz_x = np.vstack(train_set_xz_x)
    train_set_yz_x = np.vstack(train_set_yz_x)
    test_set_xy_x = np.vstack(test_set_xy_x)
    test_set_xz_x = np.vstack(test_set_xz_x)
    test_set_yz_x = np.vstack(test_set_yz_x)
    train_set_xy_patients_ids = np.vstack(train_set_xy_patients_ids)
    train_set_xz_patients_ids = np.vstack(train_set_xz_patients_ids)
    train_set_yz_patients_ids = np.vstack(train_set_yz_patients_ids)
    test_set_xy_patients_ids = np.vstack(test_set_xy_patients_ids)
    test_set_xz_patients_ids = np.vstack(test_set_xz_patients_ids)
    test_set_yz_patients_ids = np.vstack(test_set_yz_patients_ids)
    
    if normalize_x == True:
        train_set_xy_x = train_set_xy_x / train_set_xy_x.max()                               
        train_set_xz_x = train_set_xz_x / train_set_xz_x.max()                               
        train_set_yz_x = train_set_yz_x / train_set_yz_x.max()                               
        test_set_xy_x = test_set_xy_x / test_set_xy_x.max()                              
        test_set_xz_x = test_set_xz_x / test_set_xz_x.max()                              
        test_set_yz_x = test_set_yz_x / test_set_yz_x.max()                              
        
    if normalize_y == True:
        train_set_xy_y = train_set_xy_y / train_set_xy.max()
        train_set_xz_y = train_set_xz_y / train_set_xz.max()
        train_set_yz_y = train_set_yz_y / train_set_yz.max()
        test_set_xy_y = test_set_xy_y / test_set_xy_y.max()                              
        test_set_xz_y = test_set_xz_y / test_set_xz_y.max()                              
        test_set_yz_y = test_set_yz_y / test_set_yz_y.max()                              
    
    rval = [(train_set_xy_x, train_set_xy_y, train_set_xy_patients_ids),
            (train_set_xz_x, train_set_xz_y, train_set_xz_patients_ids),	
            (train_set_yz_x, train_set_yz_y, train_set_yz_patients_ids),	
            (test_set_xy_x, test_set_xy_y, test_set_xy_patients_ids),
            (test_set_xz_x, test_set_xz_y, test_set_xz_patients_ids),	
            (test_set_yz_x, test_set_yz_y, test_set_yz_patients_ids)]
    
    return rval    