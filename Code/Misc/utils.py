'''
Created on 26/02/2019
Modified on 26/02/2019

@author: Francesco Pugliese
'''

# Other imports
import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import cv2
import os

import numpy as np

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  
  return result

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show(block = False)
        plt.pause(3)
        plt.close()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image
    
def set_cpu_or_gpu(parameters):
    # Set CPU or GPU type
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    if parameters.gpu == False: 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = parameters.gpu_id

def parse_arguments(default_config_file):
    # Constructs the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", help="configuration file name", required = False)
    (arg1) = parser.parse_args()
    config_file = arg1.conf
    if config_file is None: 
        config_file = default_config_file              # Default Configuration File
    
    return config_file

if __name__=="__main__":
    plot_log('result/log.csv')
    




