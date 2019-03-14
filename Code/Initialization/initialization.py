'''
Created on 23/07/2018
Modified on 23/07/2018

@author: Francesco Pugliese, Eleonora Bernasconi
'''

import timeit
import random
import argparse
import pdb

# Define all the type of loss functions 
class Initialization:

    def __init__(self, parameters):                                              # Class Initialization (constructor) 
        # General Initializations

        random.seed(42)                                                          # Set seed
        self.valid_set_split = parameters.valid_set_perc / 100
        self.test_set_split = parameters.test_set_perc / 100
        self.global_start_time = timeit.default_timer()
        parameters.valid_set_perc = self.valid_set_split
        parameters.test_set_perc = self.test_set_split

        # Keras Initializations 
        self.deepnetworks = []
        self.default_callbacks = []

        # Kaggle Lung Cancer Detection Initializations
        if parameters.mode == 0:
            parameters.num_projections = 1
        else: 
            if parameters.num_projections > 3:
                parameters.num_projections = 3
                
        self.parameters = parameters 

    def read_initialization(self):
        return [self.valid_set_split, self.test_set_split, self.global_start_time, self.deepnetworks, self.default_callbacks, self.parameters]
        
    def parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--conf", help="configuration file name", required = False)
        #for capsnet                                
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
        parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
        parser.add_argument('--shift_fraction', default=0.1, type=float)
        parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
        parser.add_argument('--save_dir', default='./result')
        parser.add_argument('--is_training', default=1, type=int)
        parser.add_argument('--weights', default=None)
        parser.add_argument('--lr', default=0.01, type=float)
        
        return parser
