'''
Created on 22/03/2017

@author: Francesco Pugliese
'''

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
import pdb

class OneClassModel:

    @staticmethod
    def build(width, height, depth, classes, summary, weightsPath=None):
        # Activation Functions
        model = Sequential()

        #model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(depth, height, width)))
        #model.add(Dense(256, input_shape=(1022, )))
        #model.add(Activation("relu"))
        #model.add(Dropout(0.5))
        #model.add(Dense(128))
        #model.add(Activation("relu"))
        #model.add(Dropout(0.5))
        #model.add(Dense(1))
        #model.add(Activation("sigmoid")
	
        model.add(ZeroPadding2D((0,0),input_shape=(depth, height, width)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        #model.add(Activation('))

        if summary==True:
            model.summary()
        
		#if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_wights(weightsPath)
			
        return model