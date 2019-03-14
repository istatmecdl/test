'''
Created on 22/03/2017

@author: Francesco Pugliese
'''
import numpy
import theano.tensor as T
from keras import backend as K

# Define all the type of loss functions 
class LossFunctions(object):

    def __init__(self, input):
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.input.p_y_given_x)[T.arange(y.shape[0]), y])                # Negative Log-Likelihood

    def currentMeanSquaredError(yt, yp):
        return (numpy.power((yt-yp), 2).sum()/yt.shape[0]/yt.shape[1])						  # Mean Squared Error (MSE) 
		
    def f1Score(y_true, y_pred):
        distsum=((y_pred-y_true)**2).sum(keepdims=True)   
        prodsum=((y_pred*y_true)).sum(keepdims=True)   
        loss=-1/(1+(distsum/(2*(prodsum+0.00001))))
    
        return loss
		
    def fmeasure(y_true, y_pred):																# da approssimare, non differenziabile
        # Calculates the confusione matrix from the tensors	
        
        y_pred = K.round(K.clip(y_pred,0,1))
        y_true = K.transpose(y_true)
        ytypprod = K.dot(y_true, y_pred)
        truePositives = ytypprod[1][1]
        trueNegatives = ytypprod[0][0]
        falsePositives = ytypprod[0][1]
        falseNegatives = ytypprod[1][0]
		
        f1Score=(2*truePositives)/(2*truePositives+falsePositives+falseNegatives)                 # harmonic mean of precision and sensitivity
		
        return -f1Score	

    def tptnrates(y_true, y_pred):															  # da approssimare, non differenziabile	
        # Calculates the confusione matrix from the tensors	
        
        y_pred = K.round(K.clip(y_pred,0,1))
        y_true = K.transpose(y_true)
        ytypprod = K.dot(y_true, y_pred)
        truePositives = ytypprod[1][1]
        trueNegatives = ytypprod[0][0]
        falsePositives = ytypprod[0][1]
        falseNegatives = ytypprod[1][0]

        tptnRates = truePositives/(truePositives+falseNegatives)+trueNegatives/(trueNegatives+falsePositives)				# Rate of TP/P + TN/N, which is steady according to the distribution of positives and negatives withing the input for the same model unlike f1-score and accuracy
		
        return -tptnRates				
