'''
Created on 22/03/2017
Modified on 17/07/2018

@author: Francesco Pugliese, Eleonora Bernasconi
'''

from keras import backend as K

# Define all the type of loss functions 
class Metrics(object):

    def confusionMatrix(y_true, y_pred):
        TN = 0
        FN = 0
        TP = 0
        FP = 0

        #Computation of true negative, fakse negative, true positive, false positive outputs
        for j in range(0, y_pred.shape[0]):
            if y_pred[j] == 0 and y_true[j]==y_pred[j]:
                TN += 1
            if y_pred[j] == 0 and y_true[j]!=y_pred[j]:
                FN += 1
            if y_pred[j] == 1 and y_true[j]==y_pred[j]:
                TP += 1
            if y_pred[j] == 1 and y_true[j]!=y_pred[j]:
                FP += 1
		
        if TP == 0 or TN == 0 or FP == 0 or FN == 0:
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            precision = 0
            recall = 0
            accuracy = 0
            f1_score = 0
            TPTN_rates = 0
        else: 
            precision = TP/(TP+FP)                              # rate of true positives over all positives, number of matches which are relevant to the user's need
            recall = float(TP/(TP+FN))                          # rate of true positives over true positives and false negatives, sensitivy is the probability that a relevant match is retrieved   
            P = TP + FN											# All the positives are all the true positives + all false negatives (which are positives) 
            N = TN + FP										    # All the negatives are all the true negatives + all false positives (which are negatives) 
            accuracy=float((TP+TN)/(TP+TN+FP+FN))
            f1_score=float((2*TP)/(2*TP+FP+FN))                 # harmonic mean of precision and sensitivity
            TPTN_rates = float(TP/P+TN/N)						# Rate of TP/P + TN/N, which is steady according to the distribution of positives and negatives withing the input for the same model unlike f1-score and accuracy
		
        return [TN, FN, TP, FP, precision, recall, accuracy, f1_score, TPTN_rates]
		
    def fmeasure(y_true, y_pred):
        # Calculates the confusione matrix from the tensors	
        
        y_pred = K.round(K.clip(y_pred,0,1))
        y_true = K.transpose(y_true)
        ytypprod = K.dot(y_true, y_pred)
        truePositives = ytypprod[1][1]
        trueNegatives = ytypprod[0][0]
        falsePositives = ytypprod[0][1]
        falseNegatives = ytypprod[1][0]
		
        '''
        truePositives = K.sum(K.equal(y_true[:,0:2], [0,1])) - K.sum(K.equal(y_pred[:,0:2], [1,0]))
        trueNegatives = K.sum(K.equal(y_true[:,0:2], [1,0])) - K.sum(K.equal(y_pred[:,0:2], [0,1]))
        falsePositives = K.sum(K.equal(y_true[:,0:2], [1,0])) - K.sum(K.equal(y_pred[:,0:2], [1,0]))
        falseNegatives = K.sum(K.equal(y_true[:,0:2], [0,1])) - K.sum(K.equal(y_pred[:,0:2], [0,1]))
        '''
		
        '''
		truePositives = K.sum(K.equal(y_true, [0,1])) - K.sum(K.equal(y_pred, [1,0]))
        trueNegatives = K.sum(K.equal(y_true, [1,0])) - K.sum(K.equal(y_pred, [0,1]))
        falsePositives = K.sum(K.equal(y_true, [1,0])) - K.sum(K.equal(y_pred,[1,0]))
        falseNegatives = K.sum(K.equal(y_true, [0,1])) - K.sum(K.equal(y_pred,[0,1]))
        '''
        f1Score=(2*truePositives)/(2*truePositives+falsePositives+falseNegatives)                 # harmonic mean of precision and sensitivity
        #accuracy=(truePositives+trueNegatives)/(truePositives+trueNegatives+falsePositives+falseNegatives)
		
        return f1Score
		
    def tptnrates(y_true, y_pred):
        # Calculates the confusione matrix from the tensors	
        
        y_pred = K.round(K.clip(y_pred,0,1))
        y_true = K.transpose(y_true)
        ytypprod = K.dot(y_true, y_pred)
        truePositives = ytypprod[1][1]
        trueNegatives = ytypprod[0][0]
        falsePositives = ytypprod[0][1]
        falseNegatives = ytypprod[1][0]

        tptnRates = truePositives/(truePositives+falseNegatives)+trueNegatives/(trueNegatives+falsePositives)				# Rate of TP/P + TN/N, which is steady according to the distribution of positives and negatives withing the input for the same model unlike f1-score and accuracy
		
        return tptnRates		