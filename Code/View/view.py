'''
Created on 22/03/2017

@author: Francesco Pugliese
'''

import warnings
import os
import pdb

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
from Metrics.metrics import Metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas  as pd
import matplotlib.pyplot as plt


class Viewclass():
    def print_sizes(mode, batch_size, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y): 
        print ('\n\nBatch size: %i' % batch_size)
        print

        n_train_batches = train_set_x.shape[0] // batch_size
        print ('Number of training batches: %i' % n_train_batches)
        #pdb.set_trace()
        print ('\nTraining set values size (X): %i x %i' % (train_set_x.shape[0], train_set_x.shape[1]))
        print ('Training set target vector size (Y): %i x 1' % train_set_y.shape[0])
        print ('Sum of train set values (X): %.2f' % train_set_x.sum());
        print ('Sum of train set target (Y): %i' % train_set_y.sum());

        if mode == 0:
            n_val_batches = val_set_x.shape[0] // batch_size
            print ('\n\nNumber of test batches: %i' % n_val_batches)
            print ('\nValidation set values size (X): %i x %i' % (val_set_x.shape[0], val_set_x.shape[1]))
            print ('Validation set target vector size (Y): %i x 1' % val_set_y.shape[0])
            print ('Sum of Validation set values (X): %.2f' % val_set_x.sum());
            print ('Sum of Validation set target (Y): %i' % val_set_y.sum());

            n_test_batches = test_set_x.shape[0] // batch_size
            print ('\n\nNumber of test batches: %i' % n_test_batches)
            print ('\nTest set values size (X): %i x %i' % (test_set_x.shape[0], test_set_x.shape[1]))
            print ('Test set target vector size (Y): %i x 1' % test_set_y.shape[0])
            print ('Sum of test set values (X): %.2f' % test_set_x.sum());
            print ('Sum of test set target (Y): %i' % test_set_y.sum());

    def printConfusionMatrix(prediction_output): 
        print ('\nPredicted output: ') 
        print (prediction_output[0]) 
        print ('\nReal output: ') 
        print (prediction_output[1]) 

        pred_y = prediction_output[0]
        print ('\nSum of Predicted values: %i' % pred_y.sum())
        print
        real_y = prediction_output[1]
        print ('Sum of Real values: %i' % real_y.sum())
        print

        [TN, FN, TP, FP, precision, recall, accuracy, f1_score, TPTN_rates] = Metrics.confusionMatrix(real_y, pred_y)

        print ('\nConfusion Matrix...\n')
        print ('\nNumber of True Negatives %i' % TN)
        print ('Number of False Negatives %i' % FN)
        print ('Number of True Positives %i' % TP)
        print ('Number of False Positives %i' % FP)
        
        print ('\nMetrics...\n')
        print ('\nPrecision  : %f' % precision)
        print ('Recall   : %f' % recall)
        print ('Accuracy  : %f' % accuracy)
        print ('F1 score   : %f' % f1_score)
        
        print ('True positives rate + True negatives rate : %f' % TPTN_rates)
        
    '''ELEONORA'''
    
    def plot_hist(history, parameters):
        pd.DataFrame(history.history).to_csv(parameters.submission_txtfile_path + '/history.csv')                                   # save history.csv
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, parameters.epochs_number), history.history['loss'], label='train_loss', color='red', linestyle='dashed', linewidth=1)
        plt.plot(np.arange(0, parameters.epochs_number), history.history['val_loss'], label='val_loss', color='red', linestyle='solid', linewidth=1)
        plt.plot(np.arange(0, parameters.epochs_number), history.history['acc'], label='train_acc', color='blue', linestyle='dashed', linewidth=1)
        plt.plot(np.arange(0, parameters.epochs_number), history.history['val_acc'], label='val_acc', color='blue', linestyle='solid', linewidth=1)
        plt.title = ('Train acc/loss')
        plt.xlabel('Epoch', color = 'green', fontsize=18)
        plt.ylabel('Loss/Accuracy', color = 'green', fontsize=18)
        plt.legend(loc='upper right')
        plt.savefig(parameters.submission_path + '/TRAIN_Plot.jpg')
        plt.show()

    def plot_hist_test(parameters):
        history = pd.read_csv(parameters.submission_txtfile_path + '/history.csv')                                                  # read history.csv
        plt.figure()
        plt.plot(history['loss'], label='train_loss', color='y', linestyle='dashed', linewidth=1)
        plt.plot(history['val_loss'], label='val_loss', color='red', linestyle='solid', linewidth=1)
        plt.plot(history['acc'], label='train_acc', color='coral', linestyle='dashed', linewidth=1)
        plt.plot(history['val_acc'], label='val_acc', color='blue', linestyle='solid', linewidth=1)
        plt.xlabel('Epoch', color = 'green', fontsize=18)
        plt.ylabel('Loss/Accuracy', color = 'green', fontsize=18)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(parameters.submission_path + '/SAVE_CSV_TRAIN_Plot.jpg',dpi=900)
        plt.show()

    def confmat(parameters, deepnetwork, test_set_x, test_set_y):
        # Confusion matrix result
        Y_pred = deepnetwork.predict(test_set_x, verbose=2)
        y_pred = np.argmax(Y_pred, axis=1)

        for ix in range(parameters.output_size):
            print(ix, confusion_matrix(np.argmax(test_set_y,axis=1),y_pred)[ix].sum())
            
        cm = confusion_matrix(np.argmax(test_set_y,axis=1),y_pred)
        cm_normalize = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        print()
        print('Confusion matrix normalized')
        print(cm_normalize)

        # Visualizing of confusion matrix
        df_cm = pd.DataFrame(cm_normalize, range(parameters.output_size), range(parameters.output_size)) 
        plt.figure(figsize = (18,16))
        sn.set(font_scale=1.5)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={'size': 16})# font size
        plt.xlabel('Predicted label', color = 'green', fontsize=18)
        plt.ylabel('True label', color = 'green', fontsize=18)
        
        if parameters.eurosat_data == True:
            class_names = ['Annual Crop','Forest','Herbaceous Vegetation','Highway','Industrial','Pasture','Permanent Crop','Residential','River','Sea Lake']

            #class_names = ['Raccolto annuale','Foresta','Vegetazione erbacea','Autostrada','Industriale','Pascolo','Coltura permanente','Residenziale','Fiume','Lago']
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45,ha='right',fontsize=16, rotation_mode='anchor')
            plt.yticks(tick_marks, class_names,rotation=0,ha='right', fontsize=16,rotation_mode=None)
            plt.tight_layout()
            plt.savefig(parameters.submission_path  + '/TEST_Plot_confusion_matrix_normalize.jpg')
            plt.show()

    def plot_pie(colors, parameters, labels, fracs, title, save_pieplot, charts_path, pieplot_file, show, pause_time):
        label_porcent = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, fracs)]
        pie = plt.pie(fracs, shadow=False, startangle=0, colors = colors, wedgeprops = {'linewidth': 0.3, 'edgecolor': 'grey'})
        plt.title(title)
        plt.legend(pie[0], label_porcent, loc="best")
        plt.axis('equal')
        if save_pieplot == True: 
            if not os.path.exists(parameters.charts_path):
                os.makedirs(parameters.charts_path)
            plt.savefig(parameters.charts_path + '/' + pieplot_file)
        if show == True: 
            plt.show(block = False)
            plt.pause(pause_time)
            plt.close()

    def plot_bar_chart(colors, parameters, labels, fracs, title, save_pieplot, charts_path, pieplot_file, show, pause_time):
        plt.bar(labels, fracs, width = 0.8, color = colors, edgecolor='grey')
        plt.title(title)
        
        if save_pieplot == True: 
            if not os.path.exists(parameters.charts_path):
                os.makedirs(parameters.charts_path)
            plt.savefig(parameters.charts_path + '/' + parameters.plot_times_files)
        if show == True: 
            plt.show(block = False)
            plt.pause(pause_time)
            plt.close()
        
class Time: 
    # Converts time from seconds to days, hours, mins, secs, millisecs format
    def time_from_secs_to_days(self, time_in_sec):
        secs, msecs = divmod(time_in_sec, 1)
        secs = int(secs) 
        msecs = int(msecs * 1000) 
        mins = secs // 60
        secs = secs % 60
        hours = mins // 60
        mins = mins % 60
        days = hours // 24
        hours = hours % 24
        
        return [days, hours, mins, secs, msecs]

    def print_computation_time_in_days(self, time_in_sec, mode):
        days, hours, mins, secs, msecs = self.time_from_secs_to_days(time_in_sec)
        
        if mode == 1:
            print ('Global computation time:')     
            print ('Days: %i' % (days))     
            print ('Hours : %i ' % (hours))     

        print ('Minutes: %i' % (mins))     
        print ('Seconds : %i ' % (secs))     
        print ('Milli seconds : %i ' % (msecs))     
