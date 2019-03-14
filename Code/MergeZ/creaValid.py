'''
Created on Apr 10, 2017

@author: fabrizio
'''
import os,shutil

from CancerLabel import  CancerLabel
pathlabels='/home/fabrizio/Downloads/stage1_solution.csv'
CartellaNO='/home/fabrizio/Kaggle3Dnodules/NO/'
VALID_0='/home/fabrizio/Kaggle3Dnodules/valid/0/'
VALID_1='/home/fabrizio/Kaggle3Dnodules/valid/1/'


pathlabels='/home/deep/Kaggle_Data/stage1_solution.csv'
CartellaNO='/home/deep/Kaggle_Data/Segmented_new4/NO/'
VALID_0='/home/deep/Kaggle_Data/Segmented_new4/valid/0/'
VALID_1='/home/deep/Kaggle_Data/Segmented_new4/valid/1/'


elencoCartelle=os.listdir(CartellaNO)
CANCERLABEL=CancerLabel(pathlabels)
for cartella in elencoCartelle:
    print (cartella)
    LABEL= CANCERLABEL.get_label(cartella)
    print (LABEL)
    if LABEL=='0':
        shutil.move(CartellaNO+os.sep+cartella, VALID_0)
    if LABEL=='1':
        shutil.move(CartellaNO+os.sep+cartella, VALID_1)
