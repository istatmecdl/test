'''
Created on Apr 10, 2017

@author: fabrizio
'''
import os
import shutil


def muoviLeCartelleFatte(listacartellefatte,cartellaImmaginilabel,label,outFolder):
    lista=os.listdir(cartellaImmaginilabel)
    for files in lista:
            fullPathFile=cartellaImmaginilabel+os.sep+files
            #print (fullPathFile)
            if files in listacartellefatte:
                #print ("trova")
                shutil.move(fullPathFile, outFolder)
                continue
            
            #shutil.move(src, dst)
    #print len(lista) 

def ListaCartelleFatte(cartellaFatteLabel):
    #print (cartellaFatteLabel)
    s=set()
    for root, directories, files in os.walk(cartellaFatteLabel):
            #print (files)
            i=0
            for filename in files:
                s.add(filename.replace('coord_out','').replace('.csv',''))
                i+=1
            return s            

def muovi(cartellaFatteLabel,cartellaImmaginilabel,label,outFolder):
    listacartellefatte=ListaCartelleFatte(cartellaFatteLabel)
    #print listacartellefatte
    #print len(listacartellefatte)
    muoviLeCartelleFatte(listacartellefatte,cartellaImmaginilabel,label,outFolder)


label='0'
cartellaFatteLabel='/home/fabrizio/pointsKaggle/PATIENTS_357'
cartellaImmaginilabel='/home/fabrizio/imgKaggle/NO'
outFolder='/home/fabrizio/imgKaggle/done/NO'


cartellaFatteLabel='/home/deep/Kaggle_Data/pointsKaggle/PATIENTS_357'
cartellaImmaginilabel='/home/deep/Kaggle_Data/img2/0'
outFolder='/home/deep/Kaggle_Data/img2/done/0'


muovi(cartellaFatteLabel,cartellaImmaginilabel,label,outFolder)
