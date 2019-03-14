'''
Created on Feb 1, 2017

@author: fabrizio
'''



import dicom
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pylab
from CancerLabel import  CancerLabel

#import ConfigParser

import configparser
from sys import argv

config = configparser.ConfigParser()

if len (argv)<2:
    print ("bad input value:")
    sys.exit(0)
parameterFileName=argv[1]

config.read(parameterFileName)
HOME_IMG = config.get('myPrep', 'HOME_IMG')
INPUT_FOLDER = config.get('myPrep', 'INPUT_DICOMFOLDER')
CANCER_LABEL_FOLDER = config.get('myPrep', 'CANCER_LABEL_FOLDER')
TC_INFO_FILE = config.get('myPrep', 'TC_INFO_FILE')


#HOME_IMG='/home/fabrizio/img'
#INPUT_FOLDER = '/home/fabrizio/SVILUPPO_SOFTWARE/DATI/KAGGLE/kaggleSample/'
#INPUT_FOLDER = '/media/fabrizio/2E9AC3A49AC366C5/Users/utente/Desktop/SCAMBIO/stage1/stage1/'
#CANCER_LABEL_FOLDER = '/home/fabrizio/SVILUPPO_SOFTWARE/DATI/KAGGLE/stage1_labels.csv'
#TC_INFO_FILE="/home/fabrizio/tc_info2.txt"
errorFileName=HOME_IMG+os.sep+"error.log"


print ("INPUT_FOLDER: ",INPUT_FOLDER)
patients = os.listdir(INPUT_FOLDER)

patients.sort()

NB_PATIENTS=len(patients)
print ("NUMBER OF PATIENTS:",NB_PATIENTS)

global CANCER_LABEL

CANCER_LABEL=CancerLabel(CANCER_LABEL_FOLDER)


PATIENTS_INDX=0
TC_INFO=open(TC_INFO_FILE,'w')
for PATIENTS_INDX in range(NB_PATIENTS):
    print ("choose patient number:",PATIENTS_INDX)
    patientFolder=INPUT_FOLDER + patients[PATIENTS_INDX]
    
    print (patientFolder)
    
#     if not patients[PATIENTS_INDX]=='026470d51482c93efc18b9803159c960':
#         continue
    #try:
    def AnalisysIntoFolder(patientFolder):
        PatientID=patientFolder.split(str(os.sep))[-1]
        print (PatientID)
        
        CANCER_FLAG=CANCER_LABEL.get_label(PatientID)
        print (CANCER_FLAG)
        
        def load():

            slices = [dicom.read_file(patientFolder + '/' + s,force =True) for s in os.listdir(patientFolder)]
            numberOfSlices=len(slices)
            print ("numberOfSlices",numberOfSlices)
    
    
    
            def orderSlice(slices):
                
                OrdSlice=[]
                i=0
                for s in slices:
                    
                    
                    SL=None
                    try:
                        SL= s.ImagePositionPatient[2]
                    except:
                        pass
                    try:
                        SL= s.SliceLocation
                    except:#print "NO FOUND SL #################################"
                        pass
                    try:
                        SL= s['0020','1041'].value
                    except:#print "NO FOUND SL2 #################################"
                        pass
                    if SL==None:
                        print (s)
                        continue
                    OrdSlice.append([s,SL])
                    OrdSlice.sort(key= lambda x: x[1])
                    i+=1
                return OrdSlice
            OrdSlice=orderSlice(slices)
            
            Hbef=-9999
            first=True
            for s in OrdSlice[0:2]:
                H= s[1]
                DH=Hbef-H
                Hbef=H
                if first==True:
                    first=False
                    continue
            DXDY= s[0]['0028', '0030'].value 
            rowSpace= PatientID+";"+str(DXDY[0])+";"+str(DXDY[1])+";"+str(DH)
            TC_INFO.write(rowSpace+"\n")
            #return
            slices=[]
            for s in OrdSlice:
                s[0].SliceThickness = DH
                
                slices.append(s[0])
            return slices
        slices=load() 
        #return
        def get_pixels_hu(slices):


            image = np.stack([s.pixel_array for s in slices])
            # Convert to int16 (from sometimes int16), 
            # should be possible as values should always be low enough (<32k)
            image = image.astype(np.int16)

            # Set outside-of-scan pixels to 0
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0

            

#             im= image[10][0]
#             setIm=set()
#             for i in im:
#                     setIm.add(i)
#             print setIm
            
          
            # Convert to Hounsfield units (HU)
            
            set_intercept=set()
            set_slope=set()
            
            for slice_number in range(len(slices)):
                
                intercept = slices[slice_number].RescaleIntercept
                slope = slices[slice_number].RescaleSlope
                set_intercept.add(intercept)
                set_slope.add(slope)
                if slope != 1:
                    image[slice_number] = slope * image[slice_number].astype(np.float64)
                    image[slice_number] = image[slice_number].astype(np.int16)
                    
                image[slice_number] += np.int16(intercept)
            
            print ("intercept:",set_intercept)
            print ("slope:",set_slope )
            
            #HISTOGRAM
            #plt.hist(image.flatten(), bins=80, color='c')
            #plt.show()

            #plt.hist(image.flatten(), bins=80, color='c')
            #plt.figure()    
            #plt.show()
            
                
            # just bone filter    
            #image[image < 300] = 0
            
            # just lung filter    
            
            #image [ image<-600] = 0
            #image [ image>-400] = 0
            
            #just air
            #image [ image>-900] = 0
            
            
            
            return np.array(image, dtype=np.int16)
        patient=get_pixels_hu(slices)

        #plt.hist(patient.flatten(), bins=80, color='c')
        #plt.xlabel("Hounsfield Units (HU)")
        #plt.ylabel("Frequency")
        #plt.figure()

        for i in  range(len(patient)):#range(int(len(patient)/2),len(patient)): #

            DIR_PATIENT=HOME_IMG+os.sep+str(CANCER_FLAG)+os.sep+str(PatientID)
            
            try:
                os.stat(DIR_PATIENT)
            except:
                os.mkdir(DIR_PATIENT)   
            
            outFileName=DIR_PATIENT+os.sep+'cnr_'+str(CANCER_FLAG)+'_patient_'+str(PatientID)+'_slice_'+(3-len(str(i)))*"0"+str(i)+'.png'
            print (outFileName)
            #for i in patient[i]:
            #    print i
            #plt.imsave(outFileName,patient[i], cmap=pylab.cm.Paired)# Accent)#autumn)#bone)#plt.cm.gray)
            plt.imsave(outFileName,patient[i], cmap=plt.cm.gray)
            
            #manager = plt.get_current_fig_manager()
            #manager.window.showMaximized()
            #f=plt.figure()
            #f.savefig('/home/fabrizio/img/foo'+str(i)+'.png', bbox_inches='tight')
    AnalisysIntoFolder(patientFolder)
    #try:
    #    AnalisysIntoFolder(patientFolder)
    #except Exception as e:
    #    errorFile=open(errorFileName,'a')
    #    print "patien:",patientFolder
    #    print "some loading error"
    #    print str(e)
    #    s1="patien:\t"+str(patientFolder)
    #    s2="some loading error"
    #    s3=str(e)
    #    errorFile.write(s1+"\n")
    #    errorFile.write(s2+"\n")
    #    errorFile.write(s3+"\n\n")
    #    errorFile.close()
        
TC_INFO.close()        
    #test(patientFolder)