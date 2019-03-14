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

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import numpy as np

import configparser
from sys import argv

from TC_INFO import  TC_INFO


config = configparser.ConfigParser()

if len (argv)<2:
    print ("bad input value:")
    sys.exit(0)
parameterFileName=argv[1]

config.read(parameterFileName)
INPUT_DICOMFOLDER = config.get('getxyz', 'INPUT_DICOMFOLDER')
CANCER_LABEL_FOLDER = config.get('getxyz', 'CANCER_LABEL_FOLDER')
outdir = config.get('getxyz', 'OUTDIR')
directoryInputCoord = config.get('getxyz', 'DIRECTORYINPUTCOORD')
TC_INFO_FILE=config.get('getxyz', 'TC_INFO_FILE')
LunghezzaMaxNudulo=config.getint('getxyz', 'LENGTHMAXNODULE')
#RADIUSXY = config.getint('getxyz', 'RADIUSXY')
#RADIUSZ = config.getint('getxyz', 'RADIUSZ')

global TC
global CANCER_LABEL

def AnalisysIntoFolder(image3D,patientsID_OLD,patientsID,px,py,pz,radiusXY,radiusZ,patient,CANCER_FLAG):
    
    
    def load():

        slices = [dicom.read_file(patientFolder + '/' + s,force =True) for s in os.listdir(patientFolder)]
        numberOfSlices=len(slices)
        #print "numberOfSlices",numberOfSlices



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
                    #print s
                    continue
                OrdSlice.append([s,SL])
                OrdSlice.sort(key= lambda x: x[1])
                i+=1
            return OrdSlice
        OrdSlice=orderSlice(slices)
        
        Hbef=-9999
        Hset=set()
        first=True
        for s in OrdSlice[0:2]:
            H= s[1]
            DH=Hbef-H
            Hbef=H
            if first==True:
                first=False
                continue
        slices=[]
        for s in OrdSlice:
            s[0].SliceThickness = DH
            slices.append(s[0])
        return slices    
      
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
        
        #print "intercept:",set_intercept
        #print "slope:",set_slope 
        
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

    def get3planes(image3D,px,py,pz,radiusXY,radiusZ):
        '''
         
         ...#...
        '''
        
        maxZ,maxY,maxX=image3D.shape
        if py-radiusXY<0:
            py=radiusXY
        if px-radiusXY<0:
            px=radiusXY
        if pz-radiusZ<0:
            pz=radiusZ
            
        if px+radiusXY>maxX-1:
            px=maxX-1-radiusXY

        if py+radiusXY>maxY-1:
            py=maxY-1-radiusXY
            
        if pz+radiusZ>maxZ-1:
            pz=maxZ-1-radiusZ                    
        #print "range",pz-radiusZ
        
        #print "range",pz+radiusZ
        
        XY=image3D[
            pz,
            py-radiusXY:py+radiusXY,
            px-radiusXY:px+radiusXY]
        XZ=image3D[
            pz-radiusZ:pz+radiusZ,
            py,
            px-radiusXY:px+radiusXY]
        YZ=image3D[
            pz-radiusZ:pz+radiusZ,
            py-radiusXY:py+radiusXY,
            px
            ]
        
        return XY,XZ,YZ


    if (not patientsID_OLD==patientsID):
        print ("###############   NUOVO PATIENTE ###########################")
        patientFolder=INPUT_DICOMFOLDER + patientsID
    
        PatientID=patientFolder.split(str(os.sep))[-1]
        
        #print PatientID
        
        CANCER_FLAG=CANCER_LABEL.get_label(PatientID)
        
        print (CANCER_FLAG)
        
        slices=load() 
        
        patient=get_pixels_hu(slices)


    
    #print "len patient ()",len(patient)
    
    #print "patient type()",type(patient)
    
    #print "patient shape()",patient.shape

    
    XY,XZ,YZ=get3planes(patient,px,py,pz,radiusXY,radiusZ)
    tag=patientsID+'_'+str(px)+'_'+str(py)+'_'+str(pz)


    DIR_PATIENT=outdir+"/"+CANCER_FLAG+"/"+patientsID+"/"
    
    #print DIR_PATIENT
    
    try:
        os.stat(DIR_PATIENT)
    except:
        os.mkdir(DIR_PATIENT)   

    print (DIR_PATIENT+tag+'_XY'+'.png')
    plt.imsave(DIR_PATIENT+tag+'_XY'+'.png',XY, cmap=plt.cm.gray)
    plt.imsave(DIR_PATIENT+tag+'_YZ'+'.png',YZ, cmap=plt.cm.gray)
    plt.imsave(DIR_PATIENT+tag+'_XZ'+'.png',XZ, cmap=plt.cm.gray)
    

    return image3D,patientsID,patient,CANCER_FLAG
#print "INPUT_DICOMFOLDER: ",INPUT_DICOMFOLDER


CANCER_LABEL=CancerLabel(CANCER_LABEL_FOLDER)

#inputPointsFile='/home/fabrizio/KagglePoints.csv'

#cnr_0_patient_0a0c32c9e08cc2ea76a71649de56be6d_slice_30.png;791;238

def EstraiPaziente(inputPointsFile):    
    inputPoints=open(inputPointsFile,'r')
    
    def parse_fab(row):
        rowSplit=row.split(';')
        patientsID=rowSplit[0]
        px=int(rowSplit[1])
        py=int(rowSplit[2])
        pz=int(rowSplit[3])
        #radiusXY=RADIUSXY#int(rowSplit[4])
        #radiusZ=RADIUSZ#int(rowSplit[5])
        return rowSplit,patientsID,px,py,pz#,radiusXY,radiusZ
     
    def parse_sca(row):
        rowSplit=row.split(';')
        patientsLAB_ID_Z=rowSplit[0]
        patientsLAB=patientsLAB_ID_Z.split('_')[1]
        patientsID=patientsLAB_ID_Z.split('_')[3]
        patientsZ=patientsLAB_ID_Z.split('_')[5].split('.')[0]
    
        px=int(int(rowSplit[1])/2)
        py=int(int(rowSplit[2])/2)
        pz=int(patientsZ)
        #radiusXY=RADIUSXY#int(rowSplit[4])
        #radiusZ=RADIUSZ#int(rowSplit[5])        print (rowSplit,patientsID,px,py,pz,radiusXY,radiusZ)
        return rowSplit,patientsID,px,py,pz#,radiusXY,radiusZ
       
    patientsID_OLD=''
    image3D=''
    patient=''
    CANCER_FLAG=''
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     pxList=[]
#     pyList=[]
#     pzList=[]
    for row in  inputPoints:
        #rowSplit,patientsID,px,py,pz,radiusXY,radiusZ=parse_fab(row)
        rowSplit,patientsID,px,py,pz=parse_fab(row)
#         
#         
#         pxList.append(px*(0.66))
#         pyList.append(py*(0.66))
#         pzList.append(pz*2.5)
#         print (patientsID_OLD,patientsID)
#         #sys.exit(0)
        DX=TC.getX(patientsID)
        DY=TC.getY(patientsID)
        DZ=TC.getZ(patientsID)
        radiusXY=int((LunghezzaMaxNudulo/DX)/2)
        radiusZ=int((LunghezzaMaxNudulo/DZ)/2)
        print (radiusXY,radiusZ)
        image3D,patientsID_OLD,patient,CANCER_FLAG=AnalisysIntoFolder(image3D,patientsID_OLD,patientsID,px,py,pz,radiusXY,radiusZ,patient,CANCER_FLAG)
#     ax.scatter(pxList, pyList, pzList,s=250,  depthshade=True, )
#     plt.show()
TC=TC_INFO(TC_INFO_FILE)
for root, directories, files in os.walk(directoryInputCoord):
        for filename in files:
            filepath = os.path.join(root, filename)
            inputPointsFile=filepath#'/home/fabrizio/KagglePoints_2.csv'
            EstraiPaziente(inputPointsFile) 


