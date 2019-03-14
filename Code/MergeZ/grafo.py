'''
Created on Apr 4, 2017

@author: fabrizio
'''

import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import networkx as nx
from CancerLabel import  CancerLabel
from TC_INFO import  TC_INFO
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


#import ConfigParser
import configparser

from sys import argv

#config = ConfigParser.RawConfigParser()
config = configparser.ConfigParser()

if len (argv)<2:
    print ("bad input value:")
    sys.exit(0)
parameterFileName=argv[1]

config.read(parameterFileName)
dirCoordMerge = config.get('graph', 'DIRCOORDMERGE')
TC_INFO_FILE=config.get('graph', 'TC_INFO_FILE')
outBariFile = config.get('graph', 'OUTBARIFILE')
DIST=config.getint('graph', 'DIST')
MAXNUMPOINTS=config.getint('graph', 'MAXNUMPOINTS')
MAXPOINTSNODULE=config.getint('graph', 'MAXPOINTSNODULE')

TC=TC_INFO(TC_INFO_FILE)




#INPUT_FOLDER = '/home/deep/Kaggle_Data/stage1/'
#CANCER_LABEL_FOLDER = '/home/deep/Kaggle_Data/stage1_labels.csv'
#outdir='/home/deep/Kaggle_Data/Segmented_new/'
#directory='/home/deep/Kaggle_Data/pointsKaggle/PATIENTS_357'
#outBariFile= '/home/deep/Kaggle_Data/grafoOut.txt'



def EstraiPaziente(inputPointsFile):    
    inputPoints=open(inputPointsFile,'r')
    
    
    
    def parse_sca(row):
        rowSplit=row.split(';')
        patientsLAB_ID_Z=rowSplit[0]
        patientsLAB=patientsLAB_ID_Z.split('_')[1]
        patientsID=patientsLAB_ID_Z.split('_')[3]
        patientsZ=patientsLAB_ID_Z.split('_')[5].split('.')[0]
    
        px=int(int(rowSplit[1])/2)
        py=int(int(rowSplit[2])/2)
        pz=int(patientsZ)
        radiusXY=20
        radiusZ=6

        return rowSplit,patientsID,px,py,pz,radiusXY,radiusZ
    
    
    patientsID_OLD=''
    image3D=''
    patient=''

    pxList=[]
    pyList=[]
    pzList=[]
    idList=[]
    id=0
    

    for row in  inputPoints:
        
        rowSplit,patientsID,px,py,pz,radiusXY,radiusZ=parse_sca(row)
        
        px=px*TC.getX(patientsID)
        py=py*TC.getY(patientsID)
        pz=pz*TC.getZ(patientsID)

        idList.append(id)
        pxList.append(px)
        pyList.append(py)
        pzList.append(pz)
        id+=1
        
    df2 = pd.DataFrame({ 'id1' : idList,'X' : pxList, 'Y' : pyList, 'Z' : pzList})
    print (len(df2))
    if len(df2)> MAXNUMPOINTS :
        df2=df2[:MAXNUMPOINTS]
        #return   
    df2['key'] = 0
    cartesian= pd.merge(df2, df2, on = 'key')
    cartesian['dist']=((cartesian['X_x']-cartesian['X_y']))**2+((cartesian['Y_x']-cartesian['Y_y']))**2\
    +((cartesian['Z_x']-cartesian['Z_y']))**2
    cartesian= cartesian[(cartesian['dist']<DIST) & (cartesian['dist']!=0) & (cartesian['id1_x']<cartesian['id1_y'])]
    df2.drop('key',1, inplace=True)
    
    '''
    crea grafo
    '''
    
    G=nx.Graph()

    for _,row in cartesian.iterrows():
        G.add_edge(row['id1_x'],row['id1_y'])
    Noduli=nx.connected_components(G)
    xList=[]
    yList=[]
    zList=[]
    x0=[]
    y0=[]
    z0=[]
    for nodulo in Noduli:

        nod= df2[df2['id1'].isin(nodulo)]   
        
        mean= nod[['X','Y','Z']].mean()
        var= nod[['X','Y','Z']].var()
        count=len(nod)
        if count <= MAXPOINTSNODULE:

            Mx= mean['X']
            My= mean['Y']
            Mz= mean['Z']

            #print mean,var,count
            #print nod['X']
            xList=xList+list(nod['X'].values)
            yList=yList+list(nod['Y'].values)
            zList=zList+list(nod['Z'].values)
            
            x0.append(Mx)
            y0.append(My)
            z0.append(Mz)
            #yList=yList+list(nod['Y'].values)
            #zList=zList+list(nod['Z'].values)

            #yList.append(nod['Y'].values)
            #zList.append(nod['Z'].values)
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
            #print xList
    #ax.scatter(xList, yList, zList,s=2,  depthshade=True, )
    
    #ax.scatter(x0, y0, z0,s=17,  depthshade=True,color='red' )
    
    

    for (x,y,z) in zip(x0,y0,z0):
        print (x,y,z)
        try:
            x=x/TC.getX(patientsID)
            y=y/TC.getY(patientsID)
            z=z/TC.getZ(patientsID)

            print (x,y,z)
            outrow= patientsID+";"+str(int(x))+";"+str(int(y))+";"+str(int(z))
            BariOut.write(outrow+"\n")
        except:
            print ("patientsID",patientsID)
            continue    
    #ax.scatter(cartesian['X_x'], cartesian['Y_x'], cartesian['Z_x'],s=2,  depthshade=True, )
    #sys.exit(0)
    #plt.show()
    
    
BariOut=open(outBariFile,'w')
for root, diectories, files in os.walk(dirCoordMerge):
        for filename in files:
            filepath = os.path.join(root, filename)
            inputPointsFile=filepath#'/home/fabrizio/KagglePoints_2.csv'
            EstraiPaziente(inputPointsFile) 
BariOut.close()
            
            
