'''
Created on Feb 2, 2017

@author: fabrizio
'''

class TC_INFO(object):
    '''
    classdocs
    '''


    def __init__(self, FILE_LABEL_PATH):
        '''
        Constructor
        '''
        filename=FILE_LABEL_PATH
        self.dictX={row.replace('\n','').split(';')[0]:row.replace('\n','').split(';')[1] \
                    for row in open(filename,'r')}
        self.dictY={row.replace('\n','').split(';')[0]:row.replace('\n','').split(';')[2] \
                    for row in open(filename,'r')}
        self.dictZ={row.replace('\n','').split(';')[0]:row.replace('\n','').split(';')[3] \
                    for row in open(filename,'r')}
    def getX(self,label):
        try:
            return float(self.dictX[label])
        except:
            print ("##### WARNING ##### X space not found ",label)
            return 0.66
    def getY(self,label):
        try:
            return float(self.dictY[label])
        except:
            print ("##### WARNING ##### X space not found ",label)
            return 0.66
    def getZ(self,label):
        try:
            return abs(float(self.dictZ[label]))
        except:
            print ("##### WARNING ##### X space not found ",label)
            return 2.5





