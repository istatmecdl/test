'''
Created on Feb 2, 2017

@author: fabrizio
'''

class CancerLabel(object):
    '''
    classdocs
    '''


    def __init__(self, FILE_LABEL_PATH):
        '''
        Constructor
        '''
        filename=FILE_LABEL_PATH
        self.dict={row.replace('\n','').split(',')[0]:row.replace('\n','').split(',')[1]  for row in open(filename,'r')}

 
    def get_label(self,label):
        try:
            return self.dict[label]
        except:
            print ("##### WARNING ##### Patient ID not found: ",label)
            return 'NO'
