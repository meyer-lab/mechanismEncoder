'''
Collect time course AML phospho data from the
PTRC data repository. so far there are only two datasets
with time course treatment to my knowledge
'''


import synapseclient as sc
import pandas as pd

def getBeatAMLPatientData(syn):
    tabid = 'syn22156830'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    return tab

def getCellLinePilotData(syn):
    '''
    gets some data for calibration of time course data - 30 min, 3hr, 16hr
    '''

    tabid = 'syn22255396'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={'Sample':'sample','LogFoldChange':'logRatio'})
    #we need to update the time points for this to be just minutes
    return tab


def getTramData(syn):
    '''
    Here we have time course treatment of trametinib -
    less interesting because it only has two timepoints
    '''
    tabid = 'syn22986341'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={'CellType':'cellLine',"LogRatio":'logRatio',\
                      'TimePoint':'timePoint','Treatment':'treatment'})
    return tab



def getAllData(syn):
    '''
    harmonizes data from diverse experimental setups
    to have phospho, treatment, and time data
    '''

def main():
    '''
    main method
    '''
    syn = sc.login()
    getAllData(syn)

if __name__=='__main__':
    main()
