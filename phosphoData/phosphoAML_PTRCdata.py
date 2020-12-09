'''
Collect time course AML phospho data from the
PTRC data repository. so far there are only two datasets
with time course treatment to my knowledge
'''


import synapseclient
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
    tab = tab.rename(columns={'Sample':'sample', 'LogFoldChange':'logRatio'})
    #we need to update the time points for this to be just minutes
    return tab


def getTramData(syn):
    '''
    Here we have time course treatment of trametinib -
    less interesting because it only has two timepoints
    '''
    tabid = 'syn22986341'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={'CellType':'cellLine', "LogRatio":'logRatio',\
                      'TimePoint':'timePoint', 'Treatment':'treatment'})
    return tab



def getAllData(syn):
    '''
    harmonizes data from diverse experimental setups
    to have phospho, treatment, and time data
    '''
    tram = getTramData(syn)
    cl = getCellLinePilotData(syn)
    res = pd.concat([tram, cl]) #TODO if we're printing, get rid of dumb row index
    return res

def main():
    '''
    main method
    '''
    syn = synapseclient.Synapse()
    syn.login()
    res = getAllData(syn)
    res.to_csv('combinedPhosphoData.csv')
    tab = synapseclient.build_table('Combined CTRP Cell Line Phospho data','syn17084058',res)
    # syn.store(tab)

if __name__=='__main__':
    main()
