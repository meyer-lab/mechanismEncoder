'''
Collect time course AML phospho data from the
PTRC data repository. so far there are only two datasets
with time course treatment to my knowledge
'''


import synapseclient
import pandas as pd

def getCellLineTimeCourse(syn):
    '''
    Downloads "Cell Line Time Course Uncorrected Phosphoproteomics" table from synapse
    '''
    tabid = 'syn25618653'
    results = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={
        'Sample':'sample', 
        'Gene':'gene', 
        'Peptide':'peptide', 
        'LogFoldChange':'logRatio'})
    return tab

def getCytokineInducedDrugSensitivity(syn):
    '''
    Downloads "Cytokine-induced Drug Sensitivity Phosphoproteomics Unnormalized" table from synapse
    '''
    tabid = 'syn24389738'
    results = syn.tableQuery('select * from '+tabid)
    tab = pd.DataFrame(data = results)
    tab = tab.rename(columns={
        'Gene':'gene', 
        'Protein':'protein',
        'Peptide':'peptide', 
        'LogRatio':'logRatio',
        'CellType':'cellLine',
        'TimePoint':'timePoint',
        'Treatment':'treatment'})
    return tab

def getQuizResistance(syn):
    '''
    Downloads "Quizartinib Resistance Phosphoproteomics Unnormalized" table from synapse
    '''
    tabid = 'syn24366514'
    results = syn.tableQuery('select * from '+tabid)
    tab = pd.DataFrame(data = results)
    tab = tab.rename(columns={
        'Gene':'gene', 
        'Entry_name':'protein',
        'Peptide':'peptide', 
        'Sample':'sample',
        'value':'logRatio',
        'Ligand':'treatment'})
    return tab


def getAllData(syn):
    '''
    harmonizes data from diverse experimental setups
    to have phospho, treatment, and time data
    '''
    cell = getCellLineTimeCourse(syn)
    cyto = getCytokineInducedDrugSensitivity(syn)
    #quiz = getQuizResistance(syn)
    res = pd.concat([cell, cyto]) 
    return res

def main():
    '''
    main method
    '''
    syn = synapseclient.Synapse()
    syn.login('yashar', '1aiqlJhdqX6mtRW')
    res = getAllData(syn)
    res.to_csv('combinedPhosphoData.csv')
    #tab = synapseclient.build_table('Combined CTRP Cell Line Phospho data','syn17084058',res)
    # syn.store(tab)

if __name__=='__main__':
    main()
