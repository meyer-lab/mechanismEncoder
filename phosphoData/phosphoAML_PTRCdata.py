'''
Collect time course AML phospho data from the
PTRC data repository. so far there are only two datasets
with time course treatment to my knowledge
'''


import synapseclient
import pandas as pd

def getBeatAMLPatientData(syn, normalized = False):
    '''
    Collects uncorrected phosphoproteomics data or normalized
    '''
    if normalized:
        tabid = 'syn22156830'
        soraf_tabid = 'syn22314122'
        comb_tabid = 'syn22156814'

    else:
        tabid = 'syn24227903'
        soraf_tabid = 'syn24228075'
        comb_tabid = 'syn24240355'
    #get original data
    tab = syn.tableQuery("select Sample,Gene,site,Peptide,LogFoldChange from "+tabid).asDataFrame()
    #get sorafenib data
    stab = syn.tableQuery("select 'AML sample' as Sample,Gene,site,Peptide,LogFoldChange from "+soraf_tabid+" where Treatment='Vehicle' and \"Cell number\">5000000").asDataFrame()

    #get combo data
    ctab = syn.tableQuery("select distinct 'AML sample' as Sample,Gene,site,Peptide,value as LogFoldChange from "+comb_tabid).asDataFrame()

    ft = pd.concat([tab,stab,ctab])
    print (set(ft.Sample))
    return ft

def getCellLinePilotData(syn, normalized=False):
    '''
    gets some data for calibration of time course data - 30 min, 3hr, 16hr
    '''
    if normalized:
        tabid = 'syn22255396'
    else:
        tabid = 'syn25618653'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={'Sample':'sample', 'LogFoldChange':'logRatio'})
    #we need to update the time points for this to be just minutes
    tab = tab[["sample","Gene","site","cellLine","treatment","timePoint","logRatio"]]
    print(tab)
    return tab


def getGiltData(syn, normalized=False):
    '''
    Collect data from gilteritinib time course data treated from MOLM14 cell lines
    Here we have varying ligand treatments of the cells ahead of time
    to simulate early and late resistance
    '''
    if normalized:
        tabid ='syn24189487'
    else:
        tabid = 'syn24189487'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={'Sample':'sample', \
                              'CellType':'cellLine','LogRatio':'logRatio',\
                              'Time (minutes)':'timePoint'})
    tab = tab.assign(treatment=lambda x: x['Treatment']+' '+x['Ligand'])
    tab = tab[["sample","Gene","site","cellLine","treatment","timePoint","logRatio"]]
    print(tab)
    return tab

def getTramData(syn, normalized=False):
    '''
    Here we have time course treatment of trametinib -
    less interesting because it only has two timepoints
    '''
    if normalized:
        tabid = 'syn24389738'
    else:
        tabid = 'syn24389738'
    tab = syn.tableQuery('select * from '+tabid).asDataFrame()
    tab = tab.rename(columns={'CellType':'cellLine', "LogRatio":'logRatio',\
                      'TimePoint':'timePoint', 'Treatment':'treatment'})
    tab = tab[["sample","Gene","site","cellLine","treatment","timePoint","logRatio"]]
    print(tab)
    return tab



def getAllData(syn, normalized=False):
    '''
    harmonizes data from diverse experimental setups
    to have phospho, treatment, and time data
    '''
    tram = getTramData(syn, normalized)
    cl = getCellLinePilotData(syn, normalized)
    gl = getGiltData(syn, normalized)
    res = pd.concat([tram, cl,gl] ) #TODO if we're printing, get rid of dumb row index
    return res

def main():
    '''
    main method
    '''
    syn = synapseclient.Synapse()
    syn.login()
    res = getAllData(syn)
    res.to_csv('combinedPhosphoData.csv')
    #tab = synapseclient.build_table('Combined CTRP Cell Line Phospho data','syn17084058',res)
    # syn.store(tab)

    res = getBeatAMLPatientData(syn)
    res.to_csv("combinedPatientData.csv")

if __name__=='__main__':
    main()
