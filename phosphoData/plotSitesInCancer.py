'''
plots phosphosites nad proteins of interest in cancers
'''

import cptac
import argparse
from plotnine import *
import phosphoPDCdata as pdc
import pandas as pd
from matplotlib import pyplot as plt

mapk = ['RAF1', 'BRAF', 'MAP2K1', 'MAP2K2', 'MAPK3']
erk = ['HRAS', 'KRAS', 'NRAS']
rtk = ['FLT3']

def get_prots():
    protlist = mapk + erk + rtk
    return protlist

def get_data(cantype):
    '''
    Uses `cptac` PDC package to get cancer proteomics and phosphoproteomics
    '''
    dat = pdc.getDataForCancer(ctype)
    prot = dat.get_proteomics()
    phos = dat.get_phosphoproteomics()
    return prot,phos

def make_heatmap(cancerType, dataType):
    '''
    makes heatmap
    '''
    fig = plt.figure()
    prot,phos = get_data(cancerType)
    if dataType=='prot':
        data = prot
    else:
        data = phos
    prots = get_prots()
    ##now try to refactor
    df = pd.concat([data[p] for p in prots if p in data.columns]).unstack().reset_index()
    colnames = dict(data.columns)
    df.columns = ['Database_ID','Patient','LogRatio']

    fname = cancerType+'_patients_'+dataType+'cptacData.png'
    #make heatmap
