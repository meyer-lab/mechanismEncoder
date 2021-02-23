'''
Collect proteins from model and create heatmap across CPTAC samples
'''

#TODO: I can't get this import to work, but i'd like to pull directly from the pathway files
#from pathways import pw_FLT3_MAPK_AKT_STAT as path
from phosphoData import phosphoPDCdata as pdc
import pandas as pd

#TEMP FIX: enumerate all phosphosites of interest
path_sites = [('FLT3', 'Y843'),\
              ('STAT1', 'Y701'),\
              ('STAT3', 'Y704'),\
              ('STAT5A', 'Y694'),\
              ('STAT5B', 'Y699'),\
              ('RAF1', 'S338'),\
              ('BRAF', 'S447'),\
              ('MAP2K1', 'S218_S222'),\
              ('MAP2K2', 'S222_S226'),\
              ('MAPK1', 'T185_Y187'),\
              ('MAPK3', 'T202_Y204'),
              ('AKT1', 'T308'),\
              ('AKT1', 'S473'),\
              ('AKT2', 'T308'),\
              ('AKT2', 'S473'),\
              ('AKT3', 'T308'),\
              ('AKT3', 'S473')]

alltypes = ['brca','ccrcc','colon','ovarian','endometrial','luad']

#collect list of all columns
dlist=[]
for a in alltypes:
    res = pdc.getDataForCancer(a).get_phosphoproteomics()
    for sites in path_sites:
        col = None
        try:
            col = res.xs(sites[0], level='Name', axis=1).xs(sites[1], level='Site', axis=1)
        except KeyError:
            print('No values for '+sites[0]+', '+sites[1]+' in '+a)
        if col is not None:
            dlist.append(col)
    ##some how had in cancer types as 3rd index
print('Have found '+len(dlist)+'values')
#now create full data frame

#then make clustergram
