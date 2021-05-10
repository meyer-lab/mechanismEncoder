'''
Collect proteins from model and create heatmap across CPTAC samples
'''

# TODO: I can't get this import to work, but i'd like to pull directly from the pathway files
#from pathways import pw_FLT3_MAPK_AKT_STAT as path
from phosphoData import phosphoPDCdata as pdc
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# TEMP FIX: enumerate all phosphosites of interest
path_sites = [('FLT3', 'Y843'),
              ('STAT1', 'Y701'),
              ('STAT3', 'Y704'),
              ('STAT5A', 'Y694'),
              ('STAT5B', 'Y699'),
              ('RAF1', 'S338'),
              ('BRAF', 'S447'),
              ('MAP2K1', 'S218'),
              ('MAP2K1', 'S222'),
              ('MAP2K1', 'S218S222'),
              ('MAP2K2', 'S222'),
              ('MAP2K2', 'S226'),
              ('MAP2K2', 'S222S226'),
              ('MAPK1', 'T185'),
              ('MAPK1', 'Y187'),
              ('MAPK1', 'T185Y187'),
              ('MAPK3', 'T202'),
              ('MAPK3', 'Y204'),
              ('MAPK3', 'T202Y204'),
              ('AKT1', 'T308'),
              ('AKT1', 'S473'),
              ('AKT2', 'T308'),
              ('AKT2', 'S473'),
              ('AKT3', 'T308'),
              ('AKT3', 'S473')]

alltypes = ['brca', 'ccrcc', 'colon', 'ovarian', 'endometrial', 'luad']

# collect list of all columns
dlist = []
count = 0
for cancerType in alltypes:
    dat = pdc.getDataForCancer(cancerType)
    meta = dat.get_clinical()

    res = dat.get_phosphoproteomics()
    res.columns = res.columns.droplevel(
        [e for e in res.columns.names if e not in ("Name", "Site")])

    meta = dat.get_clinical()

    old_idx = res.index.to_frame()
    old_idx.insert(0, 'SampleTypes',
                   meta.loc[old_idx.index]["Sample_Tumor_Normal"].tolist())
    res.index = pd.MultiIndex.from_frame(old_idx)
    res = pd.concat({cancerType: res}, names=["CancerType"])
#     combinedTypes = [cancerType + '-' +
#                      sample for sample in meta["Sample_Tumor_Normal"].tolist()]
    slist = []
    for site in path_sites:
        col = None
        try:
            col = res.xs(site[0], level='Name', axis=1, drop_level=False).xs(
                site[1], level='Site', axis=1, drop_level=False)
            count += 1
        except KeyError:
            print('No values for '+site[0]+', '+site[1]+' in ' + cancerType)
            col = pd.DataFrame([np.nan] * res.shape[0],
                               index=res.index, columns=pd.Index(data=[site], name=("Name", "Site")))
        if col is not None:
           slist.append(col)
    # some how had in cancer types as 3rd index
    onesites = pd.concat(slist, axis=1).transpose()
    onesites = onesites.loc[~onesites.index.duplicated(keep='first')]
    plt.figure(figsize=(10, 8))
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2, bottom=0.3)
    splot = sns.heatmap(onesites, cmap="coolwarm", center=0)
    splot.get_figure().savefig("Fig1_" + cancerType + ".png")

    dlist.append(onesites)
    # some how had in cancer types as 3rd index
print('Have found ' + str(count) + ' values')
# now create full data frame
allsites = pd.concat(dlist, axis=1)
# then make clustergram
plt.figure(figsize=(30, 10))
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.3)
aplot = sns.heatmap(allsites, cmap="coolwarm", center=0)
aplot.get_figure().savefig("Fig1_allCancerTypes.png")
