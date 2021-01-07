#!/usr/local/bin/python
'''
Basic CLI to import CPTAC proteomic data
'''

import cptac
import argparse


def getDataForCancer(ctype):
    if ctype.lower() == 'brca':
        dat = cptac.Brca()
    elif ctype.lower() == 'ccrcc':
        dat = cptac.Ccrcc()
    elif ctype.lower() == 'coad':
        dat = cptac.Colon()
    elif ctype.lower() == 'ovca':
        dat = cptac.Ovarian()
    elif ctype.lower() == 'luad':
        dat = cptac.Luad()
    elif ctype.lower() == 'endometrial':
        dat = cptac.Endometrial()
    else:
        exit()
    return dat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancerType', dest='type',\
                        help='Cancer type to be collected')
    parser.add_argument('--getData',dest='get', action='store_true',\
                        default=False,help='Set flag to get all data')
    opts = parser.parse_args()

    if opts.get:
        for ds in ['brca', 'ccrcc', 'colon', 'ovarian']:
            cptac.download(dataset=ds)

    dat = getDataForCancer(opts.type)
    df = dat.get_phosphoproteomics()
    pdf = dat.get_proteomics()
   # df.columns = [' '.join(col).strip() for col in df.columns.values]

    df.to_csv(path_or_buf="phos_file.tsv",sep='\t')
    pdf.to_csv(path_or_buf='prot_file.tsv',sep='\t')

if  __name__=='__main__':
    main()
