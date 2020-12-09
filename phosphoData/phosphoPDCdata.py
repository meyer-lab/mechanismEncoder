#!/usr/local/bin/python
'''
Basic CLI to import CPTAC proteomic data
'''

import cptac
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancerType', dest='type',\
                        help='Cancer type to be collected')
    parser.add_argument('--getData',dest='get', action='store_true',\
                        default=False,help='Set flag to get all data')
    opts = parser.parse_args()

    if opts.get:
        for ds in ['brca', 'ccrcc', 'colon', 'ovarian','endometrial','luad']:
            cptac.download(dataset=ds)

    if opts.type.lower() == 'brca':
        dat = cptac.Brca()
    elif opts.type.lower() == 'ccrcc':
        dat = cptac.Ccrcc()
    elif opts.type.lower() == 'coad':
        dat = cptac.Colon()
    elif opts.type.lower() == 'ovca':
        dat = cptac.Ovarian()
    elif opts.type.lower() == 'luad':
        dat = cptac.Luad()
    elif opts.type.lower() == 'endometrial':
        dat = cptac.Endometrial()
    else:
        exit()

    df = dat.get_phosphoproteomics()
    pdf = dat.get_proteomics()
   # df.columns = [' '.join(col).strip() for col in df.columns.values]

    df.to_csv(path_or_buf="phos_file.tsv",sep='\t')
    pdf.to_csv(path_or_buf='prot_file.tsv',sep='\t')
    
if  __name__=='__main__':
    main()
