#!/usr/bin/env python
import os
import sys
sys.path.append('.')

import click as ck
import pandas as pd
import logging

@ck.command()
@ck.option(
    '--old-data', '-o', default='data/swissprot_exp_2024_06.pkl',
    help='Old UniProt/SwissProt dataset pickle file')
@ck.option(
    '--new-data', '-n', default='data/swissprot_exp_2025_03.pkl', 
    help='New UniProt/SwissProt dataset pickle file')
@ck.option(
    '--out-file', '-f', default='data/test_data.pkl',
    help='Output pickle file containing only new proteins')
def main(old_data, new_data, out_file):
    """Generate time-based evaluation data by finding proteins present only in new dataset."""
    
    # Load datasets
    print(f'Loading old dataset from {old_data}')
    old_df = pd.read_pickle(old_data)
    
    print(f'Loading new dataset from {new_data}')
    new_df = pd.read_pickle(new_data)
    
    # Get proteins only in new dataset that have experimental annotations
    old_proteins = set(old_df['proteins'])
    diff_df = new_df[
        (~new_df['proteins'].isin(old_proteins)) & 
        (new_df['prop_annotations'].apply(len) > 0)
    ]
    
    # Save difference dataset
    print(f'Saving {len(diff_df)} new proteins with experimental annotations to {out_file}')
    diff_df.to_pickle(out_file)
    
if __name__ == '__main__':
    main()
