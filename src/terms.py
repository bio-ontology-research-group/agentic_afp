import pandas as pd
import click as ck
from ontology import Ontology

@ck.command()
@ck.option(
    '--data-file', '-f', default='data/train_data.pkl',
    help='Input data file with protein sequences and annotations')
def main(data_file):
    """Load data and save list of terms for MF, BP, CC."""
    # Load the data
    df = pd.read_pickle(data_file)
    
    # Load GO
    go = Ontology('data/go.obo', with_rels=True)
    # Extract unique GO terms for each category
    mf_terms = set()
    bp_terms = set()
    cc_terms = set()
    
    for annotations in df['prop_annotations']:
        for go_id in annotations:
            if not go.has_term(go_id):
                continue
            term = go.get_term(go_id)
            if term['namespace'] == 'molecular_function':
                mf_terms.add(go_id)
            elif term['namespace'] == 'biological_process':
                bp_terms.add(go_id)
            elif term['namespace'] == 'cellular_component':
                cc_terms.add(go_id)

    # Remove root terms
    mf_terms.discard('GO:0003674')  # molecular_function
    bp_terms.discard('GO:0008150')  # biological_process
    cc_terms.discard('GO:0005575')  # cellular_component
    
    # Save the terms to data files
    df = pd.DataFrame({'terms': list(mf_terms)})
    df.to_pickle('data/mf_terms.pkl')
    df = pd.DataFrame({'terms': list(bp_terms)})
    df.to_pickle('data/bp_terms.pkl')
    df = pd.DataFrame({'terms': list(cc_terms)})
    df.to_pickle('data/cc_terms.pkl')
    print(f"Saved {len(mf_terms)} MF, {len(bp_terms)} BP, {len(cc_terms)} CC terms to data files.")

if __name__ == '__main__':
    main()