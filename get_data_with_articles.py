import click as ck
import pandas as pd
import json

def load_abstracts(pb_ids):
    abstracts = {}
    for pb_id in pb_ids:
        try:
            with open(f'data/pubmed/{pb_id}.json') as f:
                doc = json.load(f)
                abstracts[pb_id] = doc['abstract']
        except FileNotFoundError:
            print(f"File not found for PMID {pb_id}, skipping.")
            pass
    return abstracts

@ck.command()
def main():
    df = pd.read_pickle('data/test_predictions_combined.pkl')
    abstracts = {}
    with open('data/test_data.tsv') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            prot_id = parts[0]
            pb_ids = [it.strip() for it in parts[8].split(';')]
            abstracts[prot_id] = load_abstracts(pb_ids)
    print(abstracts)
    abstracts_list = []
    for i, row in df.iterrows():
        prot_id = row['proteins']
        if prot_id in abstracts:
            abstracts_list.append(abstracts[prot_id])
        else:
            abstracts_list.append({})
    df['abstracts'] = abstracts_list
    df.to_pickle('data/test_predictions_abstracts.pkl')

if __name__ == "__main__":
    main()