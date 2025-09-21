from src.deepgo.metrics import compute_metrics
import pandas as pd
import numpy as np
from src.ontology import Ontology
import click as ck
import sys

def main(test_filename, combine=False, onts = ['mf', 'bp', 'cc']):
    test_df = pd.read_pickle(test_filename)
    
    for ont in onts:
        print(f"Evaluating {ont}...")

        go = Ontology("data/go.obo", with_rels=True)
        terms = pd.read_pickle(f"data/{ont}_terms.pkl")['terms'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        
        train_data_file = f"data/train_data.pkl"
        test_data_file = f"data/test_data.pkl"

        train_annots = pd.read_pickle(train_data_file)['prop_annotations'].values
        train_annots = list(map(lambda x: set(x), train_annots))
        # train_annots = list(map(lambda x: [y for y in x if y in terms_dict], train_annots))
        # train_annots = list(map(set, train_annots))
        test_annots = test_df['prop_annotations'].values
        # test_annots = list(map(lambda x: [y for y in x if y in terms_dict], test_annots))
        test_annots = list(map(set, test_annots))

        go.calculate_ic(train_annots + test_annots)
        
        
        # preds = test_df[f"{ont}_preds"].values
        # preds = np.stack(preds, axis=0)
        preds = []

        alpha=0.5
        for i, row in enumerate(test_df.itertuples()):
            if combine:
                diam_preds = np.zeros((len(terms),), dtype=np.float32)
                for go_id, score in test_df.iloc[i]['diam_preds'].items():
                    if go_id in terms_dict:
                        diam_preds[terms_dict[go_id]] = score

                row_preds = diam_preds * alpha + test_df.iloc[i][f"{ont}_preds"] * (1 - alpha)
            else:
                row_preds = test_df.iloc[i][f"{ont}_preds"]
            preds.append(row_preds)

        preds = np.stack(preds, axis=0)
            
        fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(test_df, go, terms_dict, list(terms_dict.keys()), ont, preds, verbose=False)

        string = f"{fmax:.3f}\t{smin:.3f}\t{aupr:.3f}\t{avg_auc:.3f}"
        print(string)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_all.py <run_number> <model_name>")
        sys.exit(0)
                                
    run_number = sys.argv[1]
    model_name = sys.argv[2]
        
    onts = ['mf', 'bp', 'cc']

    # print("Evaluating refined")
    refined_test_filename = f"data/test_predictions_refined_{model_name}_run{run_number}.pkl"
    main(refined_test_filename, onts=onts)
        
    # print("Evaluating refined propagated")
    propagated_test_filename = f"data/test_predictions_refined_{model_name}_run{run_number}_propagated.pkl"
    main(propagated_test_filename, onts=onts)
        
