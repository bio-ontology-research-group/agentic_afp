from src.deepgo.metrics import compute_metrics
import pandas as pd
import numpy as np
from src.ontology import Ontology
import click as ck
import sys

def main(test_filename, rows=None, row_id= None, onts = ['mf', 'bp', 'cc']):
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
        
        
        preds = test_df[f"{ont}_preds"].values
        preds = np.stack(preds, axis=0)

        if row_id is not None:
                test_df = test_df.iloc[[row_id]]
                annots = test_df['prop_annotations'].values[0]
                annots = [a for a in annots if a in terms_dict]
                print(f"Annotations for row {row_id}: {annots}")
                preds = preds[[row_id]]
        
        if rows is not None:
            test_df = test_df.iloc[:rows]
            preds = preds[:rows]
        
        compute_metrics(test_df, go, terms_dict, list(terms_dict.keys()), ont, preds, verbose=True)
        

if __name__ == "__main__":
    rows = None
    print(f"Rows: {rows}")
    row_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"Row ID: {row_id}")
    onts = ['bp']
    test_filename = "data/test_predictions_mlp.pkl"
    main(test_filename, rows=rows, onts=onts, row_id=row_id)

    refined_test_filename = "data/test_predictions_refined.pkl"
    main(refined_test_filename, rows=rows, onts=onts, row_id=row_id)

    propagated_test_filename = "data/test_predictions_refined_propagated.pkl"
    main(propagated_test_filename, rows=rows, onts=onts, row_id=row_id)
