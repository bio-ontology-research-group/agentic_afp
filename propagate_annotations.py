from functools import partial
import pandas as pd
from tqdm import tqdm
from multiprocessing import get_context
from src.ontology import Ontology


def propagate_annots(preds, go, terms_dict):
    idx, preds = preds
    prop_annots = {}
    for go_id, j in terms_dict.items():
        score = preds[j]
        for sup_go in go.get_ancestors(go_id):
            if sup_go in prop_annots:
                prop_annots[sup_go] = max(prop_annots[sup_go], score)
            else:
                prop_annots[sup_go] = score
    for go_id, score in prop_annots.items():
        if go_id in terms_dict:
            preds[terms_dict[go_id]] = score
    return idx, preds


def main(data_root, model_name, ont):

    filename =f"{data_root}/test_predictions_{model_name}.pkl"
    filename = filename.replace("_.pkl", ".pkl")
    
    out_file = f"{data_root}/test_predictions_{model_name}_propagated.pkl"
    out_file = out_file.replace("_.pkl", ".pkl")
    
    test_df = pd.read_pickle(filename)
    preds = test_df[f'{ont}_preds'].tolist()
    indexed_preds = list(enumerate(preds))
    
    with get_context("spawn").Pool(50) as p:
        results = []
        with tqdm(total=len(preds)) as pbar:
            for output in p.imap_unordered(partial(propagate_annots, go=go, terms_dict=terms_dict), indexed_preds, chunksize=200):
                results.append(output)
                pbar.update()

        unordered_preds = [pred for pred in results]
        ordered_preds = sorted(unordered_preds, key=lambda x: x[0])
        preds = [pred[1] for pred in ordered_preds]

    test_df[f'{ont}_preds'] = preds
    test_df.to_pickle(out_file)

if __name__ == "__main__":
    data_root = "data"
    model_name = "refined"
    run = ""
    go = Ontology(f"{data_root}/go.obo")

    ont = 'bp'
    terms = pd.read_pickle(f'{data_root}/{ont}_terms.pkl')['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    main(data_root, model_name, ont)
    
