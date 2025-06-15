from multiprocessing import get_context
from tqdm import tqdm
from functools import partial
import torch as th
import pandas as pd
import os

from src.mlp_esm import DGPROModel, propagate_annots
from agents import ProteinAgent
from src.ontology import Ontology
from src.utils import get_query_cost
# from agents import ProteinAgent

# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_FREE_API_KEY")


import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_data(data_root, ont):
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    logger.info(f'Terms {len(terms)}')
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/time_data_esm.pkl')
    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)
    
    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict):
    esm_embeddings = th.zeros((len(df), 5120), dtype=th.float32)
    sequences = []
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        esm_embeddings[i, :] = th.FloatTensor(row.esm2)
        sequences.append(row.sequences)
        if not hasattr(row, 'prop_annotations'):
            continue
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return sequences, esm_embeddings, labels

        
def propagate_all_annotations(preds, go, terms_dict):
    indexed_preds = [(i, preds[i]) for i in range(len(preds))]

    with get_context("spawn").Pool(30) as p:
        results = []
        with tqdm(total=len(preds)) as pbar:
            for output in p.imap_unordered(partial(propagate_annots, go=go, terms_dict=terms_dict), indexed_preds, chunksize=200):
                results.append(output)
                pbar.update()

        unordered_preds = [pred for pred in results]
        ordered_preds = sorted(unordered_preds, key=lambda x: x[0])
        preds = [pred[1] for pred in ordered_preds]
    return preds

def main():
    data_root = "data"
    go_file = f'{data_root}/go-basic.obo'
    go = Ontology(go_file, with_rels=True)
    ont = "cc"
    model_name = "agentic_go"
    run = 1
    device = "cuda"
    trained_model_file = f'{data_root}/{ont}/mlp_1.th'
    terms_dict, _, _, test_data, test_df = load_data(data_root, ont)
    n_terms = len(terms_dict)
    test_sequences, test_features, test_labels = test_data
    assert len(test_sequences) == len(test_features) == len(test_labels), f"{len(test_sequences)}, {len(test_features)}, {len(test_labels)}"
    logger.info("Loading model..")
    model = DGPROModel(5120, n_terms, device).to(device)
    model.load_state_dict(th.load(trained_model_file))
    initial_preds = []
    final_preds= []


    terms_go_ids = list(terms_dict.keys())
    leaf_nodes = go.get_leaf_nodes(terms_go_ids)
    leaf_nodes_to_id = {go_id: terms_dict[go_id] for go_id in leaf_nodes}

    labels = []

    
    
    model.eval()
    logger.info("Starting prediction...")


    
    with th.no_grad():
        generation_ids = []
        for seq_no, (sequence, esm_embedding, seq_labels) in tqdm(enumerate(zip(test_sequences, test_features, test_labels)), total=len(test_sequences)):

            if seq_no == 20:
                break
            
            esm_embedding = esm_embedding.to(device)
            seq_labels = seq_labels.to(device)
            logits = model(esm_embedding).cpu().numpy()
            initial_preds.append(logits)

            
            go_term_to_scores = {go_id: logits[idx] for go_id, idx in terms_dict.items()}
            go_term_to_labels = {go_id: seq_labels[idx].item() for go_id, idx in terms_dict.items()}
            go_term_to_scores = {go_id: score for go_id, score in go_term_to_scores.items() if go_id in leaf_nodes_to_id}
            go_term_to_labels = {go_id: score for go_id, score in go_term_to_labels.items() if go_id in leaf_nodes_to_id}
            # go_term_to_test = {go_id: score for go_id, score in go_term_to_scores.items() if go_id == "GO:0042151"}
            
            
                    
            # continue    

            # sys.exit(0)
            
            # print(f"Initial GO term scores {len(go_term_to_scores)}\n...\n{list(go_term_to_scores.items())}")
            # prompt = f"Sequence: {sequence}\nInitial GO term scores: {go_term_to_scores}\nPlease refine the scores based on the sequence and GO terms. Provide only the scores, not additional text."
            prompt = f"Sequence: {sequence}.\nPlease refine the scores based on the sequence. Provide only the scores, not additional text."

            agent = ProteinAgent(ont, "data", terms_dict, logits.copy())
            # final_logits = logits.copy()
            # for i,(go_id, score) in enumerate(go_term_to_scores.items()):
                # label = go_term_to_labels[go_id]
                # if label == 1:
                    # print(f"GO term: {go_id}, Score: {score}, Label: {label}")
                    # idx = terms_dict[go_id]
                    # final_logits[idx] = 1 #score
                    
            max_tries = 3
            while max_tries > 0:
                max_tries -= 1
                try:
                    agent_output = agent.step(prompt)
                    refined_logits = agent_output.msgs[0].content
                    agent.process_output(refined_logits)
                    final_logits = agent.predictions
                    gen_id = agent_output.info['id']
                    generation_ids.append(gen_id)
                    break
                except Exception as e:
                    continue
            if max_tries == 0:
                final_logits = logits.copy()
                print(f"Maximum tries exceeded. Could not parse agent output.")
            # for go_id, score, explanation in refined_logits:
                # idx = terms_dict.get(go_id)
                # final_logits[idx] = score
                # print(f"GO term: {go_id}, Score: {score}, Explanation: {explanation}")
            
            # print(f"Refined scores {final_logits}")           # 
            # sys.exit(0)
            final_preds.append(final_logits)
            agent.reset()
            

    logger.info("Prediction completed. Propagating annotations...")
    propagated_initial_preds = propagate_all_annotations(initial_preds, go, terms_dict)
    propagated_final_preds = propagate_all_annotations(final_preds, go, terms_dict)
    logger.info("Annotation propagation completed.")

    # get the dataframe with the first row only
    test_df = test_df.iloc[:seq_no].copy()
    
    test_df['initial_preds'] = propagated_initial_preds
    test_df['final_preds'] = propagated_final_preds

    out_file = f'{data_root}/{ont}/predictions_{model_name}_{run}.pkl'
    test_df.to_pickle(out_file)

    with open(f'{data_root}/{ont}/generation_ids_{model_name}_{run}.txt', 'w') as f:
        for gen_id in generation_ids:
            f.write(f"{gen_id}\n")

    
if __name__ == "__main__":
    main()
    logger.info("Done.")
