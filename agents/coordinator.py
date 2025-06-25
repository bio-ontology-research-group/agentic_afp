from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
import numpy as np
import pandas as pd
from src.ontology import Ontology
from src.utils import FUNC_DICT, NAMESPACES
from agents import ProteinAgent, Heuristician
from agents.models import deepseek_model
from sklearn.metrics import roc_curve, auc
import math
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_data(data_root):
    terms_mf = pd.read_pickle(f'{data_root}/terms_mf.pkl')['terms'].values.flatten()
    terms_cc = pd.read_pickle(f'{data_root}/terms_cc.pkl')['terms'].values.flatten()
    terms_bp = pd.read_pickle(f'{data_root}/terms_bp.pkl')['terms'].values.flatten()

    terms = sorted(set(terms_mf) | set(terms_cc) | set(terms_bp))
    terms_dict = {v: i for i, v in enumerate(terms)}
    return terms_dict

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc
def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            if tpic + fpic > 0:
                wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf


class CoordinatorAgent(ChatAgent):
    def __init__(self, *args, **kwargs):

        self.data_root = "data_genome"
        model = "mlp_genome"
        run = 1
        test_data_file = f'{self.data_root}/predictions/{model}_{run}.pkl'
        self.test_df = pd.read_pickle(test_data_file)
        self.terms_dict = load_data(self.data_root)
        self.ontology = Ontology(f'{self.data_root}/go.obo', with_rels=True)
        self.protein_agents = []

        self.test_tool = FunctionTool(self.test)
        self.tool_create_protein_agents = FunctionTool(self.create_protein_agents)
        self.tool_query_protein_agents = FunctionTool(self.query_protein_agents)
        self.tool_update_predictions = FunctionTool(self.update_predictions)
        
        context = f"""You are a coordinator agent responsible for
        managing multiple protein agents. All the proteins are part of
        the same genome. You have access to genome-level
        contraints. One type of constraint are essential functions,
        which for a given genome, the essential function should be
        present in at least one protein. Your job is to ensure that
        all the essential functions are covered by the predictions of
        the protein agents without sacrificing protein centric
        performance."""
        super().__init__(*args, system_message=context,
                         tools=[self.tool_create_protein_agents, self.tool_query_protein_agents, self.tool_update_predictions], model=deepseek_model,
                         **kwargs)

    def create_protein_agent(self, idx: int) -> ProteinAgent:
        """
        Create a new protein agent with the specified index and initial predictions.
        Args:
            idx (int): The index of the protein agent.
            initial_predictions (np.ndarray): Initial predictions for the protein agent.
        Returns:
            ProteinAgent: An instance of the ProteinAgent class.
        """
        row = self.test_df.iloc[idx]
        predictions = row['preds']
        return ProteinAgent(idx, self.test_df, self.terms_dict, predictions)

    def create_protein_agents(self, idxs):
        """
        Create protein agents for a list of indices.
        Args:
            idxs (list): List of indices for which to create protein agents.
        """
        for idx in idxs:
            protein_agent = self.create_protein_agent(idx)
            self.protein_agents.append(protein_agent)
        

    def query_protein_agents(self, go_term):
        """
        Query all protein agents for a specific GO term.
        Args:
            go_term (str): The GO term to query.
        Returns:
            list: A list of responses from each protein agent.
        """
        prompt = f"""Please provide your response to the following GO term query: {go_term}. Analyze it usingyour knowledge and the information you have about the proteins."""
        responses = []
        for agent in self.protein_agents:
            response = agent.step(prompt).msgs[0].content
            responses.append(response)
        return responses
            
    def retrieve_predictions(self) -> np.ndarray:
        """
        Retrieve predictions from all protein agents.
        Returns:
            np.ndarray: An array of predictions from each protein agent.
        """
        predictions = np.array(self.test_df['preds'].tolist())
        return predictions

    def update_predictions(self, idx, go_term, new_score):
        """
        Update the predictions of a specific protein agent.
        Args:
            idx (int): The index of the protein agent to update.
            go_term (str): The GO term identifier to update.
            new_score (float): The new score to assign to the GO term.
        """
        # Retrieve the latest predictions from the agent
        go_id = self.terms_dict[go_term]
        predictions = self.test_df.preds.tolist()[idx]
        predictions[go_id] = new_score
        self.test_df.preds[idx] = predictions
        
        

    def essential_function_step(self, term: str):
        """
        Ensure that the essential function term is covered by at least one protein agent.
        Args:
            term (str): The essential function term to be covered.
        """
        try:
            logger.debug(f"Ensuring essential function term '{term}' is covered by at least one protein agent.")
            term_idx = self.terms_dict[term]
            term_label = self.ontology.get_term_name(term)
            term_definition = self.ontology.get_term_definition(term)

            logger.debug(f"Term index for '{term}': {term_idx}")
            predictions = self.retrieve_predictions()
            predictions = enumerate(predictions)
            # top_ten = sorted(predictions, key=lambda x: x[1][term_idx], reverse=True)[:100]
            # agents = [self.create_protein_agent(idx) for idx, _ in top_ten]

            prompt = f"""You are given the GO term {term}: {term_label} which is an
            essential function for the genome. Your task is to check if
            the score for this term can be increased based on information
            such as InterPro annotations, Diamond score and your
            knowledge. Please provide your resolution in the form of
            ['initial score', 'new score', 'explanation']. If you cannot
            increase the score, return ['initial score', 'initial_score', 'No
            change'].
            Term information:
            GO Term: {term}
            Term Label: {term_label}
            Term Definition: {term_definition}

            """

            protein_definitions = enumerate(self.test_df['uniprot_text'].tolist())
            protein_definitions = ", ".join([f'{idx}: {desc}' for idx, desc in protein_definitions])

            heuristician_prompt = f"""The protein descriptions are:
            {protein_definitions}. The function of interest is {term} with label {term_label} and
            definition {term_definition}"""

            heuristician = Heuristician()
            heuristician_output = heuristician.step(heuristician_prompt).msgs[0].content

            print(f"\n\n\nHeuristician output: {heuristician_output}")

            coordinator_prompt = f"""You are a coordinator agent. You have
            information that certain proteinns are plausible to be
            annotated with the term {term}. The information is as follows:
            '{heuristician_output}. Use the indices in the information to
            create protein agents"""

            general_response = self.step(coordinator_prompt).msgs[0].content
            print(f"\n\n\nGeneral response from coordinator: {general_response}")

            coordinator_prompt = f"""Please query the protein agents for the GO term {term} and return the responses."""
            query_response = self.step(coordinator_prompt).msgs[0].content

            print(f"\n\n\nQuery response: {query_response}")

            # summary_prompt = f"""Please summarize the responses from the protein agents. The responses are as follows: {general_response}"""
            # summary_response = self.step(summary_prompt).msgs[0].content

            # print(f"\n\n\nSummary response: {summary_response}")

            updating_prompt = f"""Please consider the responses from
            the protein agents and choose the most plausible protein
            agent id to update the score for the GO term {term}. The
            responses are as follows: {general_response}. Update the
            score for the GO term using the update_predictions
            function."""

            final_response = self.step(updating_prompt).msgs[0].content
            print(f"\n\n\nFinal response: {final_response}")

            
            
        except Exception as e:
            logger.error(f"Error in essential function step: {e}")
            return
        
    def test(self):
        """
        Test the performance of protein-centric function prediction.
        """
        
        train_data_file = f'{self.data_root}/train_data.pkl'
        valid_data_file = f'{self.data_root}/valid_data.pkl'
        
        # terms_mf = pd.read_pickle(f'{self.data_root}/terms_mf.pkl')['terms'].values.flatten()
        # terms_cc = pd.read_pickle(f'{self.data_root}/terms_cc.pkl')['terms'].values.flatten()
        # terms_bp = pd.read_pickle(f'{self.data_root}/terms_bp.pkl')['terms'].values.flatten()

        # terms = sorted(set(terms_mf) | set(terms_cc) | set(terms_bp))
        # terms_dict = {v: i for i, v in enumerate(terms)}

        go_rels = Ontology(f'{self.data_root}/go.obo', with_rels=True)

        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
        train_df = pd.concat([train_df, valid_df])
        
        annotations = train_df['prop_annotations'].values
        annotations = list(map(lambda x: set(x), annotations))
        test_annotations = self.test_df['prop_annotations'].values
        test_annotations = list(map(lambda x: set(x), test_annotations))
        go_rels.calculate_ic(annotations + test_annotations)

        ics = {}
        for i, term in enumerate(terms):
            ics[term] = go_rels.get_ic(term)

        # Combine scores for diamond and deepgo
        eval_preds = []

        for i, row in enumerate(self.test_df.itertuples()):
            row_preds = row.preds
            preds = row_preds
            eval_preds.append(preds)

        labels = np.zeros((len(self.test_df), len(terms)), dtype=np.float32)
        eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))

        for i, row in enumerate(self.test_df.itertuples()):
            for go_id in row.prop_annotations:
                if go_id in self.terms_dict:
                    labels[i, self.terms_dict[go_id]] = 1

        total_n = 0
        total_sum = 0
        for go_id, i in self.terms_dict.items():
            pos_n = np.sum(labels[:, i])
            if pos_n > 0 and pos_n < len(self.test_df):
                total_n += 1
                roc_auc  = compute_roc(labels[:, i], eval_preds[:, i])
                total_sum += roc_auc

        if total_sum == 0:
            total_n = 1
        avg_auc = total_sum / total_n

        fmax = 0.0
        prec_max = 0.0
        rec_max = 0.0
        tmax = 0.0
        wfmax = 0.0
        wtmax = 0.0
        avgic = 0.0
        precisions = []
        recalls = []
        smin = 1000000.0
        rus = []
        mis = []
        go_set = set()
        for ont in ['mf', 'cc', 'bp']:
            go_set |= go_rels.get_namespace_terms(NAMESPACES[ont])

        for ont in ['mf', 'cc', 'bp']:
            go_set.remove(FUNC_DICT[ont])

        labels = self.test_df['prop_annotations'].values
        labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
        spec_labels = self.test_df['exp_prop_annotations'].values
        spec_labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), spec_labels))
        fmax_spec_match = 0
        for t in range(0, 101):
            threshold = t / 100.0
            preds = [set() for _ in range(len(self.test_df))]
            for i in range(len(self.test_df)):
                annots = set()
                above_threshold = np.argwhere(eval_preds[i] >= threshold).flatten()
                for j in above_threshold:
                    annots.add(terms[j])

                if t == 0:
                    preds[i] = annots
                    continue
                new_annots = set()
                for go_id in annots:
                    new_annots |= go_rels.get_ancestors(go_id)
                preds[i] = new_annots

            # Filter classes
            preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
            fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(go_rels, labels, preds)
            spec_match = 0
            for i, row in enumerate(self.test_df.itertuples()):
                spec_match += len(spec_labels[i].intersection(preds[i]))
            # print(f'AVG IC {avg_ic:.3f}')
            precisions.append(prec)
            recalls.append(rec)
            # print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}, WFmax: {wf}')
            if fmax < fscore:
                fmax = fscore
                prec_max = prec
                rec_max = rec
                tmax = threshold
                avgic = avg_ic
                fmax_spec_match = spec_match
            if wfmax < wf:
                wfmax = wf
                wtmax = threshold
            if smin > s:
                smin = s
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)

                
        return {"fmax": fmax, "smin": smin, "aupr": aupr, "auc": avg_auc}






