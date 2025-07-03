import sys
import os
import pandas as pd
import ast
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.models import gemini_model as camel_model

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool

import os
from typing import List, Tuple

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


DATA_ROOT = 'data'

class GOTerm():
    def __init__(self, go_id, info, predicted_score, diamond_score):
        self.go_id = go_id
        self.info = info
        self.predicted_score = predicted_score
        self.diamond_score = diamond_score

    def __repr__(self):
        return f"GOTerm(go_id={self.go_id}, info={self.info}, predicted_score={self.predicted_score}, diamond_score={self.diamond_score})"

    def __str__(self):
        return f"{self.go_id} ({self.info}) - Predicted Score: {self.predicted_score}, Diamond Score: {self.diamond_score}"

        
class ProteinCentricAgent(ChatAgent):
    def __init__(self, idx, ont, ontology, data_row, terms_dict, *args, **kwargs):
        if not isinstance(data_row, pd.Series):
            raise ValueError(f"data_row must be a pandas Series object. Got {type(data_row)} instead.")

        self.idx = idx
        self.ont = ont
        self.go = ontology
        self.data_row = data_row
        self.interpro_to_go = pd.read_csv(f"{DATA_ROOT}/interpro2go.tsv", sep='\t')

        self.terms_dict = terms_dict
        
        self.sequence = self.data_row['sequences']
        self.interpros = self.get_interpro_annotations()
        
        # diamond_tool = FunctionTool(self.get_diamond_score)
        interpro_tool = FunctionTool(self.get_interpro_annotations)
        # score_query_tool = FunctionTool(self.query_score)
        # uniprot_tool = FunctionTool(self.get_uniprot_information)
        update_tool = FunctionTool(self.update_predictions)
        taxon_constraints_tool = FunctionTool(self.get_taxon_constraints)
        get_go_term_info_tool = FunctionTool(self.get_go_term_info)
        
        if ont == 'mf':
            long_ont = 'molecular function'
        elif ont == 'bp':
            long_ont = 'biological process'
        elif ont == 'cc':
            long_ont = 'cellular component'
        
        context = f"""You are a GO annotation curator that refines GO
term predictions for the {long_ont} subontology. You operate by
revising external information of a protein sequence such as the
interpro annotations or diamon score similarity. You operate in this
way: you are given a term and you need to check (1) if the term is in
the interpro annotations or if the definition is related to the
definition of interpro annotations, (2) the diamond score for the
term. You will be asked to increase or decrease the score of the term
based on the information you have access to.  """
        
        super().__init__(*args, system_message=context, tools=[interpro_tool, taxon_constraints_tool, update_tool, get_go_term_info_tool], model=camel_model, **kwargs)

    def get_interpro_annotations(self) -> list:
        """
        Retrieve InterPro annotations for a given sequence.
        Args:
            sequence (str): The protein sequence to analyze.
        Returns:
            list: A list of GOTerm objects representing the InterPro annotations.
        """
        # return []
        interpros = self.data_row['interpros']
        gos = []
        for interpro in interpros:
            if interpro not in self.interpro_to_go['interpro_id'].values:
                continue
            go_set = self.interpro_to_go[self.interpro_to_go['interpro_id'] == interpro]['go_id'].values
            gos.extend(go_set)

        gos = list(set([go for go in gos if go in self.terms_dict]))  # Ensure unique GO terms and valid ones
        go_objects = [GOTerm(go_id=go, info=self.go.get_term_info(go), predicted_score=self.query_score(go), diamond_score=self.get_diamond_score(go)) for go in gos]

        return go_objects

    def get_go_term_info(self, go_term: str) -> str:
        """
        Retrieve the information for a given GO term.
        Args:
            go_term (str): The GO term to retrieve information for.
        Returns:
            str: The information associated with the GO term.
        """
        if go_term not in self.terms_dict:
            return f"GO term {go_term} not found in terms dictionary."

        go = GOTerm(go_id=go_term, info=self.go.get_term_info(go_term), predicted_score=self.query_score(go_term), diamond_score=self.get_diamond_score(go_term))
        return str(go)
        
    
    def is_in_interpro(self, go_term: str) -> bool:
        """
        Check if a GO term is associated with any InterPro annotations.
        Args:
            go_term (str): The GO term to check.
        Returns:
            bool: True if the GO term is associated with InterPro annotations, False otherwise.
        """
        return go_term in self.interpros

    def get_diamond_score(self, go_term: str) -> float:
        """
        Retrieve the diamond score for a given sequence and hypothesis function.

        Args:
            go_term (str): The GO term to analyze.
        Returns:
            float: The diamond score for given hypothesis function. If the GO term is not found, returns None.
    """
    

        preds = self.data_row['diam_preds']

        if go_term not in preds:
            return None
        else:
            return float(preds[go_term])

    def get_uniprot_information(self) -> str:
        """
        Retrieve UniProt information for the current sequence.
        Returns:
            str: A string containing the UniProt information.
        """
        uniprot_info = self.data_row['uniprot_text']
        return uniprot_info
        
    def get_taxon_constraints(self) -> List[str]:
        """
        Retrieve taxon constraints for the current sequence.
        Returns:
            List[str]: A list of taxon constraints.
        """
        org = self.data_row['orgs']
        if not org in self.go.taxon_map:
            return {"in_taxon": [], "never_in_taxon": []}
        
        taxa = self.go.taxon_map[org]
        in_taxon = [GOTerm(go_id=go, info=self.go.get_term_info(go), predicted_score=self.query_score(go), diamond_score=self.get_diamond_score(go)) for go in taxa[0] if go in self.terms_dict]
        never_in_taxon = [GOTerm(go_id=go, info=self.go.get_term_info(go), predicted_score=self.query_score(go), diamond_score=self.get_diamond_score(go)) for go in taxa[1] if go in self.terms_dict]

        in_taxon = [str(go_obj) for go_obj in in_taxon]
        never_in_taxon = [str(go_obj) for go_obj in never_in_taxon]
        
        taxon_constraints = {"in_taxon": in_taxon, "never_in_taxon": never_in_taxon}
        # taxon_constraints = {"in_taxon": taxa[0], "never_in_taxon": taxa[1]}


        
        return taxon_constraints
        
    def query_score(self, go_term: str) -> float:
        """
        Query the initial score for a specific GO term.
        Args:
            go_term (str): The GO term to query.
        Returns:
            float: The score for the given GO term.
        """
        if go_term not in self.terms_dict:
            return None
            # return "GO term not found in terms dictionary. Ignore this term."
            # raise ValueError(f"GO term {go_term} not found in terms dictionary.")
        
        idx = self.terms_dict[go_term]
        predictions = self.data_row[f'{self.ont}_preds']
        return predictions[idx]

    def update_predictions(self, go_term: str, score: float) -> None:
        """Update the predictions dictionary with a new score for a GO term.
        Args:
            go_term (str): The GO term identifier to update.
            score (float): The new score for the GO term.
        """
        if go_term in self.terms_dict:
            go_id = self.terms_dict.get(go_term)
            self.data_row[f'{self.ont}_preds'][go_id] = score

    def get_top_terms(self):
        predictions = self.data_row[f'{self.ont}_preds']
        terms = list(self.terms_dict.keys())
        terms = self.go.get_leaf_nodes(terms)
        terms_with_scores = {go: predictions[self.terms_dict[go]] for go in terms}
        sorted_terms = sorted(terms_with_scores.items(), key=lambda item: item[1], reverse=True)
        highest_terms = sorted_terms[:10]
        return highest_terms
        
def test_diamond_agent():
    data_root = '../data'
    ont = 'mf'
    test_df = pd.read_pickle(f"{data_root}/{ont}/time_data_esm.pkl")
    sequence = test_df['sequences'].values[0]
    print(f"Testing sequence: {sequence}")

    initial_scores = {'GO:0004553': 0.28049648, 'GO:0016798': 0.50550914, 'GO:0005102': 0.14764625, 'GO:0019899': 0.11700932, 'GO:0005515': 0.6157712, 'GO:0003824': 0.13832442, 'GO:0005488': 0.791075}

    prompt = f"Sequence: {sequence}\nInitial GO term scores: {initial_scores}\nPlease refine the scores based on the sequence and GO terms. Provide only the scores, not additional text."
    agent = DiamondAgent(ont, data_root)
    response = agent.step(prompt)
    response  = process_diamond_output(response.msgs[0].content)
    print(f"Agent's interpretation: {response}")

if __name__ == "__main__":
    test_diamond_agent()


