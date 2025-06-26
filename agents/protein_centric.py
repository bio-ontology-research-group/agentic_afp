import sys
import os
import pandas as pd
import ast
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.models import deepseek_model

from src.ontology import Ontology

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


class ProteinCentricAgent(ChatAgent):
    def __init__(self, idx, ont, data_row, terms_dict, *args, **kwargs):
        if not isinstance(data_row, pd.Series):
            raise ValueError(f"data_row must be a pandas Series object. Got {type(data_row)} instead.")

        self.idx = idx
        self.go = Ontology(f"{DATA_ROOT}/go-basic.obo", with_rels=True, taxon_constraints_file=f"{DATA_ROOT}/go-computed-taxon-constraints.obo")
        self.data_row = data_row
        self.interpro_to_go = pd.read_csv(f"{DATA_ROOT}/interpro2go.tsv", sep='\t')

        self.terms_dict = terms_dict
        
        self.sequence = self.data_row['sequences']
        self.interpros = self.get_interpro_annotations()
        
        diamond_tool = FunctionTool(self.get_diamond_score)
        interpro_tool = FunctionTool(self.get_interpro_annotations)
        score_query_tool = FunctionTool(self.query_score)
        # uniprot_tool = FunctionTool(self.get_uniprot_information)
        update_tool = FunctionTool(self.update_predictions)
        taxon_constraints_tool = FunctionTool(self.get_taxon_constraints)
        
        context = f"""You are a GO annotation curator that refines GO
term predictions by revising external information of a protein
sequence such as the interpro annotations or diamon score
similarity. You operate in this way: you are given a term and you need
to check (1) if the term is in the interpro annotations or if the
definition is related to the definition of interpro annotations, (2)
the diamond score for the term. You will be asked to increase or
decrease the score of the term based on the information you have
access to.  """
        
        super().__init__(*args, system_message=context, tools=[diamond_tool, interpro_tool, score_query_tool, taxon_constraints_tool, update_tool], model=deepseek_model, **kwargs)

    def get_interpro_annotations(self) -> list:
        """
        Retrieve InterPro annotations for a given sequence.
        Args:
            sequence (str): The protein sequence to analyze.
        Returns:
            list: A list of tuples containing GO term identifiers, names, and definitions.
        """

        interpros = self.data_row['interpros']
        gos = []
        for interpro in interpros:
            if interpro not in self.interpro_to_go['interpro_id'].values:
                continue
            go_set = self.interpro_to_go[self.interpro_to_go['interpro_id'] == interpro]['go_id'].values
            gos.extend(go_set)

        gos = list(set([go for go in gos if go in self.terms_dict]))  # Ensure unique GO terms and valid ones
        gos_info = [(term, self.go.get_term_name(term), self.go.get_term_definition(term)) for term in gos]
        
        return gos_info

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

    # def get_uniprot_information(self) -> str:
        # """
        # Retrieve UniProt information for the current sequence.
        # Returns:
            # str: A string containing the UniProt information.
        # """
        # uniprot_info = self.data_row['uniprot_text']
        # return uniprot_info
        
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
        in_taxon = [self.go.get_term_info(go_id) for go_id in taxa[0] if go_id in self.terms_dict]
        never_in_taxon = [self.go.get_term_info(go_id) for go_id in taxa[1] if go_id in self.terms_dict]
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
        predictions = self.data_row['preds']
        return predictions[idx]

    def update_predictions(self, go_term: str, score: float) -> None:
        """Update the predictions dictionary with a new score for a GO term.
        Args:
            go_term (str): The GO term identifier to update.
            score (float): The new score for the GO term.
        """
        if go_term in self.terms_dict:
            go_id = self.terms_dict.get(go_term)
            self.data_row['preds'][go_id] = score


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


