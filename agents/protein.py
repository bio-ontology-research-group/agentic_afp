import sys
import os
import pandas as pd
import ast
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.models import deepseek_model
from src.hierarchy_retriever import retrieve_ancestor_scores
from src.interpro_retriever import retrieve_interpro_annotations

from src.ontology import Ontology

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool
import os
from pydantic import BaseModel
from typing import List, Tuple

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


DATA_ROOT = 'data_genome'


class ProteinAgent(ChatAgent):
    def __init__(self, idx, data, terms_dict, initial_predictions, *args, **kwargs):
        self.idx = idx
        self.go = Ontology(f"{DATA_ROOT}/go.obo", with_rels=True)
        self.data = data
        self.interpro_to_go = pd.read_csv(f"{DATA_ROOT}/interpro2go.tsv", sep='\t')

        self.terms_dict = terms_dict
        self.predictions = initial_predictions

        self.sequence = self.data['sequences'].values[idx]
        self.interpros = self.get_interpro_annotations()
        
        diamond_tool = FunctionTool(self.get_diamond_score)
        hierarchy_tool = FunctionTool(self.partial_retrieve_ancestor_scores)
        interpro_tool = FunctionTool(self.get_interpro_annotations)
        score_query_tool = FunctionTool(self.query_score)

        context = f"""You are a GO annotation curator that refines GO
term predictions by revising external information of a protein
sequence such as the interpro annotations or diamon score
similarity. You operate in this way: you are given a term and you need
to check (1) if the term is in the interpro annotations or if the
definition is related to the definition of interpro annotations, (2)
the diamond score for the term. You will be asked to increase or
decrease the score of the term based on the information you have
access to.  """
        
        super().__init__(*args, system_message=context, tools=[diamond_tool, interpro_tool, score_query_tool], model=deepseek_model, **kwargs)

            
    def partial_retrieve_ancestor_scores(self, go_term: str) -> List[Tuple[str, float]]:
        """
        Retrieve scores for ancestors of a given GO term.
        Args:
            go_term (str): The GO term for which to retrieve ancestor scores.
        Returns:
            List[Tuple[str, float]]: A list of tuples containing GO term identifiers and their scores.
        """
        return retrieve_ancestor_scores(go_term, self.go, self.terms_dict, self.predictions)

    def get_interpro_annotations(self) -> list:
        """
        Retrieve InterPro annotations for a given sequence.
        Args:
            sequence (str): The protein sequence to analyze.
        Returns:
            list: A list of tuples containing GO term identifiers, names, and definitions.
        """

        interpros = self.data[self.data['sequences'] == self.sequence]['interpros'].values[0]
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
            float: The diamond score for the given sequence and hypothesis function.
    """
    

        preds = self.data[self.data['sequences'] == self.sequence]['diam_preds'].values[0]

        if go_term not in preds:
            return 0.0
        else:
            return float(preds[go_term])

    def query_score(self, go_term: str) -> float:
        """
        Query the initial score for a specific GO term.
        Args:
            go_term (str): The GO term to query.
        Returns:
            float: The score for the given GO term.
        """
        if go_term not in self.terms_dict:
            return 0.0
            # return "GO term not found in terms dictionary. Ignore this term."
            # raise ValueError(f"GO term {go_term} not found in terms dictionary.")
        
        idx = self.terms_dict[go_term]
        return self.predictions[idx]


    def process_output(self, raw_output: str) -> dict:
        try:
            raw_output = raw_output.replace("```", "")
            raw_output = raw_output.strip()

            # Find the first occurrence of '['
            start_index = raw_output.find('[')
            if start_index == -1:
                raise ValueError(f"No '[' character found in output. Given output: {raw_output}")

            # Extract from first '[' to the end
            list_content = raw_output[start_index:]

            # Find the matching ']' - take the last one to handle nested structures
            end_index = list_content.rfind(']')
            if end_index == -1:
                raise ValueError(f"No matching ']' character found in output. Given output: {raw_output}")

            # Extract the complete list string
            list_string = list_content[:end_index + 1]

            # Parse the list
            output_list = ast.literal_eval(list_string)
            for go_id, score, explanation in output_list:
                logger.info(f"\tGO term: {go_id}, Score: {score}, Explanation: {explanation}")
                idx = self.terms_dict.get(go_id)
                self.predictions[idx] = score
            return
        except Exception as e:
            raise ValueError(f"Could not parse agent output: {e}")


            
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


