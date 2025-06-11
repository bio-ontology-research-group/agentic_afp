import sys
import os
import pandas as pd
import ast
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diamond_retriever import get_diamond_score
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



OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_FREE_API_KEY")

# max_tokens = 20000
max_tokens = 140000
# max_tokens = 1000000

# model_type="google/gemini-2.0-flash-001",
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type="deepseek/deepseek-chat-v3-0324:free",
    # model_type="meta-llama/llama-3.3-8b-instruct:free",
    # model_type="google/gemini-2.0-flash-001",
    api_key=OPENROUTER_API_KEY,
    model_config_dict={"temperature": 0.3, "max_tokens": max_tokens},
)


class DiamondResponseFormat(BaseModel):
    scores: List[Tuple[str, float]]  # List of tuples (GO term, score)


class ProteinAgent(ChatAgent):
    def __init__(self, ont, data_root, terms_dict, initial_predictions, *args, **kwargs):
        self.go = Ontology(f"{data_root}/go-basic.obo", with_rels=True)
        self.df = pd.read_pickle(f"{data_root}/{ont}/time_data_diam.pkl")
        self.interpro_to_go = pd.read_csv(f"{data_root}/interpro2go.tsv", sep='\t')

        self.terms_dict = terms_dict
        self.predictions = initial_predictions


        
        diamond_tool = FunctionTool(self.partial_get_diamond_score)
        hierarchy_tool = FunctionTool(self.partial_retrieve_ancestor_scores)
        interpro_tool = FunctionTool(self.partial_interpro_annotations)
        score_query_tool = FunctionTool(self.query_score)

        
        if ont == 'mf':
            self.long_ont = 'Molecular Function'
        elif ont == 'bp':
            self.long_ont = 'Biological Process'
        elif ont == 'cc':
            self.long_ont = 'Cellular Component'
        else:
            raise ValueError(f"Unknown ontology: {ont}")


        
        context = f"""

You are a GO annotation curator that improves prediction recall by identifying missed annotations from InterPro evidence.
Core Task
Find GO terms from InterPro that weren't initially predicted, validate with Diamond similarity, and boost scores for high-confidence matches.
Process

Extract InterPro GO terms - these are high-confidence annotations
Check Diamond scores for InterPro terms (>0.7 = strong support)
Compare with initial predictions - focus on missed terms
Boost scores when InterPro + Diamond agree but initial prediction missed/underscored

Scoring Logic

InterPro + Diamond >0.7: Score 0.8-0.9
InterPro + Diamond 0.5-0.7: Score 0.6-0.8
InterPro only: Score 0.5-0.7
Conflicts: Use InterPro evidence, score 0.4-0.6

Output Format
[("GO:XXXXXXX", score, "InterPro domain IPR### + Diamond 0.XX similarity")]
Only output terms where you're changing/adding annotations - focus on recall improvement.
"""
        
        super().__init__(*args, system_message=context, tools=[diamond_tool, interpro_tool, score_query_tool], model=model, **kwargs)

    def partial_get_diamond_score(self, sequence: str, hypothesis_function: str):
        """
        Retrieve the diamond score for a given sequence and hypothesis function.
        Args:
            sequence (str): The protein sequence to analyze.
            hypothesis_function (str): The GO term representing the hypothesized function.
        Returns:
            float: The diamond score for the given sequence and hypothesis function.
        """
        return get_diamond_score(sequence, hypothesis_function, self.df)

    def partial_retrieve_ancestor_scores(self, go_term: str) -> List[Tuple[str, float]]:
        """
        Retrieve scores for ancestors of a given GO term.
        Args:
            go_term (str): The GO term for which to retrieve ancestor scores.
        Returns:
            List[Tuple[str, float]]: A list of tuples containing GO term identifiers and their scores.
        """
        return retrieve_ancestor_scores(go_term, self.go, self.terms_dict, self.predictions)

    def partial_interpro_annotations(self, sequence: str) -> list:
        """
        Retrieve InterPro annotations for a given sequence.
        Args:
            sequence (str): The protein sequence to analyze.
        Returns:
            list: A list of GO terms associated with the sequence based on InterPro annotations.
        """

        interpros = self.df[self.df['sequences'] == sequence]['interpros'].values[0]
        gos = []
        for interpro in interpros:
            if interpro not in self.interpro_to_go['interpro_id'].values:
                continue
            go_set = self.interpro_to_go[self.interpro_to_go['interpro_id'] == interpro]['go_id'].values
            gos.extend(go_set)

        gos = list(set([go for go in gos if go in self.terms_dict]))  # Ensure unique GO terms and valid ones
            
        return gos

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


