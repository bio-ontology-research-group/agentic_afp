import sys
import os
import pandas as pd
import ast
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
    def __init__(self, go_id, info, predicted_score, diamond_score, frequency=None):
        self.go_id = go_id
        self.info = info
        self.predicted_score = predicted_score
        self.diamond_score = diamond_score
        self.frequency = frequency
        
        
    def __repr__(self):
        return f"Start of information for {self.go_id}:\n Predicted_score={self.predicted_score}, diamond_score={self.diamond_score}, annotation_frequency={self.frequency}, info={self.info}\nEnd of information for {self.go_id}.\n"

    def __str__(self):
        return f"Start of information for {self.go_id}:\n Predicted Score: {self.predicted_score}, diamond Score: {self.diamond_score}. Annotation Frequency: {self.frequency if self.frequency is not None else 'N/A'}, {self.info}\nEnd of information for {self.go_id}.\n"

        
class ProteinCentricAgent(ChatAgent):
    def __init__(self, idx, ont, ontology, data_row, terms_dict, term_frequency, *args, **kwargs):
        if not isinstance(data_row, pd.Series):
            raise ValueError(f"data_row must be a pandas Series object. Got {type(data_row)} instead.")

        self.idx = idx
        self.ont = ont
        self.go = ontology
        self.data_row = data_row
        self.interpro_to_go = pd.read_csv(f"{DATA_ROOT}/interpro2go.tsv", sep='\t')
        
        self.terms_dict = terms_dict
        self.term_frequency = term_frequency
        
        self.sequence = self.data_row['sequences']
        self.interpros = self.get_interpro_annotations()
        
        interpro_tool = FunctionTool(self.get_interpro_annotations)
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
        
        super().__init__(*args, system_message=context,
                         tools=[interpro_tool,
                                taxon_constraints_tool,
                                update_tool,
                                get_go_term_info_tool],
                         model=camel_model,
                         **kwargs)

    def get_interpro_annotations(self) -> list:
        """
        Retrieve InterPro annotations for a given sequence.
        Args:
            sequence (str): The protein sequence to analyze.
        Returns:
            list: A list of GO ids
        """

        interpros = self.data_row['interpros']
        gos = []
        for interpro in interpros:
            if interpro not in self.interpro_to_go['interpro_id'].values:
                continue
            go_set = self.interpro_to_go[self.interpro_to_go['interpro_id'] == interpro]['go_id'].values
            gos.extend(go_set)

        return gos
        # gos = list(set([go for go in gos if go in self.terms_dict]))  # Ensure unique GO terms and valid ones
        # go_objects = [self.create_go_term(go) for go in gos]
        # return go_objects

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
        go = self.create_go_term(go_term)
        return str(go)
        
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
            dict: A dictionary containing 'in_taxon' and 'never_in_taxon' lists.
        """
        org = self.data_row['orgs']
        if not org in self.go.taxon_map:
            return {"in_taxon": [], "never_in_taxon": []}
        
        taxa = self.go.taxon_map[org]

        in_taxon = taxa[0]
        never_in_taxon = taxa[1]
        
        taxon_constraints = {"in_taxon": in_taxon, "never_in_taxon": never_in_taxon}
        return taxon_constraints

    def create_go_term(self, go_term: str) -> GOTerm:
        return GOTerm(go_id=go_term,
                      info=self.go.get_term_info(go_term),
                      predicted_score=self.query_score(go_term),
                      diamond_score=self.get_diamond_score(go_term),
                      frequency=self.term_frequency[go_term])
    
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
        
        idx = self.terms_dict[go_term]
        predictions = self.data_row[f'{self.ont}_preds']
        return predictions[idx]

    def update_predictions(self, go_term: str, score: float) -> None:
        """Update the predictions dictionary with a new score for a GO term.
        Args:
            go_term (str): The GO term identifier to update.
            score (float): The new score for the GO term.
        """
        # initial_score = self.da
        if go_term in self.terms_dict:
            go_id = self.terms_dict.get(go_term)
            self.data_row[f'{self.ont}_preds'][go_id] = score
            
