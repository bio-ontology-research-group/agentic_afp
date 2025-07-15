from camel.agents import ChatAgent
import pandas as pd
from src.ontology import Ontology
from agents import ProteinCentricAgent
from agents.models import gemini_model as camel_model
import math
import re
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_data(data_root, ont, model_name):
    # load terms dict
    terms = pd.read_pickle(f'{data_root}/{ont}_terms.pkl')['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
     
    # load ontology
    ontology = Ontology(f'{data_root}/go.obo', with_rels=True, taxon_constraints_file=f'{data_root}/go-computed-taxon-constraints.obo')

    # load predictions dataframe
    train_data_file = f'{data_root}/train_data.pkl'
    go_frequency = compute_frequency(train_data_file, terms_dict)
    
    preds_data_file = f'{data_root}/test_predictions_{model_name}.pkl'
    test_df = pd.read_pickle(preds_data_file)

    return ontology, test_df, terms_dict, go_frequency


def compute_frequency(train_data_file, terms_dict):
    """
    Compute the frequency of each GO term in the training data.
    Args:
            train_data_file (str): Path to the training data file.
            terms_dict (dict): Dictionary mapping GO terms to indices.
    Returns:
            dict: A dictionary with GO terms as keys and their frequencies as values.
    """
    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    
    frequency = {term: 0 for term in terms_dict.keys()}

    total_annotations = len(annotations)
    for annots in annotations:
        for go_term in annots:
            if go_term in frequency:
                    frequency[go_term] += 1

    sorted_frequency = dict(sorted(frequency.items(), key=lambda x: x[1]))
    
    avg_frequency = sum(sorted_frequency.values()) / len(sorted_frequency)
    std_frequency = math.sqrt(sum((x - avg_frequency) ** 2 for x in sorted_frequency.values()) / len(sorted_frequency))
    min_frequency = min(sorted_frequency.values())
    max_frequency = max(sorted_frequency.values())
    print(f"Average frequency: {avg_frequency}, Std frequency: {std_frequency}, Min frequency: {min_frequency}, Max frequency: {max_frequency}. Total proteins: {len(annotations)}")
    
    return sorted_frequency
    

class CoordinatorProteinCentricAgent(ChatAgent):
    def __init__(self, ont, *args, **kwargs):

        self.data_root = "data"
        self.ont = ont
        model = "mlp"
        run = 1
        
        self.ontology, self.test_df, self.terms_dict, self.term_frequency = load_data(self.data_root, ont, model)

        self.less_frequent_terms = dict(list(self.term_frequency.items())[:20])
        print(f"Less frequent terms: {self.less_frequent_terms}")

        self.protein_agents = []

                                
        
        context = f"""This agent is a coordinator for protein-centric
        function prediction. However, in its current version, it does
        not use any of its features and instead it works as a wrapper
        of the ProteinCentricAgent.  In case you want to use the
        features, you need to implement rewrite this prompt and set
        the relevant tools to use """
        
        super().__init__(*args, system_message=context,
                         model=camel_model,
                         **kwargs)

    def protein_step(self, idx: int, verbose: bool = False):
        """
        Perform a step for a specific protein agent identified by idx.
        """
    
        protein_agent = self.create_protein_agent(idx)
        uniprot_information = protein_agent.get_uniprot_information()
        
        starting_prompt = f"""You are analyzing protein with the
following data: {uniprot_information}. 
Tasks:
1. Retrieve InterPro domain annotations
2. Identify taxonomic constraints
3. Report findings in structured format based on your knowledge

Output format:
- InterPro domains mapped as GO terms: [list with GO and descriptions]
- Taxon constraints: [specific taxonomic limitations]
- Plausible GO terms: [list of GO terms with explanations]
- Non-plausible GO terms: [list of GO terms with explanations]
- Key functional insights: [brief summary]
"""

        general_information = protein_agent.step(starting_prompt).msgs[0].content

        if verbose:
            print(f"\n\n\nGeneral information about protein {idx}: {general_information}")

        # Extract GO terms from general information to obtain their
        # descriptions. Before we ask the agent to extract the GO term
        # descriptions. However, the agent can struggle to run the
        # tool multiple times, so we extract the information
        # deterministically and provide it to the agent.
        
        go_terms = set(re.findall(r'\bGO:\d+', general_information))
        go_terms_info = [protein_agent.get_go_term_info(go_term) for go_term in go_terms]
        go_terms_info = "\n".join(go_terms_info)

        
        analysis_prompt = f""" You have now this information about the GO terms you discussed before {go_terms_info}.

For each relevant GO term suggested:
- Analyze annotation frequency: terms with low frequency should might be underrepresented and might be plausible. Consider a term underrepresented if its frequency is below 200
- Analyze supporting evidence (InterPro/Diamond/literature,heuristic) for each plausible term.
- If there is conflicting evidence, provide your resolution
- Provide Current score vs. recommended score. We want to minimize the amount of changes, so only update by incrementing or decrementing the score by 0.2 maximum.
- Confidence level (high/medium/low)
Output your report with all the points above in a structured format. Annotation frequency is based on the training data and is an important factor in your analysis.
"""
            
        constraint_analysis = protein_agent.step(analysis_prompt).msgs[0].content
        if verbose:
            print(f"\n\n\nConstraint analysis for protein {idx}: {constraint_analysis}")

        analysis_2_prompt = f"""Perfom a second analysis and make sure
the suggested changes are not that large. We aim to minimize the
changes, so only update by incrementing or decrementing the score by
0.2 maximum."""

        constraint_analysis_2 = protein_agent.step(analysis_2_prompt).msgs[0].content
        print(f"\n\n\nConstraint analysis 2 for protein {idx}: {constraint_analysis_2}")
        
        updating_prompt = f""" Apply your analysis to update GO term
scores. Perform the update and also provide a rationale for each
change. If no changes are needed, return 'No changes needed'.  """

        final_decision = protein_agent.step(updating_prompt).msgs[0].content
        if verbose:
            print(f"\n\n\nFinal decision for protein {idx}: {final_decision}")
        
        protein_predictions = protein_agent.data_row[f'{self.ont}_preds']
        all_predictions = self.test_df[f'{self.ont}_preds'].tolist()
        all_predictions[idx] = protein_predictions
        self.test_df[f'{self.ont}_preds'] = all_predictions
        
    def create_protein_agent(self, idx: int) -> ProteinCentricAgent:
        """
        Create a new protein agent with the specified index and initial predictions.
        Args:
            idx (int): The index of the protein agent.
        Returns:
            ProteinCentricAgent: An instance of the ProteinCentricAgent class.
        """
        row = self.test_df.iloc[idx]
        return ProteinCentricAgent(idx, self.ont, self.ontology, row, self.terms_dict, self.term_frequency)

