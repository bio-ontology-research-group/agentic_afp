from unittest import TestCase
import pandas as pd
import numpy as np
from agents import ProteinCentricAgent
from src.ontology import Ontology

class TestProteinCentricAgent(TestCase):

    def setUp(self):
        data_root = "data"
        self.hypothesis_function = "GO:0000165"
        self.ont = "mf"
        self.ontology = Ontology(f"{data_root}/go.obo", with_rels=True)
        terms = pd.read_pickle(f'{data_root}/{self.ont}_terms.pkl')['terms'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        mock_frequency = {term: 1 for term in terms_dict.keys()}  # Mock frequency data
        
        self.df = pd.read_pickle(f"{data_root}/test_predictions_mlp.pkl")
        self.sequence = self.df.iloc[0].sequences

        initial_predictions = [1] * len(terms)  # Mock initial predictions
        idx = 4
        self.agent = ProteinCentricAgent(idx, self.ont, self.ontology, self.df.iloc[idx], terms_dict, mock_frequency)

    def test_data_columns(self):
        column_of_interest = f"{self.ont}_preds"
        columns = self.df.columns
        self.assertIn(column_of_interest, columns)
        

    def test_diam_preds_column(self):
        column_of_interest = "diam_preds"
        columns = self.df.columns
        self.assertIn(column_of_interest, columns)
        first_diam_preds = self.df[column_of_interest].values[0]
        self.assertIsInstance(first_diam_preds, dict)

        first_score = list(first_diam_preds.items())[0][1]
        self.assertIsInstance(first_score, np.float32)

    def test_orgs_column(self):
        column_of_interest = "orgs"
        columns = self.df.columns
        self.assertIn(column_of_interest, columns)
        first_orgs = self.df[column_of_interest].values[0]
        self.assertIsInstance(first_orgs, str)
        # check if string value is numeric
        self.assertTrue(first_orgs.isnumeric())

    def test_uniprot_info_column(self):
        column_of_interest = "uniprot_text"
        columns = self.df.columns
        self.assertIn(column_of_interest, columns)
        
    def test_get_diamond_score(self):
        diamond_score = self.agent.get_diamond_score(self.hypothesis_function)
        self.assertIsInstance(diamond_score, float)
        self.assertGreater(diamond_score, 0.0)

    def test_interpros(self):
        interpros = self.agent.data_row['interpros']
        self.assertIsInstance(interpros, list)
        self.assertGreater(len(interpros), 0)
        
    def test_get_taxon_constraints(self):
        taxon_constraints = self.agent.get_taxon_constraints()
        self.assertIsInstance(taxon_constraints, dict)
        self.assertIn("in_taxon", taxon_constraints)
        self.assertIn("never_in_taxon", taxon_constraints)
        self.assertIsInstance(taxon_constraints["in_taxon"], list)
        self.assertIsInstance(taxon_constraints["never_in_taxon"], list)

    def test_update_predictions(self):
        non_existing_go_term = "GO:0000000"
        current_predictions = self.agent.data_row[f'{self.ont}_preds'].copy()
        self.agent.update_predictions(non_existing_go_term, -1)
        new_predictions = self.agent.data_row[f'{self.ont}_preds'].copy()
        self.assertEqual(len(current_predictions), len(new_predictions))
        self.assertTrue(all(np.isclose(current_predictions, new_predictions)))
                
        
        
