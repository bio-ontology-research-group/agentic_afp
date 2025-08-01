from unittest import TestCase
import pandas as pd
import numpy as np
from agents import ProteinAgent

class TestProteinAgent(TestCase):

    def setUp(self):
        data_root = "data_genome"
        self.hypothesis_function = "GO:0000003"

        terms_mf = pd.read_pickle(f'{data_root}/terms_mf.pkl')['terms'].values.flatten()
        terms_cc = pd.read_pickle(f'{data_root}/terms_cc.pkl')['terms'].values.flatten()
        terms_bp = pd.read_pickle(f'{data_root}/terms_bp.pkl')['terms'].values.flatten()

        terms = sorted(set(terms_mf) | set(terms_cc) | set(terms_bp))
        terms_dict = {v: i for i, v in enumerate(terms)}
                            
        self.df = pd.read_pickle(f"{data_root}/predictions/mlp_genome_1.pkl")
        self.sequence = self.df.iloc[0].sequences

        initial_predictions = [1] * len(terms)  # Mock initial predictions
        self.agent = ProteinAgent(2, self.df, terms_dict, initial_predictions)

    def test_data_columns(self):
        column_of_interest = "preds"
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
        
    def test_get_diamond_score(self):
        diamond_score = self.agent.get_diamond_score(self.hypothesis_function)
        self.assertIsInstance(diamond_score, float)
        self.assertGreater(diamond_score, 0.0)

    def test_agent_ancestor_scores(self):
        ancestor_scores = self.agent.partial_retrieve_ancestor_scores(self.hypothesis_function)
        self.assertIsInstance(ancestor_scores, dict)
        self.assertGreater(len(ancestor_scores), 0)
        
    def test_is_in_interpro(self):
        interpro_annotations = self.agent.is_in_interpro(self.hypothesis_function)
        self.assertIsInstance(interpro_annotations, bool)

    def test_interpros(self):
        interpros = self.agent.interpros
        self.assertIsInstance(interpros, list)
        self.assertGreater(len(interpros), 0)
        self.assertTrue(all(x[0].startswith("GO") for x in interpros))
