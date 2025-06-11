from unittest import TestCase
import pandas as pd
from agents import ProteinAgent

class TestProteinAgent(TestCase):

    def setUp(self):
        self.ont = "cc"
        data_root = "data"
        self.hypothesis_go = "GO:0012505"
        terms_file = f"{data_root}/{self.ont}/terms.pkl"
        terms = pd.read_pickle(terms_file).gos.values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}

        self.df = pd.read_pickle(f"{data_root}/{self.ont}/time_data_diam.pkl")
        self.sequence = self.df.iloc[0].sequences

        initial_predictions = [1] * len(terms)  # Mock initial predictions
        self.agent = ProteinAgent(self.ont, data_root, terms_dict, initial_predictions)

    def test_data_columns(self):
        column_of_interest = "diam_preds"
        columns = self.df.columns
        self.assertIn(column_of_interest, columns)
        
        
    def test_agent_diamond_tool(self):
        diamond_score = self.agent.partial_get_diamond_score(self.sequence, self.hypothesis_go)
        self.assertIsInstance(diamond_score, float)

    def test_agent_ancestor_scores(self):
        ancestor_scores = self.agent.partial_retrieve_ancestor_scores(self.hypothesis_go)
        self.assertIsInstance(ancestor_scores, dict)
        self.assertGreater(len(ancestor_scores), 0)
        
    def test_agent_interpro_annotations(self):
        interpro_annotations = self.agent.partial_interpro_annotations(self.sequence)
        print(f"InterPro annotations for sequence {self.sequence}: {interpro_annotations}")
        self.assertIsInstance(interpro_annotations, list)
        self.assertGreater(len(interpro_annotations), 0)
