from unittest import TestCase
import pandas as pd
import numpy as np
from agents import ProteinCentricAgent

class TestProteinCentricAgent(TestCase):

    def setUp(self):
        data_root = "data"
        self.hypothesis_function = "GO:0000165"

        terms_mf = pd.read_pickle(f'{data_root}/mf_terms.pkl')['terms'].values.flatten()
        terms_cc = pd.read_pickle(f'{data_root}/cc_terms.pkl')['terms'].values.flatten()
        terms_bp = pd.read_pickle(f'{data_root}/bp_terms.pkl')['terms'].values.flatten()
        terms = sorted(set(terms_mf) | set(terms_cc) | set(terms_bp))
        terms_dict = {v: i for i, v in enumerate(terms)}

        self.df = pd.read_pickle(f"{data_root}/predictions_mlp_1.pkl")
        self.sequence = self.df.iloc[0].sequences

        initial_predictions = [1] * len(terms)  # Mock initial predictions
        idx = 4
        self.agent = ProteinCentricAgent(idx, self.df.iloc[idx], terms_dict)

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

    def test_is_in_interpro(self):
        interpro_annotations = self.agent.is_in_interpro(self.hypothesis_function)
        self.assertIsInstance(interpro_annotations, bool)

    def test_interpros(self):
        interpros = self.agent.data_row['interpros']
        self.assertIsInstance(interpros, list)
        self.assertGreater(len(interpros), 0)
        
    def test_interpros_as_go(self):
        interpros = self.agent.interpros
        self.assertIsInstance(interpros, list)
        self.assertGreater(len(interpros), 0)
        self.assertTrue(all(x[0].startswith("GO") for x in interpros))

    def test_get_taxon_constraints(self):
        taxon_constraints = self.agent.get_taxon_constraints()
        self.assertIsInstance(taxon_constraints, dict)
        self.assertIn("in_taxon", taxon_constraints)
        self.assertIn("never_in_taxon", taxon_constraints)
        self.assertIsInstance(taxon_constraints["in_taxon"], list)
        self.assertIsInstance(taxon_constraints["never_in_taxon"], list)

    def test_update_predictions(self):
        non_existing_go_term = "GO:0000000"
        current_predictions = self.agent.data_row['preds'].copy()
        self.agent.update_predictions(non_existing_go_term, -1)
        new_predictions = self.agent.data_row['preds'].copy()
        self.assertEqual(len(current_predictions), len(new_predictions))
        self.assertTrue(all(np.isclose(current_predictions, new_predictions)))
                
        
        
