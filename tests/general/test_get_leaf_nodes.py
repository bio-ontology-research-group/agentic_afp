from unittest import TestCase
from src.ontology import Ontology
from src.hierarchy_retriever import retrieve_ancestor_scores
import pandas as pd

class TestGetLeafNodes(TestCase):

    def setUp(self):
        # This method will run before each test
        self.ontology = Ontology("data/go.obo", with_rels=True)
        terms_file = "data/cc_terms.pkl"
        self.terms = pd.read_pickle(terms_file).terms.values.tolist()
        self.terms_dict = {term: idx for idx, term in enumerate(self.terms)}
        self.predictions_file = "data/test_data.pkl"
        self.terms = pd.read_pickle(terms_file).terms.values.tolist()
    
    def test_get_leaf_nodes(self):
        leaf_nodes = self.ontology.get_leaf_nodes(self.terms)
        self.assertLess(len(leaf_nodes), len(self.terms))
        self.assertGreater(len(leaf_nodes), 0)


    def test_retrieve_ancestor_scores(self):
        # Assuming retrieve_ancestor_scores is a method of Ontology
        df = pd.read_pickle(self.predictions_file)
        initial_predictions = [1]*len(self.terms)  # Mock initial predictions
        term = "GO:0012505"
        ancestor_scores = retrieve_ancestor_scores(term, self.ontology, self.terms_dict, initial_predictions)
        self.assertGreater(len(ancestor_scores), 0)
                                        
