import unittest
from src.uniprot.get_go_annotations import get_protein_go_annotations

class TestUniProtGO(unittest.TestCase):
    
    def test_protein_has_specific_go_function(self):
        go_of_interest = "GO:0072542"
        go_ids = get_protein_go_annotations("Q14738")
        self.assertIn(go_of_interest, go_ids)


 
