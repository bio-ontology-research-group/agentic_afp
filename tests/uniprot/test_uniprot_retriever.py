from unittest import TestCase
from src.ontology import Ontology
from src.uniprot.search_uniprot import search_uniprot

class TestUniProtRetriever(TestCase):

    def test_search_uniprot(self):
        example_sequence = "MPYKLKKEKEPPKVAKCTAKPSSSGKDGGGENTEEAQPQPQPQPQPQAQSQPPSSNKRPSNSTPPPTQLSKIKYSGGPQIVKKERRQSSSRFNLSKNRELQKLPALKDSPTQEREELFIQKLRQCCVLFDFVSDPLSDLKFKEVKRAGLNEMVEYITHSRDVVTEAIYPEAVTMFSVNLFRTLPPSSNPTGAEFDPEEDEPTLEAAWPHLQLVYEFFLRFLESPDFQPNIAKKYIDQKFVLALLDLFDSEDPRERDFLKTILHRIYGKFLGLRAYIRRQINHIFYRFIYETEHHNGIAELLEILGSIINGFALPLKEEHKMFLIRVLLPLHKVKSLSVYHPQLAYCVVQFLEKESSLTEPVIVGLLKFWPKTHSPKEVMFLNELEEILDVIEPSEFSKVMEPLFRQLAKCVSSPHFQVAERALYYWNNEYIMSLISDNAARVLPIMFPALYRNSKSHWNKTIHGLIYNALKLFMEMNQKLFDDCTQQYKAEKQKGRFRMKEREEMWQKIEELARLNPQYPMFRAPPPLPPVYSMETETPTAEDIQLLKRTVETEAVQMLKDIKKEKVLLRRKSELPQDVYTIKALEAHKRAEEFLTASQEAL"
        hypothesized_function = "GO:0110165"
            
        ont = Ontology("data/go.obo")
    
        # Run the search - no need to specify blast_db_path, it will auto-download
        results = search_uniprot(
            protein_sequence=example_sequence,
            hypothesized_go_function=hypothesized_function,
            top_n=10,
            min_identity=30.0
        )

        accession = 'Q14738'  # Example accession for testing

        all_accessions = [protein['accession'] for protein in results['similar_proteins']]
        self.assertIn('Q14738', all_accessions)

        protein_of_interest = [p for p in results['similar_proteins'] if p['accession'] == accession][0]
        self.assertTrue(protein_of_interest['has_hypothesized_function'])

