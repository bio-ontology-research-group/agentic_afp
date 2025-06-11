import subprocess
import os
import tempfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import pandas as pd
import sys

class InterProScanRunner:
    def __init__(self):
        """
        Initialize InterProScan runner
        """
        
    def run_interproscan(self, sequence, seq_id="sequence_1", temp_dir=None):
        """
        Run InterProScan on a single sequence
        
        Args:
            sequence: Protein sequence string
            seq_id: Identifier for the sequence
            temp_dir: Temporary directory for files
            
        Returns:
            Path to XML output file
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        temp_dir_path = Path(temp_dir)
        
        # Create temporary FASTA file
        input_file = temp_dir_path / "input.fasta"
        with open(input_file, 'w') as f:
            f.write(f">{seq_id}\n{sequence}\n")
        
        # Set output file
        output_file = temp_dir_path / "interproscan_results.xml"
        
        # Build InterProScan command
        cmd = [
            "interproscan.sh",
            "-i", str(input_file),
            "-o", str(output_file),
            "-f", "xml",
            "--goterms",  # Enable GO term annotation
            "--pathways"  # Enable pathway annotation
        ]
        
        try:
            print(f"Running InterProScan on sequence: {seq_id}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("InterProScan completed successfully")
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"InterProScan failed with return code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def extract_go_terms_from_xml(self, xml_file):
        """
        Extract GO terms from InterProScan XML output
        
        Args:
            xml_file: Path to XML output file
            
        Returns:
            Dictionary with GO terms organized by category
        """
        go_terms = {
            'molecular_function': [],
            'biological_process': [],
            'cellular_component': []
        }
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Define namespace mapping
            namespaces = {'ns': 'https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/schemas'}
            
            # Find all GO terms in the XML using namespace
            for go_term in root.findall('.//ns:go-xref', namespaces):
                go_id = go_term.get('id')
                go_name = go_term.get('name')
                go_category = go_term.get('category')
                
                if go_id and go_name and go_category:
                    go_entry = {
                        'id': go_id,
                        'name': go_name
                    }
                    
                    if go_category == 'MOLECULAR_FUNCTION':
                        go_terms['molecular_function'].append(go_entry)
                    elif go_category == 'BIOLOGICAL_PROCESS':
                        go_terms['biological_process'].append(go_entry)
                    elif go_category == 'CELLULAR_COMPONENT':
                        go_terms['cellular_component'].append(go_entry)
            
            # Remove duplicates while preserving order
            for category in go_terms:
                seen = set()
                unique_terms = []
                for term in go_terms[category]:
                    term_key = (term['id'], term['name'])
                    if term_key not in seen:
                        seen.add(term_key)
                        unique_terms.append(term)
                go_terms[category] = unique_terms
        
        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            raise
            
        return go_terms

def retrieve_interpro_annotations(sequence, seq_id="sequence_1"):
    """
    Main function to get GO terms for a single sequence
    
    Args:
        sequence: Protein sequence string
        seq_id: Identifier for the sequence
    Returns:
        Dictionary with GO terms organized by category
    """
    runner = InterProScanRunner()
    
    # Run InterProScan
    output_file = runner.run_interproscan(sequence, seq_id)
    
    # Extract GO terms
    go_terms = runner.extract_go_terms_from_xml(output_file)
    print(go_terms)
    return go_terms

def main():
    data_root = '../data'
    ont = 'cc'
    test_df = pd.read_pickle(f"{data_root}/{ont}/time_data_esm.pkl")


    sequence = test_df['sequences'].values[0]
    name = "sequence_1"
    output = f"{data_root}/{ont}/interproscan_{name}.json"
    temp_dir = f"{data_root}/{ont}/temp_interproscan"
    os.makedirs(temp_dir, exist_ok=True)
    # temp_dir = tempfile.mkdtemp()

    try:
        # Initialize InterProScan runner
        runner = InterProScanRunner()
        
        # Run InterProScan
        output_file = runner.run_interproscan(
            sequence, 
            seq_id=name,
            temp_dir=temp_dir
        )
        
        # Extract GO terms
        go_terms = runner.extract_go_terms_from_xml(output_file)
        print(go_terms)
        total_terms = sum(len(terms) for terms in go_terms.values())
        print(f"\nSummary:")
        print(f"Total GO terms found: {total_terms}")
        print(f"Molecular Function: {len(go_terms['molecular_function'])}")
        print(f"Biological Process: {len(go_terms['biological_process'])}")
        print(f"Cellular Component: {len(go_terms['cellular_component'])}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
