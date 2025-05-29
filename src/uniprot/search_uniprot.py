import requests
import time
import json
import subprocess
import tempfile
import os
import re
import gzip
import urllib.request
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.uniprot.get_go_annotations import get_protein_go_annotations
from src.ontology import Ontology


class LocalBLASTRunner:
    """
    Class to handle local BLAST+ operations for protein sequence similarity searches
    """
    
    def __init__(self, blast_db_path: str, blast_bin_path: str = None):
        """
        Initialize BLAST runner
        
        Args:
            blast_db_path: Path to BLAST database (without extension)
            blast_bin_path: Path to BLAST+ binaries (optional, assumes in PATH)
        """
        self.blast_db_path = blast_db_path
        self.blast_bin_path = blast_bin_path or ""
        
        # Verify BLAST installation
        self._verify_blast_installation()
        
    def _verify_blast_installation(self):
        """Verify that BLAST+ is installed and accessible"""
        try:
            cmd = [self._get_blast_cmd('blastp'), '-version']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"BLAST+ version: {result.stdout.split()[1]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "BLAST+ not found. Please install BLAST+ and ensure it's in your PATH "
                "or provide the correct blast_bin_path"
            )
    
    def _get_blast_cmd(self, program: str) -> str:
        """Get full path to BLAST command"""
        if self.blast_bin_path:
            return os.path.join(self.blast_bin_path, program)
        return program
    
    def run_blastp(self, sequence: str, evalue: float = 0.001, 
                   max_hits: int = 100, min_identity: float = 30.0) -> List[Dict]:
        """
        Run BLASTp search against local database
        
        Args:
            sequence: Query protein sequence
            evalue: E-value threshold
            max_hits: Maximum number of hits to return
            min_identity: Minimum percent identity threshold
            
        Returns:
            List of BLAST hits with parsed information
        """
        # Create temporary files
        query_file = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        try:
            # Write query sequence
            query_file.write(f">query_seq\n{sequence}\n")
            query_file.close()
            output_file.close()
            
            # Build BLAST command
            cmd = [
                self._get_blast_cmd('blastp'),
                '-query', query_file.name,
                '-db', self.blast_db_path,
                '-outfmt', '15',  # JSON format
                '-out', output_file.name,
                '-evalue', str(evalue),
                '-max_target_seqs', str(max_hits),
                '-num_threads', '4'  # Use 4 threads for speed
            ]
            
            print(f"Running BLAST command: {' '.join(cmd)}")
            
            # Run BLAST
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stderr:
                print(f"BLAST stderr: {result.stderr}")
            
            # Parse results
            with open(output_file.name, 'r') as f:
                blast_data = json.load(f)
            
            # Process and filter results
            processed_hits = self._process_blast_results(blast_data, min_identity)
            
            return processed_hits
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"BLAST failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error running BLAST: {str(e)}")
        finally:
            # Cleanup temporary files
            try:
                os.unlink(query_file.name)
                os.unlink(output_file.name)
            except:
                pass
    
    def _process_blast_results(self, blast_data: dict, min_identity: float) -> List[Dict]:
        """Process raw BLAST JSON output into structured results"""
        hits = []
        
        try:
            search_results = blast_data['BlastOutput2'][0]['report']['results']['search']
            
            for hit in search_results.get('hits', []):
                # Get best HSP (High-scoring Segment Pair)
                if not hit.get('hsps'):
                    continue
                    
                best_hsp = hit['hsps'][0]  # HSPs are sorted by score
                
                # Calculate percent identity
                identity_count = best_hsp.get('identity', 0)
                align_length = best_hsp.get('align_len', 1)
                percent_identity = (identity_count / align_length) * 100
                
                # Filter by minimum identity
                if percent_identity < min_identity:
                    continue
                
                # Extract accession and other info
                hit_info = {
                    'accession': self._extract_accession(hit),
                    'description': self._extract_description(hit),
                    'length': hit.get('len', 0),
                    'evalue': best_hsp.get('evalue', 1.0),
                    'bit_score': best_hsp.get('bit_score', 0),
                    'percent_identity': round(percent_identity, 1),
                    'alignment_length': align_length,
                    'query_coverage': round((align_length / search_results.get('query_len', 1)) * 100, 1),
                    'subject_coverage': round((align_length / hit.get('len', 1)) * 100, 1),
                    'query_start': best_hsp.get('query_from', 0),
                    'query_end': best_hsp.get('query_to', 0),
                    'subject_start': best_hsp.get('hit_from', 0),
                    'subject_end': best_hsp.get('hit_to', 0)
                }
                
                hits.append(hit_info)
            
            # Sort by bit score (descending)
            hits.sort(key=lambda x: x['bit_score'], reverse=True)
            
        except KeyError as e:
            print(f"Error parsing BLAST results: {e}")
            return []
        
        return hits
    
    def _extract_accession(self, hit: dict) -> str:
        """Extract accession from BLAST hit"""
        # Try to get accession from hit accession field
        if 'accession' in hit:
            return hit['accession']
        
        # Fall back to parsing from description
        descriptions = hit.get('description', [])
        if descriptions:
            title = descriptions[0].get('title', '')
            # Look for UniProt-style accession (e.g., sp|P12345|GENE_HUMAN)
            match = re.search(r'[sp|tr]\|([A-Z0-9]+)\|', title)
            if match:
                return match.group(1)
            
            # Look for simple accession pattern
            match = re.search(r'^([A-Z0-9]+)', title)
            if match:
                return match.group(1)
        
        return hit.get('id', 'Unknown')
    
    def _extract_description(self, hit: dict) -> str:
        """Extract description from BLAST hit"""
        descriptions = hit.get('description', [])
        if descriptions:
            return descriptions[0].get('title', 'No description')
        return 'No description'


def setup_blast_database(fasta_file: str, db_name: str, blast_bin_path: str = None) -> str:
    """
    Create BLAST database from FASTA file
    
    Args:
        fasta_file: Path to input FASTA file
        db_name: Name for the database (without extension)
        blast_bin_path: Path to BLAST+ binaries (optional)
        
    Returns:
        Path to created database
    """
    makeblastdb_cmd = 'makeblastdb'
    if blast_bin_path:
        makeblastdb_cmd = os.path.join(blast_bin_path, 'makeblastdb')
    
    cmd = [
        makeblastdb_cmd,
        '-in', fasta_file,
        '-dbtype', 'prot',
        '-out', db_name,
        '-title', f'Custom protein database: {db_name}'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Database created successfully: {db_name}")
        print(result.stdout)
        return db_name
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create BLAST database: {e.stderr}")


def download_uniprot_database(output_dir: str = "blast_db") -> str:
    """
    Download UniProt Swiss-Prot database for local BLAST
    
    Args:
        output_dir: Directory to store database files
        
    Returns:
        Path to database
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs for UniProt Swiss-Prot
    swissprot_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    
    fasta_gz_path = os.path.join(output_dir, "uniprot_sprot.fasta.gz")
    fasta_path = os.path.join(output_dir, "uniprot_sprot.fasta")
    db_path = os.path.join(output_dir, "uniprot_sprot")
    
    print(f"Downloading UniProt Swiss-Prot database...")
    urllib.request.urlretrieve(swissprot_url, fasta_gz_path)
    
    print("Extracting database...")
    with gzip.open(fasta_gz_path, 'rb') as f_in:
        with open(fasta_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    print("Creating BLAST database...")
    db_path = setup_blast_database(fasta_path, db_path)
    
    # Cleanup
    os.remove(fasta_gz_path)
    os.remove(fasta_path)
    
    return db_path


def run_local_blast_search(sequence: str, blast_db_path: str, 
                          top_n: int = 10, min_identity: float = 30.0) -> List[Dict]:
    """
    Run local BLAST search and return formatted results
    
    Args:
        sequence: Query protein sequence
        blast_db_path: Path to local BLAST database
        top_n: Number of top hits to return
        min_identity: Minimum percent identity
        
    Returns:
        List of similar proteins with BLAST statistics
    """
    blast_runner = LocalBLASTRunner(blast_db_path)
    
    # Run BLAST search
    hits = blast_runner.run_blastp(
        sequence=sequence,
        evalue=0.001,
        max_hits=top_n * 2,  # Get more hits to filter
        min_identity=min_identity
    )
    
    # Format results for compatibility with your existing code
    formatted_results = []
    for hit in hits[:top_n]:
        result = {
            'accession': hit['accession'],
            'sequence_identity': hit['percent_identity'],
            'e_value': hit['evalue'],
            'bit_score': hit['bit_score'],
            'query_coverage': hit['query_coverage'],
            'description': hit['description']
        }
        formatted_results.append(result)
    
    return formatted_results


def database_exists(db_path: str) -> bool:
    """
    Check if BLAST database files exist
    
    Args:
        db_path: Path to BLAST database (without extension)
        
    Returns:
        True if database exists, False otherwise
    """
    required_extensions = ['.phr', '.pin', '.psq']
    
    for ext in required_extensions:
        if not os.path.exists(f"{db_path}{ext}"):
            return False
    return True


def search_uniprot(protein_sequence: str, hypothesized_go_function:
                   str, blast_db_path: str = None, top_n: int = 10,
                   min_identity: float = 30.0) -> Dict:
    """
    Search UniProt for proteins similar to the input sequence using local BLAST
    and check if they have the hypothesized GO function.
    
    Args:
        protein_sequence: Input protein sequence to search for
        hypothesized_go_function: GO term ID (e.g., 'GO:0003677') or term name
        blast_db_path: Path to local BLAST database (if None, will auto-download)
        top_n: Number of top similar proteins to analyze (default: 10)
        min_identity: Minimum sequence identity threshold (default: 30.0%)
    
    Returns:
        Dictionary containing:
        - similar_proteins: List of similar proteins with their details
        - function_validation: Summary of GO function validation
        - blast_summary: BLAST search statistics
    """
    print(f"Searching for proteins similar to your sequence using local BLAST...")
    print(f"Checking for GO function: {hypothesized_go_function}")


    ontology = Ontology("data/go-basic.obo", with_rels=True)
    
    # Step 1: Check if database exists, download if needed
    if blast_db_path is None:
        blast_db_path = "../data/uniprot_sprot"

    if not database_exists(blast_db_path):
        print(f"Database not found at {blast_db_path}")
        print("Downloading UniProt Swiss-Prot database (this may take a few minutes)...")

        # Extract directory from the path
        db_dir = os.path.dirname(blast_db_path) or "data"
        blast_db_path = download_uniprot_database(db_dir)
        print(f"Database downloaded and ready at: {blast_db_path}")
    else:
        print(f"Using existing database: {blast_db_path}")

    # Step 2: Run local BLAST search
    blast_results = run_local_blast_search(
        sequence=protein_sequence,
        blast_db_path=blast_db_path,
        top_n=top_n,
        min_identity=min_identity
    )

    if not blast_results:
        return {"error": "No similar proteins found in BLAST search"}

    print(f"Found {len(blast_results)} similar proteins from BLAST")

    # Step 3: Get detailed GO annotations for each protein from UniProt
    annotated_proteins = []
    failed_accessions = []

    for i, protein in enumerate(blast_results):
        print(f"Getting annotations for {protein['accession']} ({i+1}/{len(blast_results)})...")
        annotations = get_protein_go_annotations(protein['accession'])
        propagated_annotations = []
        for annotation in annotations:
            prop_annots = ontology.get_ancestors(annotation)
            propagated_annotations.append(annotation)
            propagated_annotations.extend(prop_annots)
        propagated_annotations = list(set(propagated_annotations))  # Remove duplicates
        protein['go_functions'] = propagated_annotations
        annotated_proteins.append(protein)

    if failed_accessions:
        print(f"\nFailed to get annotations for {len(failed_accessions)} proteins: {failed_accessions}")

    if not annotated_proteins:
        return {"error": "Could not retrieve GO annotations for any proteins"}

    print(f"Successfully retrieved annotations for {len(annotated_proteins)} proteins")

    # Step 4: Check which proteins have the hypothesized function
    function_validation = validate_go_function(annotated_proteins, hypothesized_go_function)

    return {
        "similar_proteins": annotated_proteins,
        "function_validation": function_validation,
        "blast_summary": {
            "total_blast_hits": len(blast_results),
            "annotated_proteins": len(annotated_proteins),
            "average_identity": sum(p['sequence_identity'] for p in blast_results) / len(blast_results),
            "best_identity": max(p['sequence_identity'] for p in blast_results),
            "worst_identity": min(p['sequence_identity'] for p in blast_results)
        },
        "search_summary": {
            "total_similar_proteins": len(annotated_proteins),
            "proteins_with_function": len([p for p in annotated_proteins if p.get('has_hypothesized_function', False)]),
            "validation_percentage": function_validation.get('percentage_with_function', 0)
        }
    }




def validate_go_function(proteins: List[Dict], hypothesized_function: str) -> Dict:
    """Check which proteins have the hypothesized GO function"""
    validation_results = []
    proteins_with_function = 0
    
    for protein in proteins:
        has_function = check_protein_has_function(protein, hypothesized_function)
        protein['has_hypothesized_function'] = has_function
        
        if has_function:
            proteins_with_function += 1
        
        validation_results.append({
            'accession': protein.get('accession', ''),
            'sequence_identity': protein.get('sequence_identity', 0),
            'has_function': has_function,
            'matching_go_terms': get_matching_go_terms(protein, hypothesized_function)
        })
    
    total_proteins = len(proteins)
    percentage_with_function = (proteins_with_function / total_proteins * 100) if total_proteins > 0 else 0
    
    return {
        'total_proteins_analyzed': total_proteins,
        'proteins_with_function': proteins_with_function,
        'percentage_with_function': round(percentage_with_function, 1),
        'validation_details': validation_results,
        'conclusion': get_validation_conclusion(percentage_with_function)
    }


def check_protein_has_function(protein: Dict, hypothesized_function: str) -> bool:
    """Check if a protein has the hypothesized GO function"""
    # Check all GO categories, not just functions
    all_go_terms = protein['go_functions']
    
    # Check if hypothesized_function is a GO ID or term name
    return any(go == hypothesized_function for go in all_go_terms)
                    
def get_matching_go_terms(protein: Dict, hypothesized_function: str) -> List[Dict]:
    """Get GO terms that match the hypothesized function"""
    all_go_terms = protein['go_functions']
    matching_terms = []
    
    for go in all_go_terms:
        if go == hypothesized_function:
            matching_terms.append(go)
                                    
    return matching_terms


def get_validation_conclusion(percentage: float) -> str:
    """Get a conclusion based on the validation percentage"""
    if percentage >= 80:
        return "Strong evidence: Most similar proteins have the hypothesized function"
    elif percentage >= 60:
        return "Moderate evidence: Many similar proteins have the hypothesized function"
    elif percentage >= 40:
        return "Weak evidence: Some similar proteins have the hypothesized function"
    else:
        return "Little evidence: Few similar proteins have the hypothesized function"

def extract_sequence(entry):
    """Safely extract sequence from entry"""
    sequence = entry.get('sequence', {})
    if isinstance(sequence, dict):
        return sequence.get('value', '')
    return ''
