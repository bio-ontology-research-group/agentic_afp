import pandas as pd
import requests
import time
import json
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniProtTextRetriever:
    def __init__(self, delay_between_requests: float = 0.1):
        """
        Initialize UniProt text retriever
        
        Args:
            delay_between_requests: Delay in seconds between API requests to be respectful
        """
        self.base_url = "https://rest.uniprot.org"
        self.delay = delay_between_requests
        
    def get_protein_info(self, accession: str) -> Dict:
        """
        Retrieve detailed protein information from UniProt
        
        Args:
            accession: UniProt accession ID
            
        Returns:
            Dictionary containing protein information
        """
        try:
            url = f"{self.base_url}/uniprotkb/{accession}"
            params = {'format': 'json'}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return self.extract_text_information(data)
            
        except Exception as e:
            logger.error(f"Error retrieving info for {accession}: {str(e)}")
            return {}
    
    def extract_text_information(self, uniprot_data: Dict) -> Dict:
        """
        Extract relevant text information for GO annotation refinement
        
        Args:
            uniprot_data: Raw UniProt JSON data
            
        Returns:
            Dictionary with extracted text information
        """
        info = {
            'accession': uniprot_data.get('primaryAccession', ''),
            'entry_name': uniprot_data.get('uniProtkbId', ''),
            'protein_names': [],
            'gene_names': [],
            'organism': '',
            'function_description': '',
            'catalytic_activity': [],
            'pathway_involvement': [],
            'go_annotations': {
                'biological_process': [],
                'molecular_function': [],
                'cellular_component': []
            },
            'subcellular_location': [],
            'domain_info': [],
            'keywords': [],
            'comments': []
        }
        
        # Extract protein names
        if 'proteinDescription' in uniprot_data:
            protein_desc = uniprot_data['proteinDescription']
            if 'recommendedName' in protein_desc:
                rec_name = protein_desc['recommendedName']
                if 'fullName' in rec_name:
                    info['protein_names'].append(rec_name['fullName']['value'])
            
            if 'alternativeName' in protein_desc:
                for alt_name in protein_desc['alternativeName']:
                    if 'fullName' in alt_name:
                        info['protein_names'].append(alt_name['fullName']['value'])
        
        # Extract gene names
        if 'genes' in uniprot_data:
            for gene in uniprot_data['genes']:
                if 'geneName' in gene:
                    info['gene_names'].append(gene['geneName']['value'])
        
        # Extract organism
        if 'organism' in uniprot_data:
            org = uniprot_data['organism']
            if 'scientificName' in org:
                info['organism'] = org['scientificName']
        
        # Extract comments (function, catalytic activity, pathway, subcellular location)
        if 'comments' in uniprot_data:
            for comment in uniprot_data['comments']:
                comment_type = comment.get('commentType', '')
                
                if comment_type == 'FUNCTION':
                    for text in comment.get('texts', []):
                        info['function_description'] += text.get('value', '') + ' '
                
                elif comment_type == 'CATALYTIC ACTIVITY':
                    reaction = comment.get('reaction', {})
                    if 'name' in reaction:
                        info['catalytic_activity'].append(reaction['name'])
                
                elif comment_type == 'PATHWAY':
                    for text in comment.get('texts', []):
                        info['pathway_involvement'].append(text.get('value', ''))
                
                elif comment_type == 'SUBCELLULAR LOCATION':
                    for location in comment.get('subcellularLocations', []):
                        if 'location' in location:
                            info['subcellular_location'].append(location['location']['value'])
                
                # General comments for additional context
                elif comment_type in ['SIMILARITY', 'DOMAIN', 'MISCELLANEOUS']:
                    for text in comment.get('texts', []):
                        info['comments'].append(f"{comment_type}: {text.get('value', '')}")
        
        # Extract GO annotations
        if 'dbReferences' in uniprot_data:
            for ref in uniprot_data['dbReferences']:
                if ref.get('type') == 'GO':
                    go_id = ref.get('id', '')
                    go_term = ''
                    go_aspect = ''
                    
                    for prop in ref.get('properties', []):
                        if prop.get('key') == 'GoTerm':
                            go_term = prop.get('value', '').split(':')[-1] if ':' in prop.get('value', '') else prop.get('value', '')
                        elif prop.get('key') == 'GoEvidenceType':
                            go_aspect = prop.get('value', '')
                    
                    # Determine GO category based on GO ID prefix
                    if go_id.startswith('GO:'):
                        if go_term:
                            go_entry = f"{go_id}: {go_term}"
                            # Simple heuristic for categorization - in practice, you'd need GO ontology mapping
                            if any(keyword in go_term.lower() for keyword in ['process', 'regulation', 'pathway', 'response']):
                                info['go_annotations']['biological_process'].append(go_entry)
                            elif any(keyword in go_term.lower() for keyword in ['activity', 'binding', 'catalytic']):
                                info['go_annotations']['molecular_function'].append(go_entry)
                            else:
                                info['go_annotations']['cellular_component'].append(go_entry)
        
        # Extract features (domains, active sites, etc.)
        if 'features' in uniprot_data:
            for feature in uniprot_data['features']:
                feature_type = feature.get('type', '')
                if feature_type in ['DOMAIN', 'REGION', 'MOTIF', 'ACT_SITE', 'BINDING']:
                    description = feature.get('description', '')
                    if description:
                        info['domain_info'].append(f"{feature_type}: {description}")
        
        # Extract keywords
        if 'keywords' in uniprot_data:
            for keyword in uniprot_data['keywords']:
                info['keywords'].append(keyword.get('name', ''))
        
        # Clean up text fields
        info['function_description'] = info['function_description'].strip()
        
        return info
    
    def create_text_summary(self, protein_info: Dict) -> str:
        """
        Create a comprehensive text summary from protein information
        
        Args:
            protein_info: Dictionary with extracted protein information
            
        Returns:
            Formatted text summary suitable for LLM training
        """
        if not protein_info.get('accession'):
            return "No UniProt information available"
        
        text_parts = []
        
        # Basic information
        if protein_info.get('protein_names'):
            text_parts.append(f"Protein: {'; '.join(protein_info['protein_names'])}")
        
        if protein_info.get('gene_names'):
            text_parts.append(f"Gene: {'; '.join(protein_info['gene_names'])}")
        
        if protein_info.get('organism'):
            text_parts.append(f"Organism: {protein_info['organism']}")
        
        # Functional information
        if protein_info.get('function_description'):
            text_parts.append(f"Function: {protein_info['function_description']}")
        
        if protein_info.get('catalytic_activity'):
            text_parts.append(f"Catalytic Activity: {'; '.join(protein_info['catalytic_activity'])}")
        
        if protein_info.get('pathway_involvement'):
            text_parts.append(f"Pathways: {'; '.join(protein_info['pathway_involvement'])}")
        
        # GO annotations
        go_parts = []
        for go_type, terms in protein_info.get('go_annotations', {}).items():
            if terms:
                go_parts.append(f"{go_type.replace('_', ' ').title()}: {'; '.join(terms)}")
        
        if go_parts:
            text_parts.append(f"GO Annotations - {'; '.join(go_parts)}")
        
        if protein_info.get('subcellular_location'):
            text_parts.append(f"Subcellular Location: {'; '.join(protein_info['subcellular_location'])}")
        
        if protein_info.get('domain_info'):
            text_parts.append(f"Domains: {'; '.join(protein_info['domain_info'])}")
        
        if protein_info.get('keywords'):
            text_parts.append(f"Keywords: {'; '.join(protein_info['keywords'])}")
        
        if protein_info.get('comments'):
            text_parts.append(f"Additional Info: {'; '.join(protein_info['comments'])}")
        
        return ' | '.join(text_parts)

    def process_dataframe_with_text_column(self, df: pd.DataFrame, accession_column: str = None) -> pd.DataFrame:
        """
        Process DataFrame and add uniprot_text column to the original dataframe
        
        Args:
            df: Input DataFrame with protein sequences and/or accessions
            accession_column: Name of the column containing UniProt accessions
            
        Returns:
            Original DataFrame with added 'uniprot_text' column
        """
        # Create a copy of the original dataframe
        result_df = df.copy()
        
        # Initialize the uniprot_text column
        result_df['uniprot_text'] = ''
        
        logger.info(f"Processing {len(df)} entries and adding uniprot_text column...")
        
        # Check if accession column exists and has data
        
        logger.info(f"Using existing accessions from column '{accession_column}'")
                    
        for idx, row in df.iterrows():
            logger.info(f"Processing entry {idx + 1}/{len(df)}")
            
            
            # Use existing accession(s)
            accession_data = row[accession_column]
            if isinstance(accession_data, list):
                accession = accession_data[0]
            else:
                raise TypeError(f"Found accession data of type {type(accession_data)}")
                    
            
            logger.info(f"Processing UniProt entry: {accession}")
            # Get detailed information
            protein_info = self.get_protein_info(accession)

            # Create text summary and add to dataframe
            text_summary = self.create_text_summary(protein_info)
            result_df.at[idx, 'uniprot_text'] = text_summary
            # Be respectful to the API
            time.sleep(self.delay)
        
        return result_df
        
    def save_text_for_llm_from_dataframe(self, df: pd.DataFrame, output_file: str):
        """
        Save text information from dataframe with uniprot_text column
        
        Args:
            df: DataFrame with uniprot_text column
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                uniprot_text = row.get('uniprot_text', '')
                if uniprot_text and uniprot_text != "No UniProt information available":
                    # Include original identifiers if available
                    identifiers = []
                    if 'proteins' in df.columns:
                        identifiers.append(f"Protein ID: {row['proteins']}")
                    if 'accessions' in df.columns:
                        identifiers.append(f"UniProt: {row['accessions']}")
                    
                    if identifiers:
                        f.write(' | '.join(identifiers) + '\n')
                    f.write(f"Description: {uniprot_text}\n")
                    if 'sequences' in df.columns:
                        f.write(f"Sequence: {row['sequences']}\n")
                    f.write("-" * 80 + "\n")
        
        logger.info(f"Text information saved to {output_file}")

def test_retriever_with_text_column(df: pd.DataFrame, accession_column: str = 'accessions'):
    """
    Test function that adds uniprot_text column to the first two entries
    
    Args:
        df: Input DataFrame with protein sequences
        accession_column: Name of the column containing accessions
    """
    print("Testing UniProt Text Retriever with first 2 entries...")
    print("=" * 60)
    
    # Create test dataframe with first 2 entries
    test_df = df.head(2).copy()
    
    # Initialize retriever with shorter delay for testing
    retriever = UniProtTextRetriever(delay_between_requests=0.1)
    
    # Process the test entries and add text column
    result_df = retriever.process_dataframe_with_text_column(test_df, accession_column)
    
    # Display results
    for idx, row in result_df.iterrows():
        print(f"\nEntry {idx + 1}:")
        print("-" * 40)
        
        # Show original data
        if 'proteins' in result_df.columns:
            print(f"Original Protein: {row.get('proteins', 'N/A')}")
        if accession_column and accession_column in result_df.columns:
            print(f"Original Accessions: {row.get(accession_column, 'N/A')}")
        if 'genes' in result_df.columns:
            print(f"Original Gene Info: {row.get('genes', 'N/A')}")
        
        # Show the generated text
        uniprot_text = row.get('uniprot_text', '')
        if uniprot_text and uniprot_text != "No UniProt information available":
            print(f"UniProt Text: {uniprot_text[:300]}...")
        else:
            print(f"UniProt Text: {uniprot_text}")
    
    # Save test results
    test_output_file = 'test_data_with_text.pkl'
    result_df.to_pickle(test_output_file)
    
    print(f"\nTest completed! Enhanced dataframe saved to: {test_output_file}")
    print(f"Added 'uniprot_text' column to original dataframe")
    
    return result_df

# Keep the old test function for backward compatibility
def test_retriever(df: pd.DataFrame, accession_column: str = 'accessions'):
    """
    Original test function that returns separate results dataframe
    """
    return test_retriever_with_text_column(df, accession_column)

def main():
    """
    Main function that accepts command line arguments for file processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrieve UniProt text information for protein sequences')
    parser.add_argument('input_file', help='Input pickle file containing DataFrame with protein sequences')
    parser.add_argument('--accession_column', default='accessions', help='Name of column containing UniProt accessions (default: accessions)')
    parser.add_argument('--output_prefix', default='uniprot_output', help='Prefix for output files (default: uniprot_output)')
    parser.add_argument('--test_only', action='store_true', help='Only process first 2 sequences for testing')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between API requests in seconds (default: 0.2)')
    
    args = parser.parse_args()
    
    try:
        # Load the pickle file
        print(f"Loading data from {args.input_file}...")
        df = pd.read_pickle(args.input_file)
        print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")
                
        # Check if accession column exists
        if args.accession_column not in df.columns:
            print(f"Warning: Column '{args.accession_column}' not found in DataFrame")
            print(f"Available columns: {list(df.columns)}")
            args.accession_column = None
        else:
            print(f"Found accession column '{args.accession_column}' - will use existing accessions")
        
        if args.test_only:
            # Run test mode
            print("\nRunning in TEST MODE (first 2 entries only)")
            results = test_retriever_with_text_column(df, args.accession_column)
        else:
            # Process full dataframe
            print(f"\nProcessing all {len(df)} entries...")
            retriever = UniProtTextRetriever(delay_between_requests=args.delay)
            results = retriever.process_dataframe_with_text_column(df, args.accession_column)
            
            # Save enhanced dataframe with text column
            output_file = args.input_file.replace('.pkl', '_with_text.pkl')
            results.to_pickle(output_file)
            
            print(f"\nProcessing completed!")
            print(f"Enhanced dataframe with 'uniprot_text' column saved to: {output_file}")
            
            # Also save just the text information separately if needed
            retriever.save_text_for_llm_from_dataframe(results, f"{args.output_prefix}_for_llm.txt")
        
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
