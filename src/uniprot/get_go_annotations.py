import requests
from typing import Dict, Optional

def extract_gene_name(entry):
    """Safely extract gene name from entry"""
    genes = entry.get('genes', [])
    if genes and isinstance(genes, list) and len(genes) > 0:
        gene = genes[0]
        if isinstance(gene, dict):
            gene_name = gene.get('geneName', {})
            if isinstance(gene_name, dict):
                return gene_name.get('value', '')
    return ''



def extract_organism_name(entry):
    """Safely extract organism name from entry"""
    organism = entry.get('organism', {})
    if isinstance(organism, dict):
        return organism.get('scientificName', '')
    return ''



def get_protein_go_annotations(accession: str) -> Optional[Dict]:

    
    """Get GO annotations for a specific protein"""
    # Clean up the accession - remove any extra characters

    
    
    accession = accession.strip().split('|')[-1].split()[0]
    
    url = f"https://rest.uniprot.org/uniprotkb/{accession}"
    params = {
        'format': 'json',
        'fields': 'accession,go,gene_names,organism_name'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # print(f"Data received for {accession}: {data}")
            
            go_functions = []
                        
            for ref in data["uniProtKBCrossReferences"]:
                if ref.get('database') == 'GO':
                    go_id = ref['id']
                    # go_id = next((p['value'] for p in properties if p['key'] == 'GoTerm'), None)
                    go_functions.append(go_id)
            return go_functions
            
        elif response.status_code == 400:
            print(f"Bad request for {accession} (400) - possibly invalid accession")
            print(f"Request URL: {response.url}")
            # Try a simplified request
            return get_protein_go_annotations_simple(accession)
        elif response.status_code == 404:
            print(f"Protein {accession} not found (404)")
            return None
        elif response.status_code == 429:
            print(f"Rate limited (429) - waiting and retrying...")
            time.sleep(2)
            return get_protein_go_annotations(accession)  # Retry once
        else:
            print(f"Failed to get annotations for {accession}: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"Error getting annotations for {accession}: {e}")
        return None

