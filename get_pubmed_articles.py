import argparse
import os
import json
import re
import time
import logging
from typing import Dict, List, Optional
import xmltodict
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBMED_DIR = os.path.join(BASE_DIR, 'data', 'pubmed')
NCBI_EUTILS_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
PMID_PATTERN = re.compile(r'^\d{8}$')
DEFAULT_RATE_LIMIT = 3  # Requests per second without API key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'pubmed_fetch.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PubMedClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
    def fetch_article(self, pmid: str) -> Optional[Dict]:
        """Fetch article data from PubMed using E-Utilities"""
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = self.session.get(NCBI_EUTILS_URL, params=params)
            response.raise_for_status()
            data = xmltodict.parse(response.content)
            return self._parse_article(data)
        except Exception as e:
            logger.error(f"Failed to fetch PMID {pmid}: {str(e)}")
            return None

    def _parse_article(self, data: Dict) -> Dict:
        """Parse XML response into structured data"""
        article = data.get('PubmedArticleSet', {}).get('PubmedArticle', {})
        medline = article.get('MedlineCitation', {})
        
        return {
            'pmid': medline.get('PMID', {}).get('#text', ''),
            'title': medline.get('Article', {}).get('ArticleTitle', ''),
            'abstract': medline.get('Article', {}).get('Abstract', {}).get('AbstractText', ''),
            'authors': [
                f"{author.get('LastName', '')} {author.get('Initials', '')}"
                for author in medline.get('Article', {}).get('AuthorList', {}).get('Author', [])
            ],
            'journal': medline.get('Article', {}).get('Journal', {}).get('Title', ''),
            'publication_date': medline.get('Article', {}).get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        }

def validate_pmid(pmid: str) -> bool:
    """Validate PMID format"""
    return bool(PMID_PATTERN.match(pmid))

def process_tsv(file_path: str) -> List[str]:
    """Extract PMIDs from TSV file"""
    pmids = []
    try:
        with open(file_path, 'r') as f:
            header = next(f).strip().split('\t')
            if len(header) < 9:
                raise ValueError("TSV file must have at least 9 columns")
                
            for line_num, line in enumerate(f, 2):
                columns = line.strip().split('\t')
                if len(columns) >= 9:
                    ids = columns[8].split(';')
                    pmids.extend([pid.strip() for pid in ids if pid.strip()])
                else:
                    logger.warning(f"Skipping line {line_num}: Insufficient columns")
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing TSV file: {str(e)}")
        raise
        
    valid_pmids = [pid for pid in pmids if validate_pmid(pid)]
    logger.info(f"Found {len(valid_pmids)} valid PMIDs from {len(pmids)} total IDs")
    return valid_pmids

def save_article(pmid: str, data: Dict) -> None:
    """Save article data to JSON file"""
    if not data:
        return
        
    output_path = os.path.join(PUBMED_DIR, f"{pmid}.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved PMID {pmid}")
    except Exception as e:
        logger.error(f"Failed to save PMID {pmid}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Fetch PubMed articles from PMIDs in TSV')
    parser.add_argument('input_tsv', help='Path to input TSV file')
    parser.add_argument('--api-key', help='NCBI API key for higher rate limits')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(PUBMED_DIR, exist_ok=True)

    # Initialize client
    client = PubMedClient(api_key=args.api_key)
    pmids = process_tsv(args.input_tsv)
    
    logger.info(f"Found {len(pmids)} valid PMIDs to process")
    
    # Enhanced rate limiting with semaphore
    max_workers = 3 if not args.api_key else 10
    delay = 1 / max_workers
    semaphore = threading.Semaphore(max_workers)
    
    def process_pmid(pmid):
        try:
            with semaphore:
                time.sleep(delay)
                data = client.fetch_article(pmid)
                if data:
                    save_article(pmid, data)
                return pmid
        except Exception as e:
            logger.error(f"Critical error processing PMID {pmid}: {str(e)}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit all PMIDs for processing
            futures = [executor.submit(process_pmid, pmid) for pmid in pmids]
            
            # Handle keyboard interrupt gracefully
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Unhandled exception: {str(e)}")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Cancelling pending tasks...")
            for future in futures:
                future.cancel()
            logger.info("Shutdown complete")

if __name__ == '__main__':
    main()