#!/usr/bin/env python3
"""
Parse InterPro to GO mapping file and generate TSV output.
Expected input format: InterPro:ID Description > GO:description ; GO:ID
Output format: TSV with columns: interpro_id, go_id, go_description
"""

import re
import sys

def parse_interpro_file(input_file, output_file):
    """
    Parse InterPro mapping file and write TSV output.
    
    Args:
        input_file (str): Path to input InterPro mapping file
        output_file (str): Path to output TSV file
    """
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write TSV header
        outfile.write("interpro_id\tgo_id\tgo_description\n")
        
        for line in infile:
            line = line.strip()
            
            # Skip comment lines and empty lines
            if line.startswith('!') or not line:
                continue
            
            # Parse the line format: InterPro:ID Description > GO:description ; GO:ID
            if line.startswith('InterPro:'):
                try:
                    # Split on ' > ' to separate InterPro part from GO part
                    interpro_part, go_part = line.split(' > ', 1)
                    
                    # Extract InterPro ID (everything after 'InterPro:' until first space)
                    interpro_match = re.match(r'InterPro:(\S+)', interpro_part)
                    if not interpro_match:
                        continue
                    interpro_id = interpro_match.group(1)
                    
                    # Split GO part on ' ; ' to separate description from ID
                    go_desc, go_id_part = go_part.split(' ; ', 1)
                    
                    # Clean up GO description (remove 'GO:' prefix if present)
                    go_description = go_desc.replace('GO:', '').strip()
                    
                    # Extract GO ID
                    go_id = go_id_part.strip()
                    
                    # Write to output file
                    outfile.write(f"{interpro_id}\t{go_id}\t{go_description}\n")
                    
                except ValueError as e:
                    print(f"Warning: Could not parse line: {line}", file=sys.stderr)
                    continue

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python interpro_parser.py <input_file> <output_file>")
        print("Example: python interpro_parser.py interpro2go.txt output.tsv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        parse_interpro_file(input_file, output_file)
        print(f"Successfully parsed {input_file} and wrote output to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
