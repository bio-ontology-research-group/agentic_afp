from collections import deque, Counter
import warnings
import numpy as np
from xml.etree import ElementTree as ET
import math
import random
import numpy as np
import os

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
HAS_FUNCTION = 'http://mowl.borg/has_function'

FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

# CAFA5 Targets
CAFA_TARGETS = set([
    '9606', '10090', '10116', '3702', '83333', '7227', '287', '4896',
    '7955', '44689', '243273', '6239', '226900', '4577', '9823',
    '8355', '85962', '99287', '160488', '170187', '223283', '224308',
    '237561', '243232', '321314', '10172', '1072389', '1094619',
    '126793', '186763', '229533', '235443', '2587412', '27300',
    '284812', '294381', '3197', '3218', '36329', '39947', '426428',
    '48703', '498257', '508771', '515849', '5823', '6253', '7159',
    '7460', '7962', '8090', '83332', '8364', '9031', '9541', '9555',
    '9601', '9615', '981087', '9913', '100989', '111177', '120305',
    '186611', '193080', '196418', '227321', '271848', '284591',
    '284592', '284811', '28985', '29156', '292442', '330879',
    '338838', '338839', '367110', '412038', '5478', '660122', '70142',
    '749593', '8654', '8670', '8671', '8673', '930089', '559292',
    '38281'])
    
def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES


class Ontology(object):

    def __init__(self, filename, with_rels=True, taxon_constraints_file=None):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0
        self.ancestors = {}
        self.leaf_nodes = None
        self._taxon_map = None
        # Load taxon constraints if file is provided
        if taxon_constraints_file:
            self.load_taxon_constraints(taxon_constraints_file)

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        self.ic = {}
        # with open('data/cafa5/IA.txt') as f:
        #     for line in f:
        #         it = line.strip().split('\t')
        #         if len(it) == 2:
        #             self.ic[it[0]] = float(it[1])
        #             self.ic_norm = max(self.ic_norm, float(it[1]))
        # return
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    obj['definition'] = None
                    # Initialize taxon constraint lists
                    obj['in_taxon'] = list()
                    obj['never_in_taxon'] = list()
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
                    elif l[0] == 'def':
                        def_text = l[1]
                        if def_text.startswith('"') and '"' in def_text[1:]:
                            end_quote = def_text.find('"', 1)
                            obj['definition'] = def_text[1:end_quote]
                        else:
                            obj['definition'] = def_text

            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
     
        return ont

    def load_taxon_constraints(self, filename):
        """
        Load taxon constraints from a taxon constraints OBO file.
        
        Args:
            filename (str): Path to the taxon constraints OBO file
        """
        current_term_id = None
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line == '[Term]':
                    current_term_id = None
                    continue
                elif line.startswith('id: '):
                    current_term_id = line.split(': ')[1]
                    # Initialize taxon constraint lists if term exists in ontology
                    if current_term_id in self.ont:
                        if 'in_taxon' not in self.ont[current_term_id]:
                            self.ont[current_term_id]['in_taxon'] = list()
                        if 'never_in_taxon' not in self.ont[current_term_id]:
                            self.ont[current_term_id]['never_in_taxon'] = list()
                elif line.startswith('relationship: ') and current_term_id:
                    # Parse relationship lines
                    rel_parts = line.split(': ')[1].split()
                    if len(rel_parts) >= 2:
                        rel_type = rel_parts[0]
                        taxon_id = rel_parts[1].split(':')[1]
                        
                        if current_term_id in self.ont:
                            if rel_type == 'RO:0002162':  # in_taxon
                                self.ont[current_term_id]['in_taxon'].append(taxon_id)
                            elif rel_type == 'RO:0002161':  # never_in_taxon
                                self.ont[current_term_id]['never_in_taxon'].append(taxon_id)
                elif line.startswith('property_value: ') and current_term_id:
                    # Parse property_value lines
                    prop_parts = line.split(': ')[1].split()
                    if len(prop_parts) >= 2:
                        prop_type = prop_parts[0]
                        taxon_id = prop_parts[1].split(':')[1]
                        
                        if current_term_id in self.ont:
                            if prop_type == 'RO:0002162':  # in_taxon
                                self.ont[current_term_id]['in_taxon'].append(taxon_id)
                            elif prop_type == 'RO:0002161':  # never_in_taxon
                                self.ont[current_term_id]['never_in_taxon'].append(taxon_id)

    def get_in_taxon(self, term_id):
        """
        Get the list of taxa that this term is constrained to be in.
        
        Args:
            term_id (str): The GO term ID
            
        Returns:
            list: List of taxon IDs (e.g., ['NCBITaxon:2759'])
        """
        if self.has_term(term_id):
            return self.ont[term_id].get('in_taxon', [])
        return []

    def get_never_in_taxon(self, term_id):
        """
        Get the list of taxa that this term is constrained to never be in.
        
        Args:
            term_id (str): The GO term ID
            
        Returns:
            list: List of taxon IDs (e.g., ['NCBITaxon:4896'])
        """
        if self.has_term(term_id):
            return self.ont[term_id].get('never_in_taxon', [])
        return []

    def is_valid_for_taxon(self, term_id, taxon_id):
        """
        Check if a GO term is valid for a given taxon based on taxon constraints.
        
        Args:
            term_id (str): The GO term ID
            taxon_id (str): The taxon ID (e.g., 'NCBITaxon:9606')
            
        Returns:
            bool: True if the term is valid for the taxon, False otherwise
        """
        if not self.has_term(term_id):
            return False
            
        # Check never_in_taxon constraints
        never_in_taxon = self.get_never_in_taxon(term_id)
        if taxon_id in never_in_taxon:
            return False
            
        # Check in_taxon constraints
        in_taxon = self.get_in_taxon(term_id)
        if in_taxon and taxon_id not in in_taxon:
            return False
            
        return True

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        if term_id in self.ancestors:
            return self.ancestors[term_id]
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        self.ancestors[term_id] = term_set
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_anchestors(term_id)
        return prop_terms

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

    def get_leaf_nodes(self, terms):
        if self.leaf_nodes is not None:
            return self.leaf_nodes
        
        leaf_nodes = set()
    
        for term in terms:
            descendants = self.get_term_set(term)
            if len(descendants) == 1 and term in descendants:
                leaf_nodes.add(term)
        self.leaf_nodes = leaf_nodes
        return self.leaf_nodes
    
    def get_term_name(self, term_id):
        """
        Get the human-readable name/label of a GO term.

        Args:
            term_id (str): The GO term ID (e.g., 'GO:0008150')

        Returns:
            str: The name/label of the term, or None if term doesn't exist
        """
        if self.has_term(term_id):
            return self.ont[term_id].get('name', None)
        return None

    def get_term_definition(self, term_id):
        """
        Get the definition of a GO term.

        Args:
            term_id (str): The GO term ID (e.g., 'GO:0008150')

        Returns:
            str: The definition of the term, or None if term doesn't exist or has no definition
        """
        if self.has_term(term_id):
            return self.ont[term_id].get('definition', None)
        return None

    def get_term_info(self, term_id):
        """
        Get comprehensive information about a GO term including name, namespace, taxon constraints, etc.

        Args:
            term_id (str): The GO term ID (e.g., 'GO:0008150')

        Returns:
            dict: Dictionary containing term information, or None if term doesn't exist
        """
        if self.has_term(term_id):
            term = self.ont[term_id]
            return {
                'id': term_id,
                'name': term.get('name', 'Unknown'),
                'namespace': term.get('namespace', 'Unknown'),
                'parents': term.get('is_a', []),
                'children': list(term.get('children', set())),
                'alt_ids': term.get('alt_ids', []),
                'in_taxon': term.get('in_taxon', []),
                'never_in_taxon': term.get('never_in_taxon', [])
            }
        return None

    @property
    def taxon_map(self):
        """
        Creates a dictionary where keys are taxon IDs and values are pairs of lists.
        The first list contains GO terms that have an in_taxon constraint for that taxon.
        The second list contains GO terms that have a never_in_taxon constraint for that taxon.
        
        Returns:
            dict: Dictionary mapping taxon IDs to pairs of lists [in_taxon_terms, never_in_taxon_terms]
        """
        if self._taxon_map is not None:
            return self._taxon_map

        taxon_map = {}
        
        # Iterate through all terms in the ontology
        for term_id, term in self.ont.items():
            # Process in_taxon constraints
            for taxon_id in term.get('in_taxon', []):
                if taxon_id not in taxon_map:
                    taxon_map[taxon_id] = [[], []]
                taxon_map[taxon_id][0].append(term_id)
                
            # Process never_in_taxon constraints
            for taxon_id in term.get('never_in_taxon', []):
                if taxon_id not in taxon_map:
                    taxon_map[taxon_id] = [[], []]
                taxon_map[taxon_id][1].append(term_id)

        self._taxon_map = taxon_map
        return self._taxon_map


def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs


class DataGenerator(object):

    def __init__(self, batch_size, is_sparse=False):
        self.batch_size = batch_size
        self.is_sparse = is_sparse

    def fit(self, inputs, targets=None):
        self.start = 0
        self.inputs = inputs
        self.targets = targets
        if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
            self.size = self.inputs[0].shape[0]
        else:
            self.size = self.inputs.shape[0]
        self.has_targets = targets is not None

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
                res_inputs = []
                for inp in self.inputs:
                    if self.is_sparse:
                        res_inputs.append(
                            inp[batch_index, :].toarray())
                    else:
                        res_inputs.append(inp[batch_index, :])
            else:
                if self.is_sparse:
                    res_inputs = self.inputs[batch_index, :].toarray()
                else:
                    res_inputs = self.inputs[batch_index, :]
            self.start += self.batch_size
            if self.has_targets:
                if self.is_sparse:
                    labels = self.targets[batch_index, :].toarray()
                else:
                    labels = self.targets[batch_index, :]
                return (res_inputs, labels)
            return res_inputs
        else:
            self.reset()
            return self.next()
