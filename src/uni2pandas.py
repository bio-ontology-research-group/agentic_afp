#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from ontology import Ontology, is_exp_code, is_cafa_target, FUNC_DICT
import torch as th
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--swissprot-file', '-sf', default='data/uniprot_sprot_2023_05.dat.gz',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
@ck.option(
    '--out-file', '-o', default='data/swissprot_exp_2023_05.pkl',
    help='Result file with a list of proteins, sequences and annotations')
def main(swissprot_file, out_file):
    go = Ontology('data/go.obo', with_rels=True)
    proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros = load_data(swissprot_file)
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'genes': genes,
        'sequences': sequences,
        'annotations': annotations,
        'string_ids': string_ids,
        'orgs': orgs,
        'interpros': interpros
    })

    logging.info('Filtering proteins with experimental annotations')
    index = []
    annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = []
        all_annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
            all_annots.append(go_id)
        # Ignore proteins without experimental annotations
        if len(annots) == 0:
            continue
        index.append(i)
        annotations.append(annots)
    df = df.iloc[index]
    df = df.reset_index()
    df['exp_annotations'] = annotations
    
    prop_annotations = []
    esm2_data = []
    missing = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            annot_set |= go.get_ancestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
        filename = f'data/esm2/{row.proteins.split("_")[1]}/{row.proteins}.pt'
        if not os.path.exists(filename):
            missing.append((row.proteins, row.sequences))
        else:
            emb = th.load(filename)
            if len(emb) > 0:
                esm2_data.append(emb)
            else:
                missing.append((row.proteins, row.sequences))
    
    with open('data/missing_esm2.fa', 'w') as f:
        for prot, seq in missing:
            record = SeqRecord(
                Seq(seq),
                id=prot,
                description=''
            )
            SeqIO.write(record, f, 'fasta')

    df['prop_annotations'] = prop_annotations
    df['esm2_data'] = esm2_data

    interpro2go = {}
    gos = set()
    with open('data/interpro2go.txt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            it = line.strip().split()
            ipr_id = it[0].split(':')[1]
            go_id = it[-1]
            if not go.has_term(go_id):
                continue
            if ipr_id not in interpro2go:
                interpro2go[ipr_id] = set()
            interpro2go[ipr_id].add(go_id)
            gos.add(go_id)

    ipr2go = []
    for i, row in enumerate(df.itertuples()):
        prot_id = row.proteins
        annots = set()
        for ipr in row.interpros:
            if ipr in interpro2go:
                annots |= interpro2go[ipr]
        ipr2go.append(annots)
    df['interpro2go'] = ipr2go
    
    df.to_pickle(out_file)
    logging.info('Successfully saved %d proteins' % (len(df),) )
    
    
def load_data(swissprot_file):
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    string_ids = list()
    orgs = list()
    genes = list()
    interpros = list()
    with gzip.open(swissprot_file, 'rt') as f:
        prot_id = ''
        seq = ''
        org = ''
        gene_id = ''
        annots = list()
        prot_ac = list()
        strs = list()
        iprs = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    string_ids.append(strs)
                    orgs.append(org)
                    genes.append(gene_id)
                    interpros.append(iprs)
                prot_id = items[1]
                seq = ''
                org = ''
                gene_id = ''
                annots = list()
                prot_ac = list()
                strs = list()
                iprs = list()
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac += [x.strip() for x in items[1].split(';') if x.strip() != '']
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                elif items[0] == 'STRING':
                    str_id = items[1]
                    strs.append(str_id)
                elif items[0] == 'GeneID':
                    gene_id = items[1]
                elif items[0] == 'InterPro':
                    ipr_id = items[1]
                    iprs.append(ipr_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        string_ids.append(strs)
        orgs.append(org)
        genes.append(gene_id)
        interpros.append(iprs)
    return proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros


if __name__ == '__main__':
    main()
