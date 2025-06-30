#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from esm import FastaBatchedDataset, ProteinBertModel, pretrained
import os
import gzip
import click as ck

class GzippedFastaBatchedDataset(FastaBatchedDataset):

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with gzip.open(fasta_file, "rt") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)


def extract_esm(fasta_file, model_location='esm2_t36_3B_UR50D',
                truncation_seq_length=1022, toks_per_batch=4096,
                device=None):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if device:
        model = model.to(device)
    
    if fasta_file.endswith('.gz'):
        dataset = GzippedFastaBatchedDataset.from_file(fasta_file)
    else:
        dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    # output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = False

    repr_layers = [36,]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if device:
                toks = toks.to(device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                result = {"label": label}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }
                folder = Path(f'data/esm2/{label.split("_")[1]}')
                folder.mkdir(exist_ok=True, parents=True)
                filename = f'{folder}/{label}.pt'
                torch.save(result["mean_representations"][36], filename)


@ck.command()
@ck.option('--fasta-file', '-ff', default='data/missing_esm2.fa',
           help='Fasta file with protein sequences')
@ck.option('--device', '-d', default='cuda:1',
           help='Device for ESMC model')
def main(fasta_file, device):
    extract_esm(fasta_file, device=device)

if __name__ == '__main__':
    main()