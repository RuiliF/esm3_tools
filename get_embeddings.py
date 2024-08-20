from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from Bio import SeqIO
import argparse
from pathlib import Path
from tqdm import tqdm
from glob import glob
import os
import pickle
import torch
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file or directory")
    parser.add_argument("--file-type", type=str, default="sequence")
    args = parser.parse_args()
    return args

def process_seq(client, seq):
    protein = ESMProtein(sequence=str(seq))
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    return output.per_residue_embedding

def process_pdb(client, pdb):

    chain = ProteinChain.from_pdb(pdb)
    protein = ESMProtein.from_protein_chain(chain)
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    return output.per_residue_embedding

def dump_emd(emd, uniprot, output_dir):
    with torch.no_grad():
        with open(os.path.join(output_dir, f"{uniprot}.pkl"), "wb") as f:
            pickle.dump(emd, f)
    

if __name__ == "__main__":

    args = parse_args()
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL)
    if args.file_type.startswith("seq"):
        fasta_name = Path(os.path.basename(args.input)).stem
        output_dir = os.path.join('embeddings', fasta_name+"_Seq_")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if args.input.endswith(".fasta") or args.input.endswith(".fa") :
            with open(args.input) as f:
                records = SeqIO.parse(f, "fasta")
                records = list(records)
            for record in tqdm(records):
                emd = process_seq(client, record.seq)
                dump_emd(emd, record.id, output_dir)

        elif args.input.endswith(".csv") or args.input.endswith(".tsv"):
            df = pd.read_csv(args.input)
            seqs = df["Sequence"]
            uniprots = df["Uniprot"]
            for i, seq in enumerate(tqdm(seqs)):
                emd = process_seq(client, seq)
                dump_emd(emd, uniprots[i], output_dir)

    elif args.file_type == "pdb" and os.path.isdir(args.input):
        if os.path.isdir(args.input):
            output_dir = os.path.join("embeddings", os.path.basename(args.input))
            file_list = glob("*.pdb")
            for pdb in tqdm(file_list):
                uniprot = Path(os.path.base_name(pdb)).stem
                emd = process_pdb(client, pdb)
                dump_emd(emd, uniprot, output_dir)
