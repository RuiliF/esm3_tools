from Bio import SeqIO
import torch                                                                                                                                                                                                       
import torch.nn.functional as F
import argparse
import attr
from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
    ESM3_function_decoder_v0,
    ESM3_structure_decoder_v0,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain

from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig, ESMProteinTensor
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.sampling import get_default_sampling_config
from tqdm import tqdm
import os
import re

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", type=str, help="input fasta files")
    parser.add_argument("--mode", type=str, default="argmax")
    parser.add_argument("--none_threshold", type=float, default=0.05)
    parser.add_argument("--annotation_threshold", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", action='count', default=0)
    args = parser.parse_args()
    return args


def function_prediction_recursive(client, seq):
    
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)

    default_sampler = get_default_sampling_config(client.tokenizers)
    output = client.forward_and_sample(protein_tensor, default_sampler)
    function_token_ids = output.protein_tensor.function
    
    return function_token_ids

def function_prediction_single_pass(client, seq):
    tokenizer = EsmSequenceTokenizer()
    tokens = tokenizer.encode(seq)
    
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64)
    sequence_tokens = sequence_tokens.cuda().unsqueeze(0)

    output = client.forward(sequence_tokens=sequence_tokens)
    #output = client.forward(sequence_tokens=torch.tensor(tokens, dtype=torch.int64).cuda().unsqueeze(0))
    
    
    return output


if __name__ == "__main__":
    args = parse()

    if args.mode == "argmax":
        client = ESM3_sm_open_v0("cuda")
    else:
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device('cuda'))

    function_decoder = ESM3_function_decoder_v0("cuda")
    function_tokenizer = EsmFunctionTokenizer()

    if not os.path.isdir('output'):
      os.mkdir('output')
  
    output_dir = os.path.join('output', os.path.basename(args.fasta).split('.')[0] + '_' + args.mode)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for record in tqdm(list(SeqIO.parse(args.fasta, "fasta"))):
        # Parsing records in fasta files. May have errors depends on the description format
        output_name = os.path.join(output_dir, re.split("\W", record.id)[0] + ".txt")
        
        if os.path.isfile(output_name):
            continue
          
        # Sequence length threshold depends on the size of your GPU 
        if len(record.seq) > 1092:
            if args.verbose > 0:
                print(record.id + " too long to handle. Consider trimming this sequence.")
            continue

        #output = function_prediction(client, str(record.seq))
        # print(output.protein_tensor.function)
        seq = str(record.seq)

        if args.verbose > 0:
            print(seq)

        if args.mode == "argmax":

            output = function_prediction_single_pass(client, seq)
            
            # Function prediction
            p_none_threshold = args.none_threshold
            log_p = F.log_softmax(output.function_logits[:, 1:-1, :], dim=3).squeeze(0)

            # Choose which positions have no predicted function.
            log_p_nones = log_p[:, :, function_tokenizer.vocab_to_index["<none>"]]
            p_none = torch.exp(log_p_nones).mean(dim=1)  # "Ensemble of <none> predictions"
            where_none = p_none > p_none_threshold  # (length,)

            log_p[~where_none, :, function_tokenizer.vocab_to_index["<none>"]] = -torch.inf
            function_token_ids = torch.argmax(log_p, dim=2)
            function_token_ids[where_none, :] = function_tokenizer.vocab_to_index["<none>"]
        else:
            function_token_ids = function_prediction_recursive(client, seq)

            
        predicted_function = function_decoder.decode(
            function_token_ids,
            tokenizer=function_tokenizer,
            annotation_threshold=args.annotation_threshold,
            annotation_min_length=5,
            annotation_gap_merge_max=3,
        )

        with open(output_name, "w") as f:

            keywords = predicted_function["function_keywords"]
        
            for entry in keywords:
                if args.verbose > 0:
                    print(entry)
                f.write(str(entry))
                f.write("\n")

    os.system(f"tar -cvzf {output_dir}.tar.gz {output_dir}")
    os.system(f"rm -rf {output_dir}")
