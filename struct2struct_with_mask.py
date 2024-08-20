# Generating predicted structure from a template structure (pdb) file with masks

from Bio import SeqIO
import torch     
import numpy as np
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

from esm.tokenization import (
    TokenizerCollectionProtocol,
    get_model_tokenizers,
)

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig, ESMProteinTensor, GenerationConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.sampling import get_default_sampling_config
from esm.utils.decoding import decode_sequence, decode_structure
from tqdm import tqdm
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb", type=str, help="input fasta files")
    parser.add_argument("--mode", type=str, default="recursive", help="choose between recursive or single_pass")
    parser.add_argument("--masks", action="append")
    #parser.add_argument("--annotation_threshold", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("-v", "--verbose", action='count', default=0)
    parser.add_argument("--schedule", type=str, default="cosine", help="noise scheduling; choose from cosine, linear, square_root_schedule, cubic, square")
    parser.add_argument("--use_coordinates", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--first_track", type=str, default="sequence", help="choose which track to generate first; choose between 'sequence' or 'structure'")
    args = parser.parse_args()
    return args

@torch.no_grad()
def decode(sequence, output, sequence_tokens, args):
    # To save on VRAM, we load these in separate functions
    decoder = ESM3_structure_decoder_v0("cuda")
    function_decoder = ESM3_function_decoder_v0("cuda")
    function_tokenizer = EsmFunctionTokenizer()

    structure_tokens = torch.argmax(output.structure_logits, dim=-1)
    structure_tokens = (
        structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
        .where(sequence_tokens != 2, 4097)  # EOS
        .where(sequence_tokens != 31, 4100)  # Chainbreak
    )

    bb_coords = (
        decoder.decode(
            structure_tokens,
            torch.ones_like(sequence_tokens),
            torch.zeros_like(sequence_tokens),
        )["bb_pred"]
        .detach()
        .cpu()
    )

    chain = ProteinChain.from_backbone_atom_coordinates(
        bb_coords, sequence="X" + sequence + "X"
    )
    chain.infer_oxygen().to_pdb("hello.pdb")




if __name__ == "__main__":
    args = parse()
    
    print("++++ Input PDB +++")
    print(args.pdb)
    print("++++ MASKS ++++")
    print(args.masks)
    print("++++ Num of Steps ++++")
    print(args.steps)
    if args.steps == -1:
        print("number of masked tokens")
    print("++++ Noise Schedule ++++")
    print(args.schedule)
    print("++++ Temperature ++++")
    print(args.temp)
    print("++++ Generating Order ++++")
    if args.first_track == "sequence":
        tracks = ["sequence", "structure"]
    else:
        tracks = ["structure", "sequence"]
    print(tracks)

    #sequence_tokenizer = EsmSequenceTokenizer()
    encoder = ESM3_structure_encoder_v0("cuda")
    model = ESM3.from_pretrained("esm3_sm_open_v1") #ESM3_sm_open_v0("cuda")
    tokenizers = get_model_tokenizers("esm3_sm_open_v1")




    chain = ProteinChain.from_pdb(args.pdb)
    sequence = chain.sequence
    sequence_tokens = torch.tensor(tokenizers.sequence.encode(sequence), dtype=torch.int64).unsqueeze(0).cuda()
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cuda()
    plddt = plddt.cuda()
    residue_index = residue_index.cuda()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)


    # Add BOS/EOS padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    # Adding Mask
    for mask in args.masks:
        mask_index = mask.split(',')
        mask_index = [int(index) for index in mask_index]
        sequence_tokens[:, mask_index[0]+1: mask_index[1]+1] = 32
        structure_tokens[:, mask_index[0]+1: mask_index[1]+1] = 4096
        if args.use_coordinates == True:
            coords[:, mask_index[0]+1:mask_index[1]+1, :, :] = np.nan
    num_mask = (sequence_tokens == 32).sum()

    #inputP = ESMProtein.from_pdb(args.pdb)
    #print(inputP)

    if args.mode == "single_pass":
        output = model.forward(
            sequence_tokens=sequence_tokens, # structure_coords=coords, per_res_plddt=plddt, 
            structure_tokens=structure_tokens
        )
        gen_protein = decode(sequence, output, sequence_tokens, args)
    elif args.mode == "recursive":
        protein_tensor = ESMProteinTensor()
        protein_tensor.sequence = sequence_tokens.squeeze()
        protein_tensor.structure = structure_tokens.squeeze()
        coords = coords.squeeze()
        if args.use_coordinates == True:
            protein_tensor.coordinates = coords
        if args.steps == -1:
            steps = num_mask
        else:
            steps = args.steps
        for track in tracks:
            config = GenerationConfig(track=track, num_steps=steps, temperature=args.temp, schedule=args.schedule)
            protein_tensor = model.generate(protein_tensor, config)
        output = protein_tensor
        
        # decode output
    
        gen_sequence = decode_sequence(output.sequence, tokenizers.sequence)
        structure_token_decoder = ESM3_structure_decoder_v0("cuda")
        gen_coordinates, gen_plddt, gen_ptm = decode_structure(
            structure_tokens=output.structure,
            structure_decoder=structure_token_decoder,
            structure_tokenizer=tokenizers.structure,
            sequence=gen_sequence,
        )
        gen_protein = ESMProtein(sequence=gen_sequence, coordinates=gen_coordinates, plddt=gen_plddt, ptm=gen_ptm)
    else:
        print("Unrecognized mode. Please choose between 'recursive' and 'single_pass'.")
        exit()
  
    gen_path = os.path.join("output_struct_gen", args.pdb[:-4]+ "_generated.pdb")
    gen_protein.to_pdb(gen_path)
    print("Generated protein pdb:", gen_path)
