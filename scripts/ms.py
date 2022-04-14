#!/projects/crispy_shifty/envs/crispy/bin/python

import argparse 
from itertools import combinations
import sys

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

# TODO there is a bug in this script that makes it not always return all the possible closely matching fragments
# TODO example: -s MSSGTEDERRELEKVARKAIEAAREGNTDEVREQLQRALEIARESGTEEAYLLALEVVMRVLCEAIRRGNKDAAKLAAKVAKEIAKQGHTKSDWLTALRELAEKACEAARQGNLEAVRKVLEELLELAREAGTEEAVKLALKAVELVSRVAKKQGNEDAVKEAEEVRKKIEEESGTGSHHHHHH -p 14306.29 
# TODO returns 
# 14305.23 Da, difference 1.06 : ARKAIEAAREGNTDEVREQLQRALEIARESGTEEAYLLALEVVMRVLCEAIRRGNKDAAKLAAKVAKEIAKQGHTKSDWLTALRELAEKACEAARQGNLEAVRKVLEELLELAREAGTEEAVKLALKAVE
# 14304.29 Da, difference 2.0 : KVARKAIEAAREGNTDEVREQLQRALEIARESGTEEAYLLALEVVMRVLCEAIRRGNKDAAKLAAKVAKEIAKQGHTKSDWLTALRELAEKACEAARQGNLEAVRKVLEELLELAREAGTEEAVKLALKA
# TODO should return 14308.11 Da, difference -1.82 : EDERRELEKVARKAIEAAREGNTDEVREQLQRALEIARESGTEEAYLLALEVVMRVLCEAIRRGNKDAAKLAAKVAKEIAKQGHTKSDWLTALRELAEKACEAARQGNLEAVRKVLEELLELAREAGT



parser = argparse.ArgumentParser(description="See if a protein has a sequence or subsequence that matches a mass spec peak")
# required arguments
parser.add_argument("-s", "--sequence", help="input sequence", required=True)
parser.add_argument("-p", "--peak", help="peak mw", required=True, type=float)

def main():
    """
    Calculate the molecular weight of a protein sequence and putative fragments

    """
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        pass
    args = parser.parse_args(sys.argv[1:])
    args_dict = vars(args)
    sequence = args_dict["sequence"]
    peak = args_dict["peak"]
    closest_match = ProteinAnalysis(sequence).molecular_weight()
    closest_match_subsequence = sequence
    matches = []
    matches.append((round(closest_match, 2), closest_match_subsequence, round(float(peak)-closest_match, 2)))
    # first establish a minimum subsequence length
    # 110.0 is the average molecular weight of an amino acid
    # 1.2 is a fudge factor in case the input sequence has a lot of small amino acids
    # TODO dynamically average the average amino acid mass of the input sequence
    min_subseq_length = int(peak / (110.0 * 1.2))
    if closest_match > float(peak):
        difference = abs(float(closest_match) - float(peak))
        # start making subsequences from both ends of the sequence
        # and see if the weight of the subsequence is closer to the peak
        # loop over subsequences of length-1, length-2, ..., min_subseq_length
        subsequence_lengths = list(reversed(range(min_subseq_length, int(len(sequence)+1))))
        for subsequence_length in tqdm(subsequence_lengths):
            all_k_subsequences = [sequence[x:y] for x, y in combinations(range(len(sequence)+1), r=2) if len(sequence[x:y]) == subsequence_length]
            for subsequence in all_k_subsequences:
                subsequence_weight = ProteinAnalysis(subsequence).molecular_weight()
                if abs(subsequence_weight - float(peak)) < difference:
                    difference = abs(subsequence_weight - float(peak))
                    closest_match = subsequence_weight
                    closest_match_subsequence = subsequence
                    matches.append((round(closest_match, 2), closest_match_subsequence, round(peak-closest_match, 2)))

    else:
        raise NotImplementedError


    delta = round(closest_match - peak, 2)
    # print top 10 matches within 3 Da of the peak
    # TODO just all within 3 Da of the peak?
    TO_REPORT = 10
    i = 0
    for i, match in enumerate(reversed(matches)):
        if i < TO_REPORT and abs(match[0] - float(peak)) < 3.0:
            print(f"{match[0]} Da, difference {match[2]} : {match[1]}")
        else:
            break


if __name__ == "__main__":
    main()
