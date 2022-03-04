
tag_256 = "AAATCCCAAAGCCATACCCTAACCTTAATTCAATGCAACTCATTTATATCTATGTAAGTCAGACACATGGAAACGCACTTTTGAACAGTTAGATAGCTACTGATTGCCCCGAAGGCAGGTACGTAGGGACCGTTCTCTGTCCTCCGGAGAGTGAGCGATCGACGGTTGGCTTCGCTGGGTGTGCGTCGGCGCCGCGGGGCCTGCTCGTGGTCTTGTTTCCACCAGCATCACGAGGATGACTAGTATTACAAGAATA"
tag_64 = "AATCCAAGCATTACCTATGAACTTTGCCCGACGTAGTCTCGCGGAGGTGGGCTGTTCACAGATA"

def capture_1shot_domesticator(stdout: str) -> str:
    """split input into lines.
    loop once through discarding lines up to ones including >.
    return joined output"""
    sequence = []
    append = False
    for line in stdout.splitlines():
        if append:
            sequence.append(line)
        else:
            pass
        if ">unknown_seq1" in line:
            append = True
        else:
            pass
    to_return = "".join(sequence)
    return to_return
