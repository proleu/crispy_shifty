# Python standard library
from typing import *
# 3rd party library imports
import pandas as pd
from tqdm import tqdm
# Rosetta library imports
# Custom library imports

def parse_scorefile_oneshot(scores:str) -> pd.DataFrame:
    """
    Read in a scores.json from PyRosettaCluster in a single shot.
    Memory intensive for a larger scorefile because it does a matrix transposition.
    """
    import pandas as pd


    scores = pd.read_json(scores, orient="records", typ="frame", lines=True)
    scores = scores.T
    mat = scores.values
    n = mat.shape[0]
    dicts = list(mat[range(n), range(n)])
    index = scores.index
    tabulated_scores = pd.DataFrame(dicts, index=index)
    return tabulated_scores


def parse_scorefile_linear(scores:str) -> pd.DataFrame:
    """
    Read in a scores.json from PyRosettaCluster line by line.
    Uses less memory thant the oneshot method but takes longer to run.
    """
    import pandas as pd
    from tqdm import tqdm

    dfs = []
    with open(scores, "r") as f:
        for line in tqdm(f):
            dfs.append(pd.read_json(line).T)
    tabulated_scores = pd.concat(dfs)
    return tabulated_scores
