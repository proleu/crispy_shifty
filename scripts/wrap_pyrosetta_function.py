__author__ = "Brian Coventry, Philip Leung"
__copyright__ = None
__credits__ = ["Philip Leung", "Rosettacommons"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Philip Leung"
__email__ = "pleung@cs.washington.edu"
__status__ = "Prototype"
# Python standard library
import argparse, binascii, bz2, collections, json, os, sys
from typing import *  # TODO explicit imports

# 3rd party library imports
# Rosetta library imports
import pyrosetta
from pyrosetta.distributed import cluster
import pyrosetta.distributed.io as io
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import maybe_init, requires_init

"""
This script wraps a pyrosetta function and manages outputs in a manner similar to 
pyrosetta distributed. It takes three required arguments:
-k or --kwargs: a JSON-formatted string as the single argument on the command line,
read as a dict of keyword arguments.
-f or --function: the name of the function to be wrapped.
-d or --directory: the directory, the absolute path to the .py file, of the function.
This can be an empty string, in which case the script will just try to import the 
function from the current working directory and installed libraries.
"""


parser = argparse.ArgumentParser(
    description="Use to distribute a pyrosetta function on a distributed system."
)
# required arguments
parser.add_argument("-k", "--kwargs", type=str, default="", required=True)


# flags = "-out:level 300 -corrections::beta_nov16 true -holes:dalphaball /home/bcov/ppi/tutorial_build/main/source/external/DAlpahBall/DAlphaBall.gcc -indexed_structure_store:fragment_store /net/databases/VALL_clustered/connect_chains/ss_grouped_vall_helix_shortLoop.h5"



def main():
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        pass
    params = vars(parser.parse_args(sys.argv[1:]))
    print("Run will proceed with the following kwargs:")
    print(params)

    pyrosetta.distributed.maybe_init(**params)

    handle = str(binascii.b2a_hex(os.urandom(24)).decode("utf-8"))
    ppose = detail_design(None, **detail_kwargs)
    if ppose is not None:
        pose = io.to_pose(ppose)
        pdbstring = io.to_pdbstring(pose)
        remark = "REMARK PyRosettaCluster: "
        scores_dict = collections.OrderedDict(sorted(pose.scores.items()))
        pdbfile_data = json.dumps(
            {
                "instance": {},
                "metadata": {},
                "scores": scores_dict,
            }
        )
        pdbstring_data = pdbstring + os.linesep + remark + pdbfile_data
        output_file = f"{handle}.pdb.bz2"
        with open(output_file, "wb") as f:
            f.write(bz2.compress(str.encode(pdbstring_data)))
        with open(f"{handle}.json", "w+") as f:
            print(json.dumps(dict(pose.scores)), file=f)


if __name__ == "__main__":
    main()
