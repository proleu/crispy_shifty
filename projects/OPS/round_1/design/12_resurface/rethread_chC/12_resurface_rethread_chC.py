#!/usr/bin/env python

import bz2
import json
import os
import sys
import pyrosetta
import pyrosetta.distributed.io as io
from pyrosetta.distributed import cluster

pyrosetta.distributed.maybe_init(
    **{
        "options": "-corrections::beta_nov16 true",
        "extra_options": {"-out:level": "100"},
    }
)

output_path = "/home/broerman/crispy_shifty/projects/OPS/round_1/design/12_resurface"

for fname in sys.argv[1:]:
    with open(fname, "rb") as f:  # read bz2 bytestream, decompress and decode
        decoy = io.to_pose(io.pose_from_pdbstring(bz2.decompress(f.read()).decode()))
    scores = pyrosetta.distributed.cluster.get_scores_dict(fname)
    for key, value in scores["scores"].items():
        pyrosetta.rosetta.core.pose.setPoseExtraScore(decoy, key, value)

    # get the chA sequence
    chA_seq = decoy.chain_sequence(1)
    # setup SimpleThreadingMover
    stm = pyrosetta.rosetta.protocols.simple_moves.SimpleThreadingMover()
    # thread the sequence from chA onto chA
    stm.set_sequence(chA_seq, start_position=decoy.chain_begin(3))
    stm.apply(decoy)

    out_fname = fname.replace("decoys_old_chC", "decoys")

    pdbfile_data = json.dumps(scores)
    # Write full .pdb record
    pdbstring_data = (
        io.to_pdbstring(decoy) + os.linesep + "REMARK PyRosettaCluster: " + pdbfile_data
    )
    with open(out_fname, "wb") as f:
        f.write(bz2.compress(str.encode(pdbstring_data)))
