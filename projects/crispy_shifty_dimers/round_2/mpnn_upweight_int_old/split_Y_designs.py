#!/usr/bin/env python

import sys

import pyrosetta
pyrosetta.distributed.maybe_init(**{
    "options": "-corrections::beta_nov16 true",
    "extra_options":{
        "-out:level": "100",
}})

sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
sw.chain_order('34')

for mpnn_fname in sys.argv[1:]:

    pose = pyrosetta.pose_from_pdb(mpnn_fname)
    sw.apply(pose)
    fname_split = mpnn_fname[:-4].split('/')
    fname_split[-2] = 'unique_mpnn_Y_designs'
    pose.dump_pdb(f"{'/'.join(fname_split)}_AB_Y.pdb")
