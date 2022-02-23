#!/usr/bin/env python

import pyrosetta
pyrosetta.distributed.maybe_init(**{
    "options": "-corrections::beta_nov16 true",
    "extra_options":{
        "-out:level": "100",
}})

import pyrosetta.distributed.io as io

import sys
sys.path.insert(0, '/home/broerman/projects/crispy_shifty/')
from crispy_shifty.protocols.mpnn import mpnn_hinge_dimers, MPNNDesign

input_fname = sys.argv[1]

ppose_in = io.pose_from_file(input_fname)

pposes = mpnn_hinge_dimers(
    packed_pose_in = ppose_in,
    num_sequences = 10,
    batch_size = 10,
    **{
        "mpnn_design_area": 'scan'
    }
)

# iterate first since mpnn_hinge_dimers returns a generator
poses = []
for ppose in pposes:
    poses.append(io.to_pose(ppose))

output_split = input_fname[:-4].split('/')
output_split[-2] = 'mpnn_designs'
output_prefix = '/'.join(output_split)

# then once all sequences are generated, build the structures
mpnn_design = MPNNDesign()
for design_area, pose in zip(['full', 'interface', 'neighborhood'], poses):
    for j, mpnn_pose in enumerate(mpnn_design.generate_all_poses(pose)):
        mpnn_pose.dump_pdb(f'{output_prefix}_{design_area}_{j}.pdb')
