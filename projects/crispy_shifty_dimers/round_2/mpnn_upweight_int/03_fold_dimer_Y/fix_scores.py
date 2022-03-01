#!/usr/bin/env python

import bz2
from glob import glob
import pandas as pd
import pyrosetta
import pyrosetta.distributed.cluster as cluster
import sys

pyrosetta.distributed.maybe_init(**{
    "options": "-corrections::beta_nov16 true",
    "extra_options":{
        "-out:level": "100",
}})

# mpnn scores were accidentally not included in the folding function
# get them from the original files

out_dir = sys.argv[1]
out_files = sorted(glob("/home/broerman/crispy_shifty/projects/crispy_shifty_dimers/round_2/mpnn_upweight_int/03_fold_dimer_Y/decoys/" + out_dir + '/*.pdb.bz2'))
indices = [index.replace("/home/broerman/crispy_shifty/projects/crispy_shifty_dimers/round_2/mpnn_upweight_int", "/pscratch/sd/b/broerman") for index in out_files]

mpnn_scores_df = pd.DataFrame(index=indices, columns=['designed_by', 'mpnn_msd_design_area', 'mpnn_msd_temperature'])

for index in indices:
    folded_path = index.replace("/pscratch/sd/b/broerman", "/home/broerman/crispy_shifty/projects/crispy_shifty_dimers/round_2/mpnn_upweight_int")
    input_path = pyrosetta.distributed.cluster.get_scores_dict(folded_path)["instance"]["tasks"]["pdb_path"].split('____')[0].replace("/global/u2/b/broerman/projects/CSD", "/home/broerman/crispy_shifty/projects/crispy_shifty_dimers")
    input_metadata = pyrosetta.distributed.cluster.get_scores_dict(input_path)
    parent, combo = input_metadata["instance"]["tasks"]["pdb_path"][:-4].split('/')[-1].split('_')
    mpnn_scores_df.loc[index, "parent"] = parent
    mpnn_scores_df.loc[index, "combo"] = combo
    mpnn_scores = input_metadata["scores"]
    mpnn_scores_df.loc[index, 'mpnn_msd_design_area'] = mpnn_scores["mpnn_msd_design_area"]
    mpnn_scores_df.loc[index, 'mpnn_msd_temperature'] = mpnn_scores["mpnn_msd_temperature"]

    folded_pose = pyrosetta.rosetta.core.pose.Pose()
    with open(folded_path, "rb") as f:
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(folded_pose, bz2.decompress(f.read()).decode())
    folded_seq_1 = folded_pose.chain_sequence(1)
    folded_seq_2 = folded_pose.chain_sequence(2)

    for mpnn_seq_num in range(9):
        mpnn_seq_id = f"mpnn_seq_{mpnn_seq_num:04d}"
        mpnn_seq_split = mpnn_scores[mpnn_seq_id].split('/')
        if mpnn_seq_split[0] == folded_seq_1 and mpnn_seq_split[1] == folded_seq_2:
            mpnn_scores_df.loc[index, 'mpnn_seq_id'] = mpnn_seq_id
            if mpnn_seq_id == "mpnn_seq_0000":
                mpnn_scores_df.loc[index, 'designed_by'] = "rosetta"
            else:
                mpnn_scores_df.loc[index, 'designed_by'] = "mpnn"
            break
        if mpnn_seq_num == 8:
            raise Exception(f"Could not find mpnn_seq_id for {index}")

    # if mpnn_scores["mpnn_seq_0000"].startswith(folded_seq_1):
    #     mpnn_scores_df.loc[index, 'designed_by'] = "rosetta"
    #     # print('rosetta!')
    # else:
    #     mpnn_scores_df.loc[index, 'designed_by'] = "mpnn"

mpnn_scores_df.to_csv(f"/home/broerman/crispy_shifty/projects/crispy_shifty_dimers/round_2/mpnn_upweight_int/03_fold_dimer_Y/fixed_scores/scores_{out_dir}.csv")
