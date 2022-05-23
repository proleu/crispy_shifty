# Python standard library
from typing import Iterator, Optional

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports

@requires_init
def mpnn_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    from itertools import product
    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NeighborhoodResidueSelector,
        TrueResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
    from crispy_shifty.protocols.mpnn import MPNNDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    if "mpnn_temperature" in kwargs:
        if kwargs["mpnn_temperature"] == "scan":
            mpnn_temperatures = [0.1, 0.2, 0.5]
        else:
            mpnn_temperature = float(kwargs["mpnn_temperature"])
            assert (
                0.0 <= mpnn_temperature <= 1.0
            ), "mpnn_temperature must be between 0 and 1"
            mpnn_temperatures = [mpnn_temperature]
    else:
        mpnn_temperatures = [0.1]
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    interface_selector = AndResidueSelector(interface_between_selectors(chA, chB), chA)
    neighborhood_selector = AndResidueSelector(
        NeighborhoodResidueSelector(
            interface_selector, distance=8.0, include_focus_in_subset=True
        ),
        chA,
    )
    full_selector = chA
    selector_options = {
        "full": full_selector,
        "interface": interface_selector,
        "neighborhood": neighborhood_selector,
    }
    # make the inverse dict of selector options
    selector_inverse_options = {value: key for key, value in selector_options.items()}
    if "mpnn_design_area" in kwargs:
        if kwargs["mpnn_design_area"] == "scan":
            mpnn_design_areas = [
                selector_options[key] for key in ["full", "interface", "neighborhood"]
            ]
        else:
            try:
                mpnn_design_areas = [selector_options[kwargs["mpnn_design_area"]]]
            except:
                raise ValueError(
                    "mpnn_design_area must be one of the following: full, interface, neighborhood"
                )
    else:
        mpnn_design_areas = [selector_options["interface"]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(product(mpnn_temperatures, mpnn_design_areas))
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area) in enumerate(list(mpnn_conditions)):
            pose = original_pose.clone()
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )
            print_timestamp("Designing interface with MPNN", start_time)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=mpnn_design_area,
                omit_AAs="X",
                temperature=mpnn_temperature,
                **kwargs,
            )
            # design the pose
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_temperature": mpnn_temperature,
                    "mpnn_design_area": selector_inverse_options[mpnn_design_area],
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def fold_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the superfold script.
    :param: kwargs: keyword arguments to be passed to the superfold script.
    :return: an iterator of PackedPose objects.
    """

    from operator import lt, gt
    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.folding import generate_decoys_from_pose, SuperfoldRunner
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # hacky split pdb_path into pdb_path and fasta_path
    pdb_path = kwargs.pop("pdb_path")
    pdb_path, fasta_path = tuple(pdb_path.split("____"))

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        # skip the kwargs check
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up for AF2", start_time)
        runner = SuperfoldRunner(
            pose=pose, fasta_path=fasta_path, load_decoys=True, **kwargs
        )
        runner.setup_runner(file=fasta_path)
        # initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        flag_update = {
            "--initial_guess": initial_guess,
            "--reference_pdb": reference_pdb,
        }
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # setup prefix, rank_on, filter_dict (in this case we can't get from kwargs)
        filter_dict = {
            "mean_plddt": (gt, 90.0),
        }
        rank_on = "mean_plddt"
        prefix = "mpnn_seq"
        print_timestamp("Generating decoys", start_time)
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=True,
            label_first=True,
            prefix=prefix,
            rank_on=rank_on,
        ):
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy

@requires_init
def fold_unbound(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the Superfold script.
    :param: kwargs: keyword arguments to be passed to the Superfold script.
    :return: an iterator of PackedPose objects.
    """

    import os
    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # get the pdb_path from the kwargs
    pdb_path = kwargs.pop("pdb_path")
    # there are multiple paths in the pdb_path, we need to split them and rejoin them
    pdb_paths = pdb_path.split("____")
    pdb_path = " ".join(pdb_paths)

    # this function is special, we don't want a packed_pose_in ever, we maintain it as
    # a kwarg for backward compatibility with PyRosettaCluster
    if packed_pose_in is not None:
        raise ValueError("This function is not intended to have a packed_pose_in")
    else:
        pass

    print_timestamp("Setting up for AF2", start_time)
    runner = SuperfoldMultiPDB(input_file=pdb_path, load_decoys=True, **kwargs)
    runner.setup_runner(chains_to_keep=[1])
    print_timestamp("Running AF2", start_time)
    runner.apply()
    print_timestamp("AF2 complete, updating pose datacache", start_time)
    # get the updated poses from the runner
    tag_pose_dict = runner.get_tag_pose_dict()
    # filter the decoys
    filter_dict = {
        "mean_plddt": (gt, 90.0),
        "rmsd_to_input": (lt, 2.0),
    }
    rank_on = "mean_plddt"
    print_timestamp("Generating decoys", start_time)
    for tag, pose in tag_pose_dict.items():
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=False,
            label_first=False,
            prefix=tag,
            rank_on=rank_on,
        ):
            scores = dict(decoy.scores)
            bound_pose = None
            for original_path in pdb_paths:
                if tag in original_path:
                    bound_pose = next(
                        path_to_pose_or_ppose(
                            path=original_path, cluster_scores=True, pack_result=False
                        )
                    )
                    final_pose = Pose()
                    break
                else:
                    continue
            if bound_pose is None:
                raise RuntimeError
            else:
                pass
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(bound_pose, key, value)
            final_ppose = io.to_packed(final_pose)
            yield final_ppose

