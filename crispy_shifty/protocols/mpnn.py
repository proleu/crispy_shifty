# Python standard library
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


class MPNNRunner(ABC):
    """
    Abstract base class for MPNN runners.
    """
    import pyrosetta.distributed.io as io

    def __init__(
        self,
        pose: Pose,
        residue_selector: Optional[ResidueSelector] = None,
        batch_size: int = 8,
        num_sequences: int = 64,
        temperature: float = 0.1,
    ):
        """
        Initialize the base class for MPNN runners with common attributes.
        """
        self.pose = io.to_pose(pose)
        self.residue_selector = residue_selector
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.temperature = temperature
        # setup standard command line flags for MPNN
        self.flags = [
            "--backbone_noise 0.05 ",
            "--checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch51_step255000.pt' ",
            "--decoding_order 'random' ",
            "--hidden_dim 192 ",
            "--max_length 10000 ",
            "--num_connections 64 ",
            "--num_layers 3 ",
            "--protein_features 'full' ",
        ]
                                                    


    @abstractmethod
    def run(self) -> None:
        """
        This function needs to be implemented by the child class of MPNNRunner.
        """
        pass

def run_mpnn(
    pose: Pose,
    residue_selector: Optional[ResidueSelector] = None,
    num_iterations: int = 64,
    batch_size: int = 8,
) -> Iterator[Pose]:    

    """
    :param: pose: Pose to run MPNN on.
    :param: residue_selector: Residue selector to use for MPNN.
    :param: num_iterations: Number of sequences to generate.
    :param: batch_size: Number of sequences to generate at a time.
    :return: Iterator of poses generated by MPNN. TODO
    Runs MPNN on a pose. Manages file I/O. If a residue selector is provided, will only
    run MPNN on the selected residues.
    """

    import binascii, os
    import pyrosetta.distributed.io as io

    # use TMPDIR if os.environ['TMPDIR'] is set
    if 'TMPDIR' in os.environ:
        tmpdir = os.environ['TMPDIR']
    else:
        tmpdir = os.getcwd()

    # dump a clean pdbstring of the pose as a temp pdb

    # make a jsonl from the pdb and delete the pdb

    # make masked


def thread_mpnn_sequence(
    pose: Pose, 
    sequence: str,
    start_res: int=1,
) -> Pose:
    """
    :param: pose: Pose to thread sequence onto.
    :param: sequence: Sequence to thread onto pose.
    :return: Pose with threaded sequence.
    Threads an MPNN sequence onto a pose after cloning the input pose.
    Doesn't require the chainbreak '/' to be cleaned.
    """
    from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover

    try:
        assert sequence.count("/") == 0
    except AssertionError:
        print("Cleaning chain breaks in sequence.")
        sequence = sequence.replace("/", "")

    pose = pose.clone()
    stm = SimpleThreadingMover()
    stm.set_sequence(sequence, start_res)
    stm.apply(pose)

    return pose



@requires_init
def mpnn_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to looping protocol.
    :return: an iterator of PackedPose objects.
    Assumes that pyrosetta.init() has been called with `-corrections:beta_nov16` .
    """

    from copy import deepcopy
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io

    sys.path.insert(0, "/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        clear_constraints,
        clear_terms_from_scores,
        gen_std_layer_design,
        gen_task_factory,
        packrotamers,
        score_ss_sc,
        score_wnm,
        struct_profile,
    )
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

    looped_poses = []
    sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    for pose in poses:
        scores = dict(pose.scores)
        # get parent length from the scores
        parent_length = int(float(scores["trimmed_length"]))
        looped_poses = []
        sw.chain_order("123")
        sw.apply(pose)
        min_loop_length = int(parent_length - pose.chain_end(2))
        max_loop_length = 5
        loop_start = pose.chain_end(1) + 1
        pre_looped_length = pose.chain_end(2)
        print_timestamp("Generating loop extension...", start_time, end="")
        closure_type = loop_extend(pose=pose, min_loop_length=min_loop_length, max_loop_length=max_loop_length, connections="[A+B],C")
        if closure_type == "not_closed":
            continue  # move on to next pose, we don't care about the ones that aren't closed
        else:
            sw.chain_order("12")
            sw.apply(pose)
            # get new loop resis
            new_loop_length = pose.chain_end(1) - pre_looped_length
            new_loop_str = ",".join(
                [str(i) for i in range(loop_start, loop_start + new_loop_length)]
            )
            scores["new_loop_str"] = new_loop_str
            scores["looped_length"] = pose.chain_end(1) 
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            looped_poses.append(pose)

    # hardcode precompute_ig
    pyrosetta.rosetta.basic.options.set_boolean_option("packing:precompute_ig", True)
    layer_design = gen_std_layer_design()
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
    )

    for looped_pose in looped_poses:
        scores = dict(looped_pose.scores)
        new_loop_str = scores["new_loop_str"]
        print_timestamp("Designing loop...", start_time, end="")
        new_loop_sel = (
            pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                new_loop_str
            )
        )
        design_sel = (
            pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
                new_loop_sel, 8, True
            )
        )
        task_factory = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=2,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=False,
            restrict_pro_gly=False,
            ifcl=True,  
        )
        struct_profile(
            looped_pose,
            design_sel,
        )
        packrotamers(
            looped_pose,
            task_factory,
            design_sfxn,
        )
        clear_constraints(looped_pose)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(looped_pose)
        total_length = looped_pose.total_residue()
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            looped_pose, "total_length", total_length
        )
        dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(
            looped_pose
        )
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            looped_pose, "dssp", dssp.get_dssp_secstruct()
        )
        score_ss_sc(looped_pose, False, True, "loop_sc")
        scores.update(looped_pose.scores)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, key, value)
        clear_terms_from_scores(looped_pose)
        ppose = io.to_packed(looped_pose)
        yield ppose

