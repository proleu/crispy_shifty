from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from typing import *


@requires_init
def path_to_pose_or_ppose(
    path: "", cluster_scores=False, pack_result=False
) -> Generator[str, Union[PackedPose, Pose], None]:
    """
    Generate PackedPose objects given an input path to a file on disk to read in.
    Can do pdb, pdb.bz2, pdb.gz or binary silent file formats.
    To use silents, must initialize Rosetta with "-in:file:silent_struct_type binary".
    Does not save scores unless reading in pdb.bz2 and cluster_scores is set to true.
    This function can be distributed (best for single inputs) or run on a host process
    """
    import bz2
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed import cluster

    if ".silent" in path:
        pposes = io.poses_from_silent(path)  # returns a generator
    elif ".bz2" in path:
        with open(path, "rb") as f:  # read bz2 bytestream, decompress and decode
            ppose = io.pose_from_pdbstring(bz2.decompress(f.read()).decode())
        if cluster_scores:  # set scores in pose after unpacking, then repack
            scores = pyrosetta.distributed.cluster.get_scores_dict(path)["scores"]
            pose = io.to_pose(ppose)
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(value))
            ppose = io.to_packed(pose)
        else:
            pass
        pposes = [ppose]
    elif ".pdb" in path:  # should handle pdb.gz as well
        pposes = [io.pose_from_file(path)]
    else:
        raise RuntimeError("Must provide a pdb, pdb.gz, pdb.bz2, or binary silent")
    for ppose in pposes:
        if pack_result:
            yield ppose
        else:
            pose = io.to_pose(ppose)
            yield pose


@requires_init
def remove_terminal_loops(packed_pose_in=None, **kwargs) -> List[PackedPose]:
    """
    Use DSSP and delete region mover to idealize inputs. Add metadata.
    Assumes a monomer. Must provide either a packed_pose_in or "pdb_path" kwarg.
    """

    import pyrosetta
    import pyrosetta.distributed.io as io
    import sys

    # TODO import crispy shifty module
    sys.path.append("/home/pleung/projects/crispy_shifty")
    from protocols.cleaning import path_to_pose_or_ppose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=False, pack_result=False
        )
    final_pposes = []
    for pose in poses:
        # get secondary structure
        pyrosetta.rosetta.core.scoring.dssp.Dssp(pose).insert_ss_into_pose(pose, True)
        dssp = pose.secstruct()
        # get leading loop from ss
        if dssp[0] == "H":  # in case no leading loop is detected
            py_idx_n_term = 0
        else:  # get beginning index of first occurrence of LH in dssp
            py_idx_n_term = dssp.find("LH")
        # get trailing loop from ss
        if dssp[-1] == "H":  # in case no trailing loop is detected
            py_idx_c_term = -1
        else:  # get ending index of last occurrence of HL in dssp
            py_idx_c_term = dssp.rfind("HL") + 1
        rosetta_idx_n_term, rosetta_idx_c_term = str(py_idx_n_term + 1), str(
            py_idx_c_term + 1
        )
        trimmed_pose = pose.clone()
        # setup trimming mover
        trimmer = pyrosetta.rosetta.protocols.grafting.simple_movers.KeepRegionMover()
        trimmer.start(rosetta_idx_n_term)
        trimmer.end(rosetta_idx_c_term)
        trimmer.apply(trimmed_pose)
        # setup rechain mover to sanitize the trimmed pose
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        rechain.chain_order("1")
        rechain.apply(trimmed_pose)
        trimmed_length = len(trimmed_pose.residues)
        if "metadata" in kwargs:
            metadata = kwargs["metadata"]
        else:
            metadata = {}
        metadata["trimmed_length"] = str(trimmed_length)
        for key, value in metadata.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(value))
        final_pposes.append(io.to_packed(pose))
    return final_pposes
