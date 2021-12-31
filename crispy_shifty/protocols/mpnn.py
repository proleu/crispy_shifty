# Python standard library
from typing import Iterator, List, Optional, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


def run_mpnn(
    pose: Union[Pose, PackedPose],
    residue_selector: Optional[ResidueSelector] = None,
    num_iterations: int = 10,
    num_threads: int = 1,
    verbose: bool = False,
) -> Iterator[Pose]:    

    """
    Runs MPNN on a pose.
    """

    import pyrosetta.distributed.io as io



    


def grow_terminal_helices(
    pose: Pose,
    chain: int,
    extend_c_term: int = 0,
    extend_n_term: int = 0,
    idealize: bool = False,
) -> Pose:
    """
    :param: extend_c_term: Number of residues to extend the C-terminal helix.
    :param: extend_n_term: Number of residues to extend the N-terminal helix.
    :param: pose: Pose to extend the terminal helices of.
    :param: chain: Chain to extend the terminal helices of. Needs to be > 7 residues.
    :return: Pose with the terminal helices extended.
    Extend the terminal helices of a pose by a specified number of residues.
    Mutates the first and last residues of the specified chain to VAL prior to extending.
    """

    import pyrosetta
    from pyrosetta.rosetta.core.pose import append_pose_to_pose, Pose

    # Get the chains of the pose
    chains = list(pose.split_by_chain())
    # PyRosetta indexing starts at 1 for chains
    chain_to_extend = chains[chain - 1].clone()
    try:
        assert chain_to_extend.total_residue() > 7
    except AssertionError:
        raise ValueError("Chain to extend must be > 7 residues.")

    # Build a backbone stub for each termini that will be extended.
    ideal_c_term = pyrosetta.pose_from_sequence("V" * (extend_c_term + 7))
    ideal_n_term = pyrosetta.pose_from_sequence("V" * (extend_n_term + 7))
    # Set the torsions of the stubs to be ideal helices.
    for stub in [ideal_c_term, ideal_n_term]:
        for i in range(1, stub.total_residue()):
            stub.set_phi(i, -60)
            stub.set_psi(i, -60)
            stub.set_omega(i, 180)
    # For each non-zero terminal extension, align the ideal helix to the pose at the termini.
    if extend_c_term > 0:
        # align first 7 residues of ideal_c_term to the last 7 residues of chain_to_extend
        range_CA_align(
            pose_a=ideal_c_term,
            pose_b=chain_to_extend,
            start_a=1,
            end_a=7,
            start_b=chain_to_extend.chain_end(1) - 6,
            end_b=chain_to_extend.chain_end(1),
        )
        # build a new chain by appending the ideal_c_term to the chain_to_extend
        extended_c_term = Pose()
        # first add the chain_to_extend into the extended_c_term but without the C-terminal residue
        for i in range(chain_to_extend.chain_begin(1), chain_to_extend.chain_end(1)):
            extended_c_term.append_residue_by_bond(chain_to_extend.residue(i))
        # append the ideal_c_term to the extended_c_term plus one additional residue
        for i in range(ideal_c_term.chain_begin(1) + 6, ideal_c_term.chain_end(1) + 1):
            extended_c_term.append_residue_by_bond(ideal_c_term.residue(i))
        new_pose = Pose()
        for i, subpose in enumerate(chains, start=1):
            if i == chain:
                append_pose_to_pose(new_pose, extended_c_term, new_chain=True)
            else:
                append_pose_to_pose(new_pose, subpose, new_chain=True)

        # make pose point to new pose
        pose = new_pose
        # get the chains again
        chains = list(pose.split_by_chain())
        # clone the chain again in case we are extending the N-terminal helix too
        chain_to_extend = chains[chain - 1].clone()
    else:
        pass
    if extend_n_term > 0:
        # align last 7 residues of ideal_n_term to the first 7 residues of chain_to_extend
        range_CA_align(
            pose_a=ideal_n_term,
            pose_b=chain_to_extend,
            start_a=ideal_n_term.total_residue() - 6,
            end_a=ideal_n_term.total_residue(),
            start_b=chain_to_extend.chain_begin(1),
            end_b=chain_to_extend.chain_begin(1) + 6,
        )
        # build a new chain by appending the chain_to_extend to the ideal_n_term
        extended_n_term = Pose()
        # first add the ideal_n_term into the extended_n_term plus one additional residue
        for i in range(
            ideal_n_term.chain_begin(1), ideal_n_term.chain_end(1) - 5
        ):  # -7+1+1
            extended_n_term.append_residue_by_bond(ideal_n_term.residue(i))
        for i in range(
            chain_to_extend.chain_begin(1) + 1, chain_to_extend.chain_end(1) + 1
        ):
            extended_n_term.append_residue_by_bond(chain_to_extend.residue(i))
        new_pose = Pose()
        for i, subpose in enumerate(chains, start=1):
            if i == chain:
                append_pose_to_pose(new_pose, extended_n_term, new_chain=True)
            else:
                append_pose_to_pose(new_pose, subpose, new_chain=True)
        pose = new_pose
    else:
        pass
    if idealize:
        idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
        idealize_mover.apply(pose)
    else:
        pass

    return pose




@requires_init
def make_bound_states(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    Wrapper for distributing BoundStateMaker.
    :param packed_pose_in: The input pose.
    :param kwargs: The keyword arguments to pass to BoundStateMaker.
    :return: An iterator of PackedPoses.
    """
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io

    sys.path.insert(0, "/mnt/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=True, pack_result=False
        )
    final_pposes = []
    for pose in poses:
        # make a new BoundStateMaker for each pose
        state_maker = BoundStateMaker(pose, **kwargs)
        # generate states
        for ppose in state_maker.generate_states():
            yield ppose
