# 3rd party library imports
# Rosetta library imports
import pyrosetta
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector


def score_rmsd(
    pose: Pose,
    refpose: Pose,
    sel: ResidueSelector = None,
    refsel: ResidueSelector = None,
    rmsd_type: pyrosetta.rosetta.core.scoring.rmsd_atoms = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca,
    name: str = "rmsd",
):
    # written by Adam Broerman
    rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
    rmsd_metric.set_comparison_pose(refpose)
    if sel == None:
        sel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    rmsd_metric.set_residue_selector(sel)
    if refsel == None:
        refsel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    rmsd_metric.set_residue_selector_reference(refsel)
    rmsd_metric.set_rmsd_type(rmsd_type)  # Default is rmsd_all_heavy
    rmsd_metric.set_run_superimpose(True)
    rmsd = rmsd_metric.calculate(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, rmsd)
    return rmsd


def find_top_alignment_index(a,b):
    """
    Finds the index of top alignment for b in a. 
    Top alignment meaning the alignment where the most entries in b match 
    their currently aligned counterparts in a 
    """
    top_score = 0
    top_idx = -1
    L = len(b)
    
    for i in range(len(a) - L + 1):
        
        chunk = a[i:i+L]
        
        cur_score = 0
        for b_char, chunk_char in zip(b, chunk):
            if b_char == chunk_char:
                cur_score += 1
        
        if cur_score > top_score:
            top_idx = i
            top_score = cur_score

    return top_idx


def model_hinge_alt_state(
    pose: Pose,
    alt_state: Pose, # hinge alt state should be chain 1
    end_N_side: int, # for our hinges, should be the last helical residue before the new loop
    start_C_side: int, # for our hinges, should be the first helical residue after the new loop
):
    # written by Adam Broerman
    from copy import deepcopy
    from crispy_shifty.protocols.states import range_CA_align, fuse_two_poses

    pose_chains = list(pose.split_by_chain())
    fusion_pose = pose_chains[0]

    # find top sequence alignment of hinge
    top_idx = find_top_alignment_index(fusion_pose.sequence(), alt_state.chain_sequence(1))

    # align alt_state to fusion_pose from resi 1 to end_N_side
    range_CA_align(alt_state, fusion_pose, 1, end_N_side, 1+top_idx, end_N_side+top_idx)

    # extract C-terminal side of pose into a new C-pose
    # hacky leave an extra residue so the actual first residue is not the Nterm variant
    extract_C_side = pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover(1, start_C_side-2)
    extract_C_side.set_rechain(True)
    C_pose = deepcopy(fusion_pose)
    extract_C_side.apply(C_pose)

    # align C-pose to alt_state from start of next helix to end of alt_state
    range_CA_align(C_pose, alt_state, 2, alt_state.chain_end(1)-start_C_side+2, start_C_side, alt_state.chain_end(1))

    # rebuild pose in alt state with the loop from alt_state
    alt_pose = fuse_two_poses(fusion_pose, alt_state, end_N_side+top_idx, end_N_side+1, 1, start_C_side-1)
    # start_b from non-Nterm variant actual first residue
    alt_pose = fuse_two_poses(alt_pose, C_pose, start_C_side+top_idx-1, 2)

    # add any additional chains from the original pose
    for additional_chain in pose_chains[1:]:
        pyrosetta.rosetta.core.pose.append_pose_to_pose(alt_pose, additional_chain, True)

    # add any additional chains from alt_state to pose
    for additional_chain in list(alt_state.split_by_chain())[1:]:
        pyrosetta.rosetta.core.pose.append_pose_to_pose(alt_pose, additional_chain, True)
    
    return alt_pose


def add_interaction_partner(
    pose: Pose,
    int_state: Pose,
    int_chain: int,
):
    # written by Adam Broerman
    from crispy_shifty.protocols.states import range_CA_align

    alt_pose = pose.clone()

    # find top sequence alignment of the chain of the interaction that is in the pose
    top_idx = find_top_alignment_index(alt_pose.sequence(), int_state.chain_sequence(int_chain))

    # align int_state to pose
    range_CA_align(int_state, alt_pose, int_state.chain_begin(int_chain), int_state.chain_end(int_chain), 1+top_idx, int_state.chain_end(int_chain)-int_state.chain_begin(int_chain)+1+top_idx)

    # add additional chains of the interaction from int_state to pose
    for i, additional_chain in enumerate(list(int_state.split_by_chain()), start=1):
        if i != int_chain:
            pyrosetta.rosetta.core.pose.append_pose_to_pose(alt_pose, additional_chain, True)
    
    return alt_pose