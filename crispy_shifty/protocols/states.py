# Python standard library
from typing import *

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose

# Custom library imports


def range_CA_align(
    pose_a: Pose,
    pose_b: Pose,
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> None:
    """
    Align poses by superimposition of CA given two ranges of indices.
    Pose 1 is moved. Alignment is applied to the input poses themselves.
    Modified from @apmoyer.
    """
    import pyrosetta

    pose_a_residue_selection = range(start_a, end_a)
    pose_b_residue_selection = range(start_b, end_b)

    assert len(pose_a_residue_selection) == len(pose_b_residue_selection)

    pose_a_coordinates = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
    pose_b_coordinates = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()

    for pose_a_residue_index, pose_b_residue_index in zip(
        pose_a_residue_selection, pose_b_residue_selection
    ):
        pose_a_coordinates.append(pose_a.residues[pose_a_residue_index].xyz("CA"))
        pose_b_coordinates.append(pose_b.residues[pose_b_residue_index].xyz("CA"))

    rotation_matrix = pyrosetta.rosetta.numeric.xyzMatrix_double_t()
    pose_a_center = pyrosetta.rosetta.numeric.xyzVector_double_t()
    pose_b_center = pyrosetta.rosetta.numeric.xyzVector_double_t()

    pyrosetta.rosetta.protocols.toolbox.superposition_transform(
        pose_a_coordinates,
        pose_b_coordinates,
        rotation_matrix,
        pose_a_center,
        pose_b_center,
    )

    pyrosetta.rosetta.protocols.toolbox.apply_superposition_transform(
        pose_a, rotation_matrix, pose_a_center, pose_b_center
    )
    return


def clash_check(pose: Pose) -> float:
    """
    Get fa_rep score for a pose with weight 1. Backbone only.
    Mutate all residues to glycine then return the score of the mutated pose.
    """

    import pyrosetta

    # initialize empty sfxn
    sfxn = pyrosetta.rosetta.core.scoring.ScoreFunction()
    sfxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1)
    # make the pose into a backbone without sidechains
    all_gly = pose.clone()
    true_sel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    true_x = true_sel.apply(all_gly)
    # settle the pose
    pyrosetta.rosetta.protocols.toolbox.pose_manipulation.repack_these_residues(
        true_x, all_gly, sfxn, False, "G"
    )
    score = sfxn(all_gly)
    return score


def count_interface_check(pose: Pose, int_cutoff: float) -> bool:
    """
    Given a state that has a helix bound, check that both halves have 10 interfacial
    contacts with the bound helix. Return true if so, return false if not.
    int_cutoff determines how asymmetric the interfaces are allowed to be. If it is
    set very low, one interface is allowed to have many more residues than the
    other.
    """

    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        InterGroupInterfaceByVectorSelector,
    )

    A = ChainSelector("A")
    B = ChainSelector("B")
    C = ChainSelector("C")

    AC_int = InterGroupInterfaceByVectorSelector(A, C)
    BC_int = InterGroupInterfaceByVectorSelector(B, C)

    AC_int_count = sum(list(AC_int.apply(pose)))
    BC_int_count = sum(list(BC_int.apply(pose)))

    # interfaces need at least 10 residues
    if AC_int_count < 10 or BC_int_count < 10:
        return False
    elif AC_int_count / BC_int_count < int_cutoff:
        return False
    elif BC_int_count / AC_int_count < int_cutoff:
        return False
    else:
        return True


def count_interface(pose: Pose, sel_a, sel_b) -> int:
    """
    Given a pose and two residue selectors, return the number of
    residues in the interface between them.
    """

    import pyrosetta

    int_sel = pyrosetta.rosetta.core.select.residue_selector.InterGroupInterfaceByVectorSelector(
        sel_a, sel_b
    )
    int_sel.nearby_atom_cut(3)
    int_sel.vector_dist_cut(5)
    int_sel.cb_dist_cut(7)
    int_count = sum(list(int_sel.apply(pose)))
    return int_count


def measure_CA_dist(pose: Pose, resi_a: int, resi_b: int) -> float:
    resi_a_coords = pose.residue(resi_a).xyz("CA")
    resi_b_coords = pose.residue(resi_b).xyz("CA")
    dist = resi_a_coords.distance(resi_b_coords)
    return dist


def helix_dict_maker(pose: Pose) -> dict:
    """
    Make a dictionary mapping of residue indices to the helix indices.
    Keys are helix indices, values are lists of residue indices.
    """
    import pyrosetta

    ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    helix_dict = {}
    n = 1
    for i in range(1, len(pose.sequence())):
        if (ss.get_dssp_secstruct(i) == "H") & (ss.get_dssp_secstruct(i - 1) != "H"):
            helix_dict[n] = [i]
        if (ss.get_dssp_secstruct(i) == "H") & (ss.get_dssp_secstruct(i + 1) != "H"):
            helix_dict[n].append(i)
            n += 1
    return helix_dict


def get_helix_endpoints(pose: Pose, n_terminal: bool) -> dict:
    """
    Make a dictionary of the start (n_terminal=True) or end residue indices of each helix
    """
    helix_dict = helix_dict_maker(pose)
    helix_endpoints = {}
    if n_terminal:
        index = 0  # helix start residue
    else:
        index = -1  # helix end residue
    for helix, residue_list in helix_dict.items():
        helix_endpoints[helix] = residue_list[index]
    return helix_endpoints


def combine_two_poses(
    pose_a: Pose,
    pose_b: Pose,
    end_a: int,
    start_b: int,
) -> Pose:
    """
    Make a new pose, containing pose_a up to end_a, then pose_b starting from start_b
    Assumes pose_a has only one chain.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    length = len(pose_a.sequence())
    newpose = Pose()
    for i in range(1, end_a + 1):
        newpose.append_residue_by_bond(pose_a.residue(i))
    newpose.append_residue_by_jump(
        pose_b.residue(start_b), newpose.chain_end(1), "CA", "CA", 1
    )
    for i in range(start_b + 1, length + 1):
        newpose.append_residue_by_bond(pose_b.residue(i))
    return newpose


@requires_init
def make_bound_states(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Generator[PackedPose, PackedPose, None]:
    """
    Generate alternative helix-bound states from the input PackedPose or pdb path.
    This is done by splitting, superimposing and rotating one full heptad up and one
    full heptad down for helices before and after the break, defined by the
    pre_break_helix kwarg, then appending helices that either preceded or followed the
    pre_break_helix in index.
    """
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose
    from pyrosetta.rosetta.core.pose import setPoseExtraScore

    sys.path.insert(0, "/mnt/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    def shift_chB_by_i(
        pose: Pose,
        i: int,
        starts: dict,
        ends: dict,
        pivot_helix: int,
        pre_break_helix: int,
    ) -> tuple:
        """
        Use alignment-based docking on CA atoms of a pivot helix to shift a half of a
        pose by i residues while maintaining a realistic interface at the pivot helix.
        """
        import pyrosetta
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        pose = pose.clone()
        copypose = pose.clone()
        # get the start and end residue indices for the pivot_helix
        start = starts[pivot_helix]
        end = ends[pivot_helix]
        starts_tup = tuple(start for start in 4 * [start])
        ends_tup = tuple(end for end in 4 * [end])
        # make sure there's enough helix to align against going forwards
        if (i >= 0) and ((start + 10 + i) <= end):
            offsets = 3, 10, 3 + i, 10 + i
            start_a, end_a, start_b, end_b = tuple(map(sum, zip(starts_tup, offsets)))
        # make sure there's enough helix to align against going backwards
        elif (i <= 0) and ((end - 10 + i) >= start):
            offsets = -10, -3, -10 + i, -3 + i
            start_a, end_a, start_b, end_b = tuple(map(sum, zip(ends_tup, offsets)))
        else:
            raise RuntimeError("not enough overlap to align")
        # dock by aligning along the pivot helix with an offset of i
        range_CA_align(copypose, pose, start_a, end_a, start_b, end_b)
        end_pose_a, start_pose_b = ends[pre_break_helix], starts[pre_break_helix + 1]
        # stitch the pose together after alignment-based docking
        shifted_pose = combine_two_poses(pose, copypose, end_pose_a, start_pose_b)
        # add in the bound helix
        shifts = []
        # reuse the poses from before
        pose_sequence = copypose, pose
        helices_to_dock = (pivot_helix - 1, pivot_helix + 1)
        for p, helix_to_dock in enumerate(helices_to_dock):
            dock = shifted_pose.clone()
            dock.append_residue_by_jump(
                pose_sequence[p].residue(starts[helix_to_dock]),
                dock.chain_end(1),
                "CA",
                "CA",
                1,
            )
            for resid in range(starts[helix_to_dock] + 1, ends[helix_to_dock] + 1):
                dock.append_residue_by_bond(pose_sequence[p].residue(resid))

            setPoseExtraScore(dock, "docked_helix", str(helix_to_dock))
            shifts.append(dock)

        shifts = tuple(shifts)
        return shifts

    sys.path.insert(0, "/mnt/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=True, pack_result=False
        )

    if "clash_cutoff" in kwargs:
        clash_cutoff = kwargs["clash_cutoff"]
    else:
        clash_cutoff = 999999
    if "int_cutoff" in kwargs:
        int_cutoff = kwargs["int_cutoff"]
    else:
        int_cutoff = 0.000001

    rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    rechain.chain_order("123")
    final_pposes = []
    for pose in poses:
        scores = dict(pose.scores)
        if not "name" in kwargs:
            original_name = pose.pdb_info().name()
        else:
            original_name = kwargs["name"]
        try:
            pre_break_helix = kwargs["pre_break_helix"]
        except KeyError:
            raise RuntimeError("need to supply pre_break_helix")
        parent_length = len(pose.residues)
        starts = get_helix_endpoints(pose, n_terminal=True)
        ends = get_helix_endpoints(pose, n_terminal=False)
        states = []
        post_break_helix = pre_break_helix + 1
        # scan 1 heptad forwards and backwards
        for i in range(-7, 8):
            # first do the pre break side, then do the post break side
            for pivot_helix in [pre_break_helix, post_break_helix]:
                try:
                    shifts = shift_chB_by_i(
                        pose, i, starts, ends, pivot_helix, pre_break_helix
                    )
                    for shift in shifts:
                        docked_helix = shift.scores["docked_helix"]
                        rechain.apply(shift)
                        # mini filtering block
                        bb_clash = clash_check(shift)
                        # check if clash is too high
                        if bb_clash > clash_cutoff:
                            continue
                        # check if interface residue counts are acceptable
                        elif not count_interface_check(shift, int_cutoff):
                            continue
                        else:
                            pass
                        for key, value in scores.items():
                            setPoseExtraScore(shift, key, str(value))
                        setPoseExtraScore(shift, "bb_clash", float(bb_clash))
                        setPoseExtraScore(shift, "parent", original_name)
                        setPoseExtraScore(shift, "parent_length", str(parent_length))
                        setPoseExtraScore(shift, "pivot_helix", str(pivot_helix))
                        setPoseExtraScore(
                            shift, "pre_break_helix", str(pre_break_helix)
                        )
                        setPoseExtraScore(shift, "shift", str(i))
                        setPoseExtraScore(
                            shift,
                            "state",
                            f"{original_name}_p_{str(pivot_helix)}_s_{str(i)}_d_{docked_helix}",
                        )
                        ppose = io.to_packed(shift)
                        states.append(ppose)
                # for cases where there isn't enough to align against
                except:
                    continue
        final_pposes += states
    for ppose in final_pposes:
        yield ppose


@requires_init
def make_dimer_states(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Generator[PackedPose, PackedPose, None]:
    """
    Generate alternative helix-bound states from the input PackedPose or pdb path.
    This is done by splitting, superimposing and rotating one full heptad up and one
    full heptad down for helices before and after the break, defined by the
    pre_break_helix kwarg, then conbinatorially docking states rotated around opposite
    helices to form dimer states.
    """
    from copy import deepcopy
    import itertools
    from time import time
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose
    from pyrosetta.rosetta.core.pose import setPoseExtraScore

    sys.path.insert(0, "/mnt/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    start_time = time()

    def print_timestamp(print_str, end="\n", *args):
        time_min = (time() - start_time) / 60
        print(f"{time_min:.2f} min: {print_str}", end=end)
        for arg in args:
            print(arg)

    def shift_chB_by_i(
        pose: Pose,
        i: int,
        starts: dict,
        ends: dict,
        pivot_helix: int,
        pre_break_helix: int,
        full_helix=False,
    ) -> Pose:

        pose = pose.clone()
        copypose = pose.clone()
        start = starts[pivot_helix]
        end = ends[pivot_helix]
        if full_helix:
            if (i >= 0) and (
                end - start >= i + 7
            ):  # ensures there is at least a heptad aligned (prevents formation of states with no side domain contact)
                start_a = start
                start_b = start + i
                end_a = end - i
                end_b = end
            elif (i < 0) and (
                start - end <= i - 7
            ):  # ensures there is at least a heptad aligned (prevents formation of states with no side domain contact)
                start_a = start - i
                start_b = start
                end_a = end
                end_b = end + i
            else:
                raise RuntimeError("insufficient overlap for alignment")
        else:
            starts_tup = tuple(start for start in 4 * [start])
            ends_tup = tuple(end for end in 4 * [end])
            # make sure there's enough helix to align against going forwards
            if (i >= 0) and ((start + 10 + i) <= end):
                offsets = 3, 10, 3 + i, 10 + i
                start_a, end_a, start_b, end_b = tuple(
                    map(sum, zip(starts_tup, offsets))
                )
            # make sure there's enough helix to align against going backwards
            elif (i <= 0) and ((end - 10 + i) >= start):
                offsets = -10, -3, -10 + i, -3 + i
                start_a, end_a, start_b, end_b = tuple(map(sum, zip(ends_tup, offsets)))
            else:
                raise RuntimeError("insufficient overlap for alignment")
        range_CA_align(copypose, pose, start_a, end_a, start_b, end_b)
        end_pose_a, start_pose_b = ends[pre_break_helix], starts[pre_break_helix + 1]
        # Combines ensuring the domain containing the pivot helix remains aligned to the parent DHR
        if pivot_helix == pre_break_helix:
            shifted_pose = combine_two_poses(pose, copypose, end_pose_a, start_pose_b)
        else:
            shifted_pose = combine_two_poses(copypose, pose, end_pose_a, start_pose_b)
        return shifted_pose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        scores = dict(pose.scores)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)

        if not "name" in kwargs:
            original_name = pose.pdb_info().name()
        else:
            original_name = kwargs["name"]
        print_timestamp(f"Generating states from {original_name}")

        try:
            pre_break_helix = kwargs["pre_break_helix"]
        except KeyError:
            raise RuntimeError("Need to supply pre_break_helix")
        try:
            bb_clash_cutoff = kwargs["bb_clash_cutoff"]
        except KeyError:
            raise RuntimeError("Need to supply bb_clash_cutoff")
        try:
            loop_dist_cutoff = kwargs["loop_dist_cutoff"]
        except KeyError:
            raise RuntimeError("Need to supply loop_dist_cutoff")
        try:
            dhr_int_frac_cutoff = kwargs["dhr_int_frac_cutoff"]
        except KeyError:
            raise RuntimeError("Need to supply dhr_int_frac_cutoff")

        parent_length = len(pose.residues)
        starts = get_helix_endpoints(pose, n_terminal=True)
        ends = get_helix_endpoints(pose, n_terminal=False)
        states_A = []
        states_B = []
        post_break_helix = pre_break_helix + 1
        parent_loop_dist = measure_CA_dist(
            pose, ends[pre_break_helix], starts[post_break_helix]
        )
        # print(parent_loop_dist)

        sel_a = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        sel_a.set_index_range(1, ends[pre_break_helix])
        sel_b = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        sel_b.set_index_range(starts[post_break_helix], parent_length)
        dhr_int_count = count_interface(pose, sel_a, sel_b)
        #     print(f"{dhr_int_count} residues in DHR interface")
        int_cutoff = (
            dhr_int_count * dhr_int_frac_cutoff
        )  # DHR interface fraction cutoff is the fraction of the DHR interface size that the side domains must make with the DHR

        # scan 1 heptad forwards and backwards
        for i in range(-7, 8):
            # first do the pre break side, then do the post break side
            for pivot_helix, protomer, states in zip(
                [pre_break_helix, post_break_helix], ["A", "B"], [states_A, states_B]
            ):
                for full_helix in [0, 1]:
                    print_timestamp(
                        f"Generating state {protomer} {i} {full_helix}...", end=""
                    )
                    try:
                        shift = shift_chB_by_i(
                            pose,
                            i,
                            starts,
                            ends,
                            pivot_helix,
                            pre_break_helix,
                            full_helix,
                        )
                        loop_dist = measure_CA_dist(
                            shift, ends[pre_break_helix], ends[pre_break_helix] + 1
                        )
                        # print(loop_dist)
                        if abs(loop_dist - parent_loop_dist) > loop_dist_cutoff:
                            print("failed due to difference in loop length.")
                            continue
                        bb_clash = clash_check(shift)
                        if bb_clash > bb_clash_cutoff:
                            print("failed due to backbone clashes.")
                            continue

                        # Rebuild PDBInfo
                        pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(shift)
                        shift.pdb_info(pdb_info)
                        shift.pdb_info().name(
                            f"{original_name}_{protomer}_{i}_{full_helix}"
                        )

                        # setPoseExtraScore(shift, f'state_{protomer}', f"{original_name}_{protomer}_{i}_{full_helix}")
                        setPoseExtraScore(shift, f"bb_clash_{protomer}", bb_clash)
                        setPoseExtraScore(shift, f"loop_dist_{protomer}", loop_dist)
                        setPoseExtraScore(shift, f"pivot_helix_{protomer}", pivot_helix)
                        setPoseExtraScore(shift, f"shift_{protomer}", i)
                        print("success.")
                        states.append(shift)
                    except RuntimeError as e:  # for cases where there isn't enough to align against
                        print(f"failed due to {e}.")
                        continue

        an_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
        ac_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("B")
        bn_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("C")
        bc_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("D")
        dhr_sel = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(
            an_sel, bc_sel
        )
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        rechain.chain_order("1234")

        # Combine states from opposite pivots into dimers
        for state_A, state_B in itertools.product(states_A, states_B):
            combo_name = state_A.pdb_info().name() + "_" + state_B.pdb_info().name()
            # combo_name = state_A.scores['state_A'] + '_' + state_B.scores['state_B']
            print_timestamp(f"Generating state combination {combo_name}...", end="")

            combined_state = deepcopy(
                state_A
            )  # copy required; each state_A is reused for every state_B
            (
                state_B_A,
                state_B_B,
            ) = (
                state_B.split_by_chain()
            )  # append_pose_to_pose appends as a single chain
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                combined_state, state_B_A, True
            )
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                combined_state, state_B_B, True
            )

            # Rebuild PDBInfo
            pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(combined_state)
            combined_state.pdb_info(pdb_info)
            rechain.apply(combined_state)

            bb_clash = clash_check(combined_state)
            if bb_clash > bb_clash_cutoff:
                print("failed due to backbone clashes.")
                continue

            # check if interface residue counts are acceptable
            dhr_ac_int_count = count_interface(combined_state, ac_sel, dhr_sel)
            #         print(f"{dhr_ac_int_count} residues in AC DHR interface") # this print statement was generated by CoPilot- crazy!
            dhr_bn_int_count = count_interface(combined_state, bn_sel, dhr_sel)
            #         print(f"{dhr_bn_int_count} residues in BN DHR interface")
            if dhr_ac_int_count < int_cutoff or dhr_bn_int_count < int_cutoff:
                print("failed due to insufficient interface.")
                continue

            combined_state.pdb_info().name(combo_name)
            for key, value in scores.items():
                setPoseExtraScore(combined_state, key, value)
            setPoseExtraScore(combined_state, f"parent", original_name)
            setPoseExtraScore(combined_state, f"parent_path", pdb_path)
            setPoseExtraScore(combined_state, f"parent_length", parent_length)
            setPoseExtraScore(combined_state, f"parent_loop_dist", parent_loop_dist)
            setPoseExtraScore(combined_state, f"pre_break_helix", pre_break_helix)
            setPoseExtraScore(combined_state, f"dhr_int_count", dhr_int_count)
            for key, value in state_A.scores.items():
                setPoseExtraScore(combined_state, key, value)
            for key, value in state_B.scores.items():
                setPoseExtraScore(combined_state, key, value)
            setPoseExtraScore(combined_state, "bb_clash", bb_clash)
            setPoseExtraScore(combined_state, "dhr_ac_int_count", dhr_ac_int_count)
            setPoseExtraScore(combined_state, "dhr_bn_int_count", dhr_bn_int_count)

            ppose = io.to_packed(combined_state)
            print("success.")
            yield ppose
