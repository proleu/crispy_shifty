# Python standard library
from typing import *
# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
# Custom library imports


@requires_init
def make_bound_states(
    packed_pose_in:Optional[PackedPose] = None, **kwargs
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
    from pyrosetta.distributed.packed_pose.core import PackedPose
    from pyrosetta.rosetta.core.pose import Pose
    
    sys.path.insert(0, "/mnt/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    def range_CA_align(
        pose_a:Pose, 
        pose_b:Pose, 
        start_a:int,
        end_a:int,
        start_b:int, 
        end_b:int,
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

        pose_a_coordinates = (
            pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
        )
        pose_b_coordinates = (
            pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
        )

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
        Get fa_rep score for a pose.
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
            if (ss.get_dssp_secstruct(i) == "H") & (
                ss.get_dssp_secstruct(i - 1) != "H"
            ):
                helix_dict[n] = [i]
            if (ss.get_dssp_secstruct(i) == "H") & (
                ss.get_dssp_secstruct(i + 1) != "H"
            ):
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

            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                dock, "docked_helix", str(helix_to_dock)
            )
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
    final_pposes = []

    for pose in poses:
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
                        bb_clash = clash_check(shift)
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, "bb_clash", float(bb_clash)
                        )
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, "parent", original_name
                        )
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, "parent_length", str(parent_length)
                        )
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, "pivot_helix", str(pivot_helix)
                        )
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, "pre_break_helix", str(pre_break_helix)
                        )
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, "shift", str(i)
                        )
                        docked_helix = shift.scores["docked_helix"]
                        pyrosetta.rosetta.core.pose.setPoseExtraScore(
                            shift, 
                            "state", 
                            f"{original_name}_p_{str(pivot_helix)}_s_{str(shift)}_d_{docked_helix}",
                        )
                        ppose = io.to_packed(shift)
                        states.append(ppose)
                except:  # for cases where there isn't enough to align against
                    continue
        final_pposes += states
    for ppose in final_pposes: 
        yield ppose
