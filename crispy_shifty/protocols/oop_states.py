# Python standard library
from abc import ABC, abstractmethod
from typing import Generator, Iterator, Optional, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose

# Custom library imports


class StateMaker(ABC):
    """
    This abstract class is used to derive classes that make states from an input Pose
    or PackedPose.
    """

    import pyrosetta.distributed.io as io

    def __init__(self, pose: Union[PackedPose, Pose], pre_break_helix: int, **kwargs):

        self.input_pose = pose
        self.pre_break_helix = pre_break_helix
        self.post_break_helix = self.pre_break_helix + 1
        self.scores = dict(self.input_pose.scores)
        if not "name" in kwargs:
            self.original_name = self.input_pose.pdb_info().name()
        else:
            self.original_name = kwargs["name"]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def check_pairwise_interfaces(self, pose: Pose, int_count_cutoff: int, int_ratio_cutoff: float, pairs: Optional[List[tuple]]) -> bool:
        """
        Check for interfaces between all pairs of chains in the pose.
        Given cutoffs for counts of interfacial residues and ratio of interfacial 
        between all pairs of chains, return True if the pose passes the cutoffs.
        :param pose: Pose to check for interfaces.
        :param int_count_cutoff: Minimum number of residues in all interfaces.
        :param int_ratio_cutoff: Minimum ratio of counts between all interfaces.
        :param pairs: List of pairs of chains to check for interfaces.
        :return: True if the pose passes the cutoffs.
        TODO: This function is not generalizable to other StateMakers.
        """

        from itertools import combinations
        import pyrosetta
        from pyrosetta.rosetta.core.select.residue_selector import (
            ChainSelector,
            InterGroupInterfaceByVectorSelector,
        )

        # make a dict mapping of chain indices to chain letters

        max_chains = list(string.ascii_uppercase)
        index_to_letter_dict = dict(zip(range(1, len(max_chains) + 1), max_chains))

        pose_chains = [index_to_letter_dict[i] for i in range(1, pose.num_chains() + 1)]

        if pairs is not None:
            # if pairs are given, use those
            chain_pairs = pairs
        else:
            # otherwise, use all combinations of chains
            chain_pairs = list(combinations(pose_chains, 2))

        # make a dict of all interface counts and check all counts
        interface_counts = {}
        for pair in chain_pairs:
            interface_name = "".join(pair)
            pair_a, pair_b = ChainSelector(pair[0]), ChainSelector(pair[1])
            interface = InterGroupInterfaceByVectorSelector(pair_a, pair_b)
            interface_count = sum(list(interface.apply(pose)))
            if interface_count < int_count_cutoff:
                return False
            else:
                pass
            interface_counts[interface_name] = interface_count

        # check all possible ratios
        for count_pair in combinations(interface_counts.values(), 2):
            if interface_counts[count_pair[0]] / interface_counts[count_pair[1]] < int_ratio_cutoff:
                return False

            else:
                pass
        return True


    @staticmethod
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

    @staticmethod
    def measure_CA_dist(pose: Pose, resi_a: int, resi_b: int) -> float:
        resi_a_coords = pose.residue(resi_a).xyz("CA")
        resi_b_coords = pose.residue(resi_b).xyz("CA")
        dist = resi_a_coords.distance(resi_b_coords)
        return dist

    @staticmethod
    def get_helix_endpoints(pose: Pose, n_terminal: bool) -> dict:
        """
        Use dssp to get the endpoints of helices.
        Make a dictionary of the start (n_terminal=True) or end residue indices of each helix
        """
        import pyrosetta

        ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        # first make a dictionary of all helical residues, indexed by helix number
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

        # now get the start and end residues of each helix from the helix_dict
        helix_endpoints = {}
        if n_terminal:
            index = 0  # helix start residue
        else:
            index = -1  # helix end residue
        for helix, residue_list in helix_dict.items():
            helix_endpoints[helix] = residue_list[index]
        return helix_endpoints

    @staticmethod
    def combine_two_poses(
        pose_a: Pose,
        pose_b: Pose,
        end_a: int,
        start_b: int,
    ) -> Pose:
        """
        Make a new pose, containing pose_a up to end_a, then pose_b starting from start_b
        Assumes pose_a has only one chain.
        TODO: this is a hack, fix it to be more safe and general
        TODO: maybe this shouldn't be a static method?
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

    def shift_pose_by_i(
        self,
        pose: Pose,
        i: int,
        starts: dict,
        ends: dict,
        pivot_helix: int,
        full_helix: bool = False,
    ) -> Union[Pose, None]:
        """
        Shift a pose by i residues.
        If full_helix is true, align along the entire helix, otherwise only align 10
        residues. If i is positive, shift the pose forward, if negative, shift the pose
        backward. Returns None if the shift alignment is not sufficient.
        """

        pose = pose.clone()
        shifted_pose = pose.clone()
        start = starts[pivot_helix]
        end = ends[pivot_helix]
        if full_helix:
            # ensures there is at least a heptad aligned
            # prevents formation of states with no side domain contact
            if (i >= 0) and (end - start >= i + 7):
                start_a = start
                start_b = start + i
                end_a = end - i
                end_b = end
            elif (i < 0) and (start - end <= i - 7):
                start_a = start - i
                start_b = start
                end_a = end
                end_b = end + i
            else:
                return None
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
                return None
        StateMaker.range_CA_align(shifted_pose, pose, start_a, end_a, start_b, end_b)
        return shifted_pose

    @abstractmethod
    def generate_states(self) -> None:
        """
        This function needs to be implemented by the child class of StateMaker.
        :param state_name: The name of the state.
        :return: The state.
        """
        return None


class FreeStateMaker(StateMaker):
    """
    A class for generating free states, AKA shifty crispies.
    """



    def __init__(self, *args, **kwargs):
        """
        initialize the parent class then modify attributes with additional kwargs
        """
        super(FreeStateMaker, self).__init__(*args, **kwargs)
        if "clash_cutoff" in kwargs:
            self.clash_cutoff = kwargs["clash_cutoff"]
        else:
            self.clash_cutoff = 999999
        if "int_count_cutoff" in kwargs:
            self.int_count_cutoff = kwargs["int_count_cutoff"]
        else:
            self.int_ratio_cutoff = 11
        if "int_ratio_cutoff" in kwargs:
            self.int_ratio_cutoff = kwargs["int_ratio_cutoff"]
        else:
            self.int_ratio_cutoff = 0.000001

        self.parent_length = len(self.input_pose.residues)

    def generate_states(self) -> Iterator[PackedPose]:
        """
        Generate all free states that pass default or supplied cutoffs.
        """
        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        rechain.chain_order("12")
        starts = self.get_helix_endpoints(self.input_pose, n_terminal=True)
        ends = self.get_helix_endpoints(self.input_pose, n_terminal=False)
        states = []
        # scan 1 heptad forwards and backwards
        for i in range(-7, 8):
            # first do the pre break side, then do the post break side
            for pivot_helix in [self.pre_break_helix, self.post_break_helix]:
                shifted_pose = self.shift_pose_by_i(
                    self.input_pose, i, starts, ends, pivot_helix, False
                )
                # handle situations where the shift is not possible
                if shifted_pose is None:
                    continue
                else:
                    pass
                end_pose_a, start_pose_b = (
                    ends[self.pre_break_helix],
                    starts[self.post_break_helix],
                )
                # stitch the pose together after alignment-based docking
                if pivot_helix == self.pre_break_helix:
                    combined_pose = self.combine_two_poses(
                        self.input_pose, shifted_pose, end_pose_a, start_pose_b
                    )
                else:
                    combined_pose = self.combine_two_poses(
                        shifted_pose, self.input_pose, end_pose_a, start_pose_b
                    )
                """TODO: Rebuild PDBInfo for the combined pose?"""
                # fix PDBInfo and numbering
                rechain.apply(combined_pose)
                # mini filtering block
                bb_clash = self.clash_check(combined_pose)
                # check if clash is too high
                if bb_clash > self.clash_cutoff:
                    continue
                # check if interface residue counts are acceptable
                # TODO: uncomment this once it works
                # elif not self.count_interface_check(combined_pose, self.int_ratio_cutoff):
                #     continue
                else:
                    pass
                for key, value in self.scores.items():
                    setPoseExtraScore(combined_pose, key, str(value))
                setPoseExtraScore(combined_pose, "bb_clash", float(bb_clash))
                setPoseExtraScore(combined_pose, "parent", self.original_name)
                setPoseExtraScore(
                    combined_pose, "parent_length", str(self.parent_length)
                )
                setPoseExtraScore(combined_pose, "pivot_helix", str(pivot_helix))
                setPoseExtraScore(
                    combined_pose, "pre_break_helix", str(self.pre_break_helix)
                )
                setPoseExtraScore(combined_pose, "shift", str(i))
                setPoseExtraScore(
                    combined_pose,
                    "state",
                    f"{self.original_name}_p_{str(pivot_helix)}_s_{str(i)}",
                )
                ppose = io.to_packed(combined_pose)
                states.append(ppose)

        for ppose in states:
            yield ppose

@requires_init
def make_free_states(packed_pose_in: Optional[PackedPose] = None, **kwargs) -> Iterator[PackedPose]:
    """
    Wrapper for FreeStateMaker.
    :param packed_pose_in: The input pose.
    :param kwargs: The keyword arguments to pass to FreeStateMaker.
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
        # make a new FreeStateMaker for each pose
        state_maker = FreeStateMaker(pose, **kwargs)
        # generate states
        for ppose in state_maker.generate_states():
            yield ppose






