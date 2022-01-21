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


def extend_helix_termini(
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
    :param: chain: Chain to extend the terminal helices of.
    :return: Pose with the terminal helices extended.
    Extend the terminal helices of a pose by a specified number of residues at the 
    provided chain. The new residues are valines.
    """

    import pyrosetta
    from pyrosetta.rosetta.core.pose import append_pose_to_pose, Pose
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    # get the scores of the pose
    scores = dict(pose.scores)
    # Get the chains of the pose
    chains = pose.num_chains()
    # make a SwitchChainOrderMover to rebuild the pose
    sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    chain_order = "".join(str(i) for i in range(1, chains + 1))
    sw.chain_order(chain_order)
    sw.apply(pose)
    # make a dictionary of the chains
    chain_dict = {}
    for i, subpose in enumerate(pose.split_by_chain(), start=1):
        chain_dict[i] = subpose.clone()
    # get the chain we want to extend
    slicer = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    slicer.chain_order(str(chain))
    pose_to_extend = pose.clone()
    slicer.apply(pose_to_extend)
    # fix the phi/psi/omega of the first and last residues of the chain
    # steal them from the next (for the first) and previous (for the last) residues
    last_res = pose_to_extend.total_residue()
    pose_to_extend.set_phi(1, pose_to_extend.phi(2))
    pose_to_extend.set_psi(1, pose_to_extend.psi(2))
    pose_to_extend.set_omega(1, pose_to_extend.omega(2))
    pose_to_extend.set_phi(last_res, pose_to_extend.phi(last_res - 1))
    pose_to_extend.set_psi(last_res, pose_to_extend.psi(last_res - 1))
    pose_to_extend.set_omega(last_res, pose_to_extend.omega(last_res - 1))
    # f-strings for inserters
    inserter_c_term = f"""
    <MOVERS>
        <InsertResMover 
            name="cterm" 
            chain="A" 
            residue="{pose_to_extend.chain_end(1)}"
            steal_angles_from_res="{pose_to_extend.chain_end(1)}"
            grow_toward_Nterm="false"
            additionalResidue="{extend_c_term}" />
    </MOVERS>
    """
    inserter_n_term = f"""
    <MOVERS>
        <InsertResMover 
            name="nterm" 
            chain="A" 
            residue="1"
            steal_angles_from_res="2"
            grow_toward_Nterm="true"
            additionalResidue="{extend_n_term}" />
    </MOVERS>
    """
    if extend_c_term > 0:
        c_term_extender = XmlObjects.create_from_string(inserter_c_term)
        c_term_extender = c_term_extender.get_mover("cterm")
        c_term_extender.apply(pose_to_extend)
    else:
        pass
    if extend_n_term > 0:
        n_term_extender = XmlObjects.create_from_string(inserter_n_term)
        n_term_extender = n_term_extender.get_mover("nterm")
        n_term_extender.apply(pose_to_extend)
    else:
        pass
    pose = Pose()
    pre_extended_chain = chain_dict[chain]

    range_CA_align(
        pose_to_extend, 
        pre_extended_chain, 
        extend_n_term + 1, 
        extend_n_term + len(pre_extended_chain.residues),
        pre_extended_chain.chain_begin(1),
        pre_extended_chain.chain_end(1),
    )
    chain_dict[chain] = pose_to_extend
    for i, subpose in chain_dict.items():
        append_pose_to_pose(pose, subpose, new_chain=True)
    sw.apply(pose)
    if idealize:
        idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
        idealize_mover.apply(pose)
    else:
        pass
    for key, value in scores.items():
        pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
    return pose

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
    :param pose_a: Pose 1
    :param pose_b: Pose 2
    :param start_a: start index of range 1
    :param end_a: end index of range 1
    :param start_b: start index of range 2
    :param end_b: end index of range 2
    :return: None
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
    Get fa_rep score for an all-glycine pose.
    Mutate all residues to glycine then return the score of the mutated pose.
    :param pose: Pose
    :return: fa_rep score
    """

    import pyrosetta

    # initialize empty sfxn
    sfxn = pyrosetta.rosetta.core.scoring.ScoreFunction()
    # add fa_rep to sfxn only
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


def combine_two_poses(
    pose_a: Pose,
    pose_b: Pose,
    end_a: int,
    start_b: int,
) -> Pose:
    """
    Make a new pose, containing pose_a up to end_a, then pose_b starting from start_b
    Assumes pose_a has only one chain.
    If you don't know why you are using this function then you should probably use the
    pyrosetta.rosetta.core.pose.append_pose_to_pose() method instead.
    :param pose_a: Pose 1
    :param pose_b: Pose 2
    :param end_a: end index of Pose 1
    :param start_b: start index of Pose 2
    :return: new Pose
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    newpose = Pose()
    for i in range(1, end_a + 1):
        newpose.append_residue_by_bond(pose_a.residue(i))
    newpose.append_residue_by_jump(
        pose_b.residue(start_b), newpose.chain_end(1), "CA", "CA", 1
    )
    for i in range(
        start_b + 1, pose_b.total_residue() + 1
    ):  # TODO make sure this doesn't break the statemakers
        newpose.append_residue_by_bond(pose_b.residue(i))
    return newpose


def check_pairwise_interfaces(
    pose: Pose,
    int_count_cutoff: int,
    int_ratio_cutoff: float,
    pairs: Optional[List[tuple]] = None,
) -> bool:
    """
    Check for interfaces between all pairs of chains in the pose.
    Given cutoffs for counts of interfacial residues and ratio of interfacial
    between all pairs of chains, return True if the pose passes the cutoffs.
    :param pose: Pose to check for interfaces.
    :param int_count_cutoff: Minimum number of residues in all interfaces.
    :param int_ratio_cutoff: Minimum ratio of counts between all interfaces.
    :param pairs: List of pairs of chains to check for interfaces.
    :return: True if the pose passes the cutoffs.
    """

    from itertools import combinations
    import string
    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        InterGroupInterfaceByVectorSelector,
    )

    # make a dict mapping of chain indices to chain letters
    max_chains = list(string.ascii_uppercase)
    index_to_letter_dict = dict(zip(range(1, len(max_chains) + 1), max_chains))
    # use the dict to get the chain letters for each chain in the pose
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
    for count_pair in combinations(interface_counts.keys(), 2):
        if (
            interface_counts[count_pair[0]] / interface_counts[count_pair[1]]
            < int_ratio_cutoff
        ):
            return False
        else:
            pass
    return True


def count_interface(pose: Pose, sel_a: ResidueSelector, sel_b: ResidueSelector) -> int:
    """
    Given a pose and two residue selectors, return the number of
    residues in the interface between them.
    :param pose: Pose
    :param sel_a: ResidueSelector
    :param sel_b: ResidueSelector
    :return: int
    """

    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import (
        InterGroupInterfaceByVectorSelector,
    )

    int_sel = InterGroupInterfaceByVectorSelector(sel_a, sel_b)
    int_sel.nearby_atom_cut(3)
    int_sel.vector_dist_cut(5)
    int_sel.cb_dist_cut(7)
    int_count = sum(list(int_sel.apply(pose)))
    return int_count


def measure_CA_dist(pose: Pose, resi_a: int, resi_b: int) -> float:
    """
    Given a pose and two residue indices, return the distance between the CA atoms of
    two residues.
    :param pose: Pose to measure CA distance in.
    :param resi_a: Residue index of residue A.
    :param resi_b: Residue index of residue B.
    :return: Distance between CA atoms of residues A and B, in Angstroms.
    """

    resi_a_coords = pose.residue(resi_a).xyz("CA")
    resi_b_coords = pose.residue(resi_b).xyz("CA")
    dist = resi_a_coords.distance(resi_b_coords)
    return dist


class StateMaker(ABC):
    """
    This abstract class is used to derive classes that make states from an input Pose
    or PackedPose.
    """

    import pyrosetta.distributed.io as io

    def __init__(self, pose: Union[PackedPose, Pose], pre_break_helix: int, **kwargs):
        """
        Initialize the base class with common attributes.
        """

        self.input_pose = io.to_pose(pose)
        self.pre_break_helix = pre_break_helix
        self.post_break_helix = self.pre_break_helix + 1
        self.scores = dict(self.input_pose.scores)
        self.starts = self.get_helix_endpoints(self.input_pose, n_terminal=True)
        self.ends = self.get_helix_endpoints(self.input_pose, n_terminal=False)
        if not "name" in kwargs:
            self.original_name = self.input_pose.pdb_info().name()
        else:
            self.original_name = kwargs["name"]

    @staticmethod
    def get_helix_endpoints(pose: Pose, n_terminal: bool) -> dict:
        """
        Use dssp to get the endpoints of helices.
        Make a dictionary of the start (n_terminal=True) or end residue indices of
        helices in the pose.
        :param pose: Pose to get endpoints from.
        :param n_terminal: If True, get the start residue indices of helices.
        :return: Dictionary of start or end residue indices of helices in the pose.
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
        :param pose: Pose to shift.
        :param i: Number of residues to shift by.
        :param starts: Dictionary of start residue indices of helices in the pose.
        :param ends: Dictionary of end residue indices of helices in the pose.
        :param pivot_helix: Helix number to pivot the shift around.
        :param full_helix: If True, align the entire helix.
        :return: Pose shifted by i residues.
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
        range_CA_align(shifted_pose, pose, start_a, end_a, start_b, end_b)
        return shifted_pose

    @abstractmethod
    def generate_states(self) -> None:
        """
        This function needs to be implemented by the child class of StateMaker.
        """
        pass


class FreeStateMaker(StateMaker):
    """
    A class for generating free states, AKA shifty crispies.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parent class then modify attributes with additional kwargs
        """
        super(FreeStateMaker, self).__init__(*args, **kwargs)
        if "clash_cutoff" in kwargs:
            self.clash_cutoff = kwargs["clash_cutoff"]
        else:
            self.clash_cutoff = 999999
        if "int_count_cutoff" in kwargs:
            self.int_count_cutoff = kwargs["int_count_cutoff"]
        else:
            self.int_count_cutoff = 11
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

        # setup the rechain mover
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        # we expect 2 chains in the resulting poses after states are generated
        rechain.chain_order("12")
        states = []
        # scan 1 heptad forwards and backwards
        for i in range(-7, 8):
            # first do the pre break side, then do the post break side
            for pivot_helix in [self.pre_break_helix, self.post_break_helix]:
                shifted_pose = self.shift_pose_by_i(
                    self.input_pose, i, self.starts, self.ends, pivot_helix, False
                )
                # handle situations where the shift is not possible
                if shifted_pose is None:
                    continue
                else:
                    pass
                end_pose_a, start_pose_b = (
                    self.ends[self.pre_break_helix],
                    self.starts[self.post_break_helix],
                )
                # stitch the hinge pose together after alignment-based docking
                # maintain the original position of the side that was not moved
                if pivot_helix == self.pre_break_helix:
                    combined_pose = combine_two_poses(
                        self.input_pose, shifted_pose, end_pose_a, start_pose_b
                    )
                else:
                    combined_pose = combine_two_poses(
                        shifted_pose, self.input_pose, end_pose_a, start_pose_b
                    )
                # fix PDBInfo and chain numbering
                rechain.apply(combined_pose)
                # mini filtering block
                bb_clash = clash_check(combined_pose)
                # check if clash is too high
                if bb_clash > self.clash_cutoff:
                    continue
                # check if interface residue counts are acceptable
                elif not check_pairwise_interfaces(
                    pose=combined_pose,
                    int_count_cutoff=self.int_count_cutoff,
                    int_ratio_cutoff=self.int_ratio_cutoff,
                ):
                    continue
                else:
                    pass
                # set and update the scores of the combined pose
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
        # yield all packed poses that were generated
        for ppose in states:
            yield ppose


class BoundStateMaker(StateMaker):
    """
    A class for generating bound states, AKA canonical crispy shifties.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the parent class then modify attributes with additional kwargs
        """
        super(BoundStateMaker, self).__init__(*args, **kwargs)
        if "clash_cutoff" in kwargs:
            self.clash_cutoff = kwargs["clash_cutoff"]
        else:
            self.clash_cutoff = 999999
        if "int_count_cutoff" in kwargs:
            self.int_count_cutoff = kwargs["int_count_cutoff"]
        else:
            self.int_count_cutoff = 11
        if "int_ratio_cutoff" in kwargs:
            self.int_ratio_cutoff = kwargs["int_ratio_cutoff"]
        else:
            self.int_ratio_cutoff = 0.000001

        self.parent_length = len(self.input_pose.residues)

    def generate_states(self) -> Iterator[PackedPose]:
        """
        Generate all bound states that pass default or supplied cutoffs.
        """

        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # setup the rechain mover
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        # we expect 3 chains in the resulting poses after states are generated
        rechain.chain_order("123")
        states = []
        # scan 1 heptad forwards and backwards
        for i in range(-7, 8):
            # first do the pre break side, then do the post break side
            for pivot_helix in [self.pre_break_helix, self.post_break_helix]:
                shifted_pose = self.shift_pose_by_i(
                    self.input_pose, i, self.starts, self.ends, pivot_helix, False
                )
                # handle situations where the shift is not possible
                if shifted_pose is None:
                    continue
                else:
                    pass
                end_pose_a, start_pose_b = (
                    self.ends[self.pre_break_helix],
                    self.starts[self.post_break_helix],
                )
                # stitch the hinge pose together after alignment-based docking
                # maintain the original position of the side that was not moved
                if pivot_helix == self.pre_break_helix:
                    combined_pose = combine_two_poses(
                        self.input_pose, shifted_pose, end_pose_a, start_pose_b
                    )
                    # setup the order for docking helices into the combined pose
                    docking_order = [shifted_pose, self.input_pose]
                else:
                    combined_pose = combine_two_poses(
                        shifted_pose, self.input_pose, end_pose_a, start_pose_b
                    )
                    # setup the order for docking helices into the combined pose
                    docking_order = [self.input_pose, shifted_pose]
                # we want to dock the helices before and after the pivot helix
                helices_to_dock = [pivot_helix - 1, pivot_helix + 1]
                # dock the bound helix into the hinge pose
                for hinge_pose, helix_to_dock in zip(docking_order, helices_to_dock):
                    # reuse the combined pose as the docking target
                    dock = combined_pose.clone()
                    # add the first residue of the helix to dock to the target
                    dock.append_residue_by_jump(
                        hinge_pose.residue(self.starts[helix_to_dock]),
                        dock.chain_end(1),
                        "CA",
                        "CA",
                        1,
                    )
                    # add the rest of the residues of the helix to dock to the target
                    for resid in range(
                        self.starts[helix_to_dock] + 1, self.ends[helix_to_dock] + 1
                    ):
                        dock.append_residue_by_bond(hinge_pose.residue(resid))

                    # fix PDBInfo and chain numbering
                    rechain.apply(dock)
                    # extend the bound helix
                    dock = extend_helical_termini(
                        pose=dock,
                        chain=3,
                        extend_n_term=5,
                        extend_c_term=5,
                        idealize=False,
                    )
                    # fix PDBInfo and chain numbering again
                    rechain.apply(dock)
                    # mini filtering block
                    bb_clash = clash_check(dock)
                    # check if clash is too high
                    if bb_clash > self.clash_cutoff:
                        continue
                    # check if interface residue counts are acceptable
                    elif not check_pairwise_interfaces(
                        pose=dock,
                        int_count_cutoff=self.int_count_cutoff,
                        int_ratio_cutoff=self.int_ratio_cutoff,
                    ):
                        continue
                    else:
                        pass
                    # set and update the scores of the combined pose
                    for key, value in self.scores.items():
                        setPoseExtraScore(dock, key, str(value))
                    setPoseExtraScore(dock, "bb_clash", float(bb_clash))
                    setPoseExtraScore(dock, "docked_helix", str(helix_to_dock))
                    setPoseExtraScore(dock, "parent", self.original_name)
                    setPoseExtraScore(dock, "parent_length", str(self.parent_length))
                    setPoseExtraScore(dock, "pivot_helix", str(pivot_helix))
                    setPoseExtraScore(
                        dock, "pre_break_helix", str(self.pre_break_helix)
                    )
                    setPoseExtraScore(dock, "shift", str(i))
                    setPoseExtraScore(
                        dock,
                        "state",
                        f"{self.original_name}_p_{str(pivot_helix)}_s_{str(i)}_d_{helix_to_dock}",
                    )
                    ppose = io.to_packed(dock)
                    states.append(ppose)
        # yield all the packed poses that were generated
        for ppose in states:
            yield ppose


# TODO could use a factory function to return the appropriate class
# TODO this would then serve as a general wrapper for the state maker classes
# TODO and would use Union["free", "bound", ...] that maps to the appropriate class
# TODO but that wouldn't save that many lines of code
# TODO it would make the notebooks a little more confusing too so I'm leaving it


@requires_init
def make_free_states(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    Wrapper for distributing FreeStateMaker.
    :param packed_pose_in: The input pose.
    :param kwargs: The keyword arguments to pass to FreeStateMaker.
    :return: An iterator of PackedPoses.
    """
    import sys
    from pathlib import Path
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    from pathlib import Path
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
