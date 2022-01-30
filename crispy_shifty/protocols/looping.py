# Python standard library
from typing import Iterator, List, Optional, Tuple

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.distributed import requires_init


def loop_match(pose: Pose, length: int, connections: str = "[A+B]") -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: length: The length of the loop.
    :param: connections: The connections to use.
    :return: Whether the loop was successfully inserted.
    Runs ConnectChainsMover.
    """
    import pyrosetta

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <ConnectChainsMover name="connectchains" 
                chain_connections="{connections}" 
                loopLengthRange="{length},{length}" 
                resAdjustmentRangeSide1="0,0" 
                resAdjustmentRangeSide2="0,0" 
                RMSthreshold="1.0"/>
        </MOVERS>
        """
    )
    cc_mover = objs.get_mover("connectchains")
    try:
        cc_mover.apply(pose)
        closure_type = "loop_match"
    except RuntimeError:  # if ConnectChainsMover cannot find a closure
        closure_type = "not_closed"
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    return closure_type


def loop_extend(
    pose: Pose,
    min_loop_length: int = 2,
    max_loop_length: int = 5,
    connections: str = "[A+B]",
) -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: min_loop_length: The minimum length of the loop.
    :param: max_loop_length: The maximum length of the loop.
    :param: connections: The connections to use.
    :return: Whether the loop was successfully inserted.

    Runs ConnectChainsMover.
    May increase the loop length relative to the parent
    """
    import pyrosetta

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <ConnectChainsMover name="connectchains" 
                chain_connections="{connections}" 
                loopLengthRange="{min_loop_length},{max_loop_length}" 
                resAdjustmentRangeSide1="0,3" 
                resAdjustmentRangeSide2="0,3" 
                RMSthreshold="0.8"/>
        </MOVERS>
        """
    )
    cc_mover = objs.get_mover("connectchains")
    try:
        cc_mover.apply(pose)
        closure_type = "loop_extend"
    except RuntimeError:  # if ConnectChainsMover cannot find a closure
        closure_type = "not_closed"
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    return closure_type


def phi_psi_omega_to_abego(phi: float, psi: float, omega: float) -> str:
    """
    :param: phi: The phi angle.
    :param: psi: The psi angle.
    :param: omega: The omega angle.
    :return: The abego string.
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    if psi == None or phi == None:
        return "X"
    if omega == None:
        omega = 180

    if abs(omega) < 90:
        return "O"
    elif phi > 0:
        if -100.0 <= psi < 100:
            return "G"
        else:
            return "E"
    else:
        if -75.0 <= psi < 50:
            return "A"
        else:
            return "B"


def abego_string(phi_psi_omega: List[Tuple[float]]) -> str:
    """
    :param: phi_psi_omega: A list of tuples of phi, psi, and omega angles.
    :return: The abego string.
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    out = ""
    for x in phi_psi_omega:
        out += phi_psi_omega_to_abego(x[0], x[1], x[2])
    return out


def get_torsions(pose: Pose) -> List[Tuple[float]]:
    """
    :param: pose: The pose to get the torsions from.
    :return: A list of tuples of phi, psi, and omega angles.
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    torsions = []
    for i in range(1, pose.total_residue() + 1):
        phi = pose.phi(i)
        psi = pose.psi(i)
        omega = pose.omega(i)
        if i == 1:
            phi = None
        if i == pose.total_residue():
            psi = None
            omega = None
        torsions.append((phi, psi, omega))
    return torsions


def remodel_helper(
    pose: Pose,
    loop_length: int,
    loop_dssp: Optional[str] = None,
    remodel_before_loop: int = 1,
    remodel_after_loop: int = 1,
    surround_loop_with_helix: bool = False,
) -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: loop_length: The length of the fragment to insert.
    :param: loop_dssp: The dssp string of the fragment to insert.
    :param: remodel_before_loop: The number of residues to remodel before the loop.
    :param: remodel_after_loop: The number of residues to remodel after the loop.
    :return: The filename of the blueprint file to be used to remodel the pose.
    Writes a blueprint file to the current directory or TMPDIR and returns the filename.
    """

    import os, uuid
    import pyrosetta

    tors = get_torsions(pose)
    abego_str = abego_string(tors)
    dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(pose)
    # name blueprint a random 32 long hex string
    if "TMPDIR" in os.environ:
        tmpdir_root = os.environ["TMPDIR"]
    else:
        tmpdir_root = os.getcwd()
    filename = os.path.join(tmpdir_root, uuid.uuid4().hex + ".bp")
    # write a temporary blueprint file
    if not os.path.exists(tmpdir_root):
        os.makedirs(tmpdir_root, exist_ok=True)
    else:
        pass
    with open(filename, "w+") as f:
        end1, begin2 = (
            pose.chain_end(1),
            pose.chain_begin(2),
        )
        end2 = pose.chain_end(2)
        for i in range(1, end1 + 1):
            if i >= end1 - (remodel_before_loop - 1):
                if surround_loop_with_helix:
                    position_dssp = "H"
                else:
                    position_dssp = dssp[i - 1]
                print(
                    str(i),
                    pose.residue(i).name1(),
                    position_dssp + "X",
                    "R",
                    file=f,
                )
            else:
                print(
                    str(i),
                    pose.residue(i).name1(),
                    dssp[i - 1] + abego_str[i - 1],
                    ".",
                    file=f,
                )
        if loop_dssp is None:
            for i in range(loop_length):
                print("0", "V", "LX", "R", file=f)
        else:
            try:
                assert len(loop_dssp) == loop_length
            except AssertionError:
                raise ValueError("loop_dssp must be the same length as loop_length")
            for i in range(loop_length):
                print(
                    "0",
                    "V",
                    f"{loop_dssp[i]}X",
                    "R",
                    file=f,
                )
        for i in range(begin2, end2 + 1):
            if i <= begin2 + (remodel_after_loop - 1):
                if surround_loop_with_helix:
                    position_dssp = "H"
                else:
                    position_dssp = dssp[i - 1]
                print(
                    str(i),
                    pose.residue(i).name1(),
                    position_dssp + "X",
                    "R",
                    file=f,
                )
            else:
                print(
                    str(i),
                    pose.residue(i).name1(),
                    dssp[i - 1] + abego_str[i - 1],
                    ".",
                    file=f,
                )

    return filename


def loop_remodel(
    pose: Pose,
    length: int,
    attempts: int = 10,
    loop_dssp: Optional[str] = None,
    remodel_before_loop: int = 1,
    remodel_after_loop: int = 1,
    remodel_lengths_by_vector: bool = False,
    surround_loop_with_helix: bool = False,
) -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: length: The length of the loop.
    :param: attempts: The number of attempts to make.
    :param: remodel_before_loop: The number of residues to remodel before the loop.
    :param: remodel_after_loop: The number of residues to remodel after the loop.
    :param: remodel_lengths_by_vector: Use the vector angles of chain ends to determine what length to remodel.
    :return: Whether the loop was successfully inserted.
    Remodel a new loop using Blueprint Builder. Expects a pose with two chains.
    DSSP and SS agnostic in principle but in practice more or less matches.
    """
    import os
    import numpy as np
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    # computes the number of residues to remodel before and after the loop by finding which residue-residue vectors point towards the helix to loop to
    # probably works best for building a loop between two helices
    # still uses the default lengths to remodel if none of the vectors are good (dot>0)
    if remodel_lengths_by_vector:
        end1, begin2 = (pose.chain_end(1), pose.chain_begin(2))
        max_dot_1 = 0
        max_dot_2 = 0
        vec_12 = pose.residue(begin2).xyz("CA") - pose.residue(end1).xyz("CA")
        for i in range(3):
            vec_1 = pose.residue(end1 - i).xyz("CA") - pose.residue(end1 - i - 1).xyz(
                "CA"
            )
            dot_1 = vec_12.dot(
                vec_1.normalize()
            )  # normalization accounts for slight differences in Ca-Ca distances dependent on secondary structure
            if dot_1 > max_dot_1:
                max_dot_1 = dot_1
                remodel_before_loop = i + 1
            vec_2 = pose.residue(begin2 + i + 1).xyz("CA") - pose.residue(
                begin2 + i
            ).xyz("CA")
            dot_2 = vec_12.dot(vec_2.normalize())
            if dot_2 > max_dot_2:
                max_dot_2 = dot_2
                remodel_after_loop = i + 1

    if loop_dssp is None:
        bp_file = remodel_helper(
            pose,
            length,
            remodel_before_loop=remodel_before_loop,
            remodel_after_loop=remodel_after_loop,
            surround_loop_with_helix=surround_loop_with_helix,
        )
    else:
        bp_file = remodel_helper(
            pose,
            length,
            loop_dssp=loop_dssp,
            remodel_before_loop=remodel_before_loop,
            remodel_after_loop=remodel_after_loop,
            surround_loop_with_helix=surround_loop_with_helix,
        )

    bp_sfxn = pyrosetta.create_score_function("fldsgn_cen.wts")
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb, 1.0)
    bp_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint, 1.0
    )
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint, 1.0)
    bp_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0
    )

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <BluePrintBDR name="blueprintbdr" 
            blueprint="{bp_file}" 
            use_abego_bias="0" 
            use_sequence_bias="0" 
            rmdl_attempts="20"/>
        </MOVERS>
        """
    )
    bp_mover = objs.get_mover("blueprintbdr")
    bp_mover.scorefunction(bp_sfxn)

    closure_type = "not_closed"
    for _ in range(attempts):
        bp_mover.apply(pose)
        if pose.num_chains() == 1:
            closure_type = "loop_remodel"
            break

    os.remove(bp_file)

    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, "remodel_before_loop", str(remodel_before_loop)
    )
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, "remodel_after_loop", str(remodel_after_loop)
    )
    return closure_type


@requires_init
def loop_dimer(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    Assumes that pyrosetta.init() has been called with `-corrections:beta_nov16` .
    """
    import sys
    from copy import deepcopy
    from pathlib import Path
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.utils.io import print_timestamp
    from crispy_shifty.protocols.design import (
        gen_std_layer_design,
        gen_task_factory,
        pack_rotamers,
        struct_profile,
        clear_constraints,
        score_wnm,
        score_ss_sc,
    )

    # testing to properly set the TMPDIR on distributed jobs
    # import os
    # os.environ['TMPDIR'] = '/scratch'
    # print(os.environ['TMPDIR'])

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

    for pose in poses:
        scores = dict(pose.scores)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)
        # get parent length from the score
        parent_length = int(float(scores["parent_length"]))

        looped_poses = []
        sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        for chains_to_loop in ["12", "34"]:
            sw.chain_order(chains_to_loop)
            looped_pose = deepcopy(pose)
            sw.apply(looped_pose)

            loop_length = int(parent_length - looped_pose.chain_end(2))

            # is this naive? Phil did something more complicated with residue selectors, looking at the valines.
            # Wondering if I'm missing some edge cases for which this approach doesn't work.
            loop_start = int(looped_pose.chain_end(1)) + 1
            new_loop_str = ",".join(
                str(resi) for resi in range(loop_start, loop_start + loop_length)
            )
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                looped_pose, "new_loop_resis", new_loop_str
            )

            print_timestamp("Attempting closure by loop match...", start_time, end="")
            closure_type = loop_match(looped_pose, loop_length)
            # closure by loop matching was successful, move on to the next dimer or continue to scoring
            # should I use a check like 'pose_to_loop.num_chains() == 1' to determine if the pose is closed?
            if closure_type != "not_closed":
                print("success.")
            else:
                print("failed.")

                print_timestamp(
                    "Attempting closure by loop remodel...", start_time, end=""
                )
                closure_type = loop_remodel(looped_pose, loop_length, 10, 1, 1, True)
                if closure_type != "not_closed":
                    print("success.")
                else:
                    print("failed. Exiting.")
                    # couldn't close this monomer; stop trying with the whole dimer
                    break

            looped_poses.append(looped_pose)

        # if we couldn't close the dimer, continue to the next pose and skip scoring, labeling, and yielding the pose (so nothing is written to disk)
        if closure_type == "not_closed":
            continue

        # The code will only reach here if both loops are closed.
        # Loop closure is fast but has a high failure rate, so more efficient to first see if all loops can be closed,
        # and only design and score if so.

        layer_design = gen_std_layer_design()
        design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
        design_sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
        )

        for looped_pose in looped_poses:

            print_timestamp("Designing loop...", start_time, end="")
            new_loop_sel = (
                pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                    new_loop_str
                )
            )
            design_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
                new_loop_sel, 6, True
            )
            task_factory = gen_task_factory(
                design_sel=design_sel,
                pack_nbhd=True,
                extra_rotamers_level=2,
                limit_arochi=True,
                prune_buns=True,
                upweight_ppi=False,
                restrict_pro_gly=False,
                ifcl=True,  # to respect precompute_ig
                layer_design=layer_design,
            )
            struct_profile(
                looped_pose, design_sel
            )  # Phil's code used eliminate_background=False...
            pack_rotamers(looped_pose, task_factory, design_sfxn)
            clear_constraints(looped_pose)
            print("complete.")

            print_timestamp("Scoring...", start_time, end="")
            total_length = len(looped_pose.residues)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                looped_pose, "total_length", total_length
            )
            dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(looped_pose)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "dssp", dssp)
            tors = get_torsions(looped_pose)
            abego_str = abego_string(tors)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                looped_pose, "abego_str", abego_str
            )

            # score_wnm is fine since it's only one chain
            # should also be fast since the database is already loaded from CCM
            score_wnm(looped_pose)
            score_ss_sc(looped_pose, False, True, "loop_sc")
            print("complete.")

        combined_looped_pose = deepcopy(looped_poses[0])
        pyrosetta.rosetta.core.pose.append_pose_to_pose(
            combined_looped_pose, looped_poses[1], True
        )
        sw.chain_order("12")
        sw.apply(combined_looped_pose)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(combined_looped_pose)

        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                combined_looped_pose, key, value
            )
        for protomer, looped_pose in zip(["A", "B"], looped_poses):
            for key, value in looped_pose.scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(
                    combined_looped_pose, key + "_" + protomer, value
                )

        ppose = io.to_packed(combined_looped_pose)
        yield ppose


@requires_init
def loop_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be looped.
    :param: kwargs: keyword arguments to be passed to looping protocol.
    :return: an iterator of PackedPose objects.
    Assumes that pyrosetta.init() has been called with `-corrections:beta_nov16` .
    `-indexed_structure_store:fragment_store \
    /net/databases/VALL_clustered/connect_chains/ss_grouped_vall_helix_shortLoop.h5`
    TODO check if this is the correct fragment store
    """

    import sys
    from copy import deepcopy
    from pathlib import Path
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        clear_constraints,
        clear_terms_from_scores,
        gen_std_layer_design,
        gen_task_factory,
        pack_rotamers,
        score_ss_sc,
        score_wnm,
        struct_profile,
    )
    from crispy_shifty.protocols.states import clash_check
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
        bb_clash_pre = clash_check(pose)
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
        closure_type = loop_extend(
            pose=pose,
            min_loop_length=min_loop_length,
            max_loop_length=max_loop_length,
            connections="[A+B],C",
        )
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
            bb_clash_post = clash_check(pose)
            scores["bb_clash_delta"] = bb_clash_post - bb_clash_pre
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
        pack_rotamers(
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
        dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(looped_pose)
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
