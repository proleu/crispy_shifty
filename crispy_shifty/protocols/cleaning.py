# Python standard library
from typing import Iterator, Optional, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose

# Custom library imports


@requires_init
def path_to_pose_or_ppose(
    path="",
    cluster_scores: Optional[bool] = False,
    pack_result: Optional[bool] = False,
) -> Iterator[Union[PackedPose, Pose]]:
    """
    :param path: Path to pdb or silent file
    :param cluster_scores: If True, yield objects with cluster scores set after reading
    :param pack_result: If True, yield PackedPose objects instead of Pose objects
    :return: Iterator of Pose or PackedPose objects
    Generate PackedPose objects given an input path to a file on disk to read in.
    Can do pdb, pdb.bz2, pdb.gz or binary silent file formats.
    To use silents, must initialize Rosetta with "-in:file:silent_struct_type binary".
    If `cluster_scores` is set to True, will attempt to set cluster scores from the
    file path if they exist (e.g. has line "REMARK PyRosettaCluster: {"instance": ...}")
    This function can be distributed (best for single inputs) or run on a host process

    """
    import bz2
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed import cluster

    if ".silent" in path:
        pposes = io.poses_from_silent(path)  # returns a generator
    elif ".bz2" in path:
        with open(path, "rb") as f:  # read bz2 bytestream, decompress and decode
            ppose = io.pose_from_pdbstring(bz2.decompress(f.read()).decode())
        if cluster_scores:  # set scores in pose after unpacking, then repack
            try:
                scores = pyrosetta.distributed.cluster.get_scores_dict(path)["scores"]
            except IOError:
                print("Scores may be absent or incorrectly formatted")
                scores = {}
            pose = io.to_pose(ppose)
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(value))
            ppose = io.to_packed(pose)
        else:
            pass
        pposes = [ppose]
    elif ".pdb" in path:  # should handle pdb.gz as well
        ppose = io.pose_from_file(path)
        if cluster_scores:  # set scores in pose after unpacking, then repack
            try:
                scores = pyrosetta.distributed.cluster.get_scores_dict(path)["scores"]
            except IOError:
                print("Scores may be absent or incorrectly formatted")
                scores = {}
            pose = io.to_pose(ppose)
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(value))
            ppose = io.to_packed(pose)
        else:
            pass
        pposes = [ppose]
    else:
        raise RuntimeError("Must provide a pdb, pdb.gz, pdb.bz2, or binary silent")
    for ppose in pposes:
        if pack_result:
            yield ppose
        else:
            pose = io.to_pose(ppose)
            yield pose


@requires_init
def remove_terminal_loops(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose to remove terminal loops from.
    :param kwargs: kwargs such as "pdb_path" and "metadata".
    :return: Iterator of PackedPose objects with terminal loops removed.
    Use DSSP and delete region mover to idealize inputs. Add metadata.
    Trims chain A. Must provide either a packed_pose_in or "pdb_path" kwarg.
    """

    import sys
    from pathlib import Path
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.scoring.dssp import Dssp

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    if "chains_to_keep" in kwargs:
        chains_to_keep = kwargs["chains_to_keep"]
    else:
        chains_to_keep = "1"

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=False, pack_result=False
        )
    final_pposes = []
    for pose in poses:
        # setup rechain mover to sanitize the trimmed pose
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        rechain.chain_order(chains_to_keep)
        trimmed_pose = pose.clone()
        rechain.apply(trimmed_pose)
        # get secondary structure
        Dssp(pose).insert_ss_into_pose(trimmed_pose, True)
        dssp = trimmed_pose.secstruct()
        # get leading loop from ss
        if dssp[0] == "H":  # in case no leading loop is detected
            pass
        elif dssp[0:2] == "LH":  # leave it alone if it has a short terminal loop
            pass
        else:  # get beginning index of first occurrence of LH in dssp
            rosetta_idx_n_term = dssp.find("LH") + 1
            # setup trimming mover
            trimmer = (
                pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
            )
            trimmer.region(str(trimmed_pose.chain_begin(1)), str(rosetta_idx_n_term))
            trimmer.apply(trimmed_pose)
            rechain.apply(trimmed_pose)
        # get secondary structure
        Dssp(trimmed_pose).insert_ss_into_pose(trimmed_pose, True)
        dssp = trimmed_pose.secstruct()
        # get trailing loop from ss
        if dssp[-1] == "H":  # in case no trailing loop is detected
            pass
        elif dssp[-2:] == "HL":
            pass
        else:  # get ending index of last occurrence of HL in dssp
            # rosetta_idx_c_term = str(dssp.rfind("HL") + 2)
            # setup trimming mover
            # trimmer = (
            #     pyrosetta.rosetta.protocols.grafting.simple_movers.KeepRegionMover()
            # )
            # trimmer.start("1")
            # trimmer.end(rosetta_idx_c_term)
            rosetta_idx_c_term = dssp.rfind("HL") + 2
            # setup trimming mover
            trimmer = (
                pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
            )
            trimmer.region(str(rosetta_idx_c_term), str(trimmed_pose.chain_end(1)))
            trimmer.apply(trimmed_pose)
            rechain.apply(trimmed_pose)
        # trimmed_length = len(trimmed_pose.residues)
        trimmed_length = trimmed_pose.chain_end(1)
        if "metadata" in kwargs:
            metadata = kwargs["metadata"]
        else:
            metadata = {}
        metadata["trim_n"] = rosetta_idx_n_term
        metadata["trimmed_length"] = str(trimmed_length)
        for key, value in metadata.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(trimmed_pose, key, str(value))
        final_pposes.append(io.to_packed(trimmed_pose))

    for ppose in final_pposes:
        yield ppose


def break_all_disulfides(pose: Pose) -> Pose:
    """
    :param pose: Pose to break disulfides in
    :return: Pose with all disulfides broken
    Quickly break all disulfides in a pose
    """
    import pyrosetta

    seq = pose.sequence()
    all_cys_resi_indexes = [i for i, r in enumerate(seq, start=1) if r == "C"]
    for i in all_cys_resi_indexes:
        for j in all_cys_resi_indexes:
            if pyrosetta.rosetta.core.conformation.is_disulfide_bond(
                pose.conformation(), i, j
            ):
                pyrosetta.rosetta.core.conformation.break_disulfide(
                    pose.conformation(), i, j
                )
            else:
                pass
    return pose


@requires_init
def redesign_disulfides(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose object.
    :param kwargs: kwargs such as "pdb_path".
    :return: Iterator of PackedPose objects with disulfides redesigned.
    fixbb fastdesign with beta_nov16 on all cys residues using layerdesign.
    Requires the following init flags:
    `-corrections::beta_nov16 true`
    `-detect_disulf false`
    `-holes:dalphaball /software/rosetta/DAlphaBall.gcc`
    """
    from pathlib import Path
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import (
        path_to_pose_or_ppose,
        break_all_disulfides,
    )

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=False, pack_result=False
        )

    pyrosetta.rosetta.basic.options.set_boolean_option("in:detect_disulf", False)

    xml = """
    <ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="sfxn" weights="beta_nov16" symmetric="0"/>
    </SCOREFXNS>
    <RESIDUE_SELECTORS>
        <ResidueName name="cys" residue_name3="CYS"/>
        <Not name="not_cys" selector="cys"/>
        <Neighborhood name="around_cys" selector="cys"/>
        <Or name="cys_or_around_cys" selectors="cys,around_cys"/>
        <Not name="not_cys_or_around_cys" selector="cys_or_around_cys"/>
        <Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
        <Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
        <Layer name="core" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true"/>
        <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
        <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
        <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
        <And name="helix_cap" selectors="entire_loop">
            <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
        </And>
        <And name="helix_start" selectors="entire_helix">
            <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
        </And>
        <And name="helix" selectors="entire_helix">
            <Not selector="helix_start"/>
        </And>
        <And name="loop" selectors="entire_loop">
            <Not selector="helix_cap"/>
        </And>
        <Chain name="chA" chains="A" />
    </RESIDUE_SELECTORS>
    <TASKOPERATIONS>
        <RestrictAbsentCanonicalAAS name="design" keep_aas="ADEFGHIKLMNPQRSTVWY"/>
        <OperateOnResidueSubset name="pack" selector="not_cys">
        <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>
        <OperateOnResidueSubset name="lock" selector="not_cys_or_around_cys">
        <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <DesignRestrictions name="layer_design">
        <Action selector_logic="surface AND helix_start"  aas="DEHKPQR"/>
        <Action selector_logic="surface AND helix"        aas="EHKQR"/>
        <Action selector_logic="surface AND sheet"        aas="EHKNQRST"/>
        <Action selector_logic="surface AND loop"         aas="DEGHKNPQRST"/>
        <Action selector_logic="boundary AND helix_start" aas="ADEHIKLNPQRSTVWY"/>
        <Action selector_logic="boundary AND helix"       aas="ADEHIKLNQRSTVWYM"/>
        <Action selector_logic="boundary AND sheet"       aas="DEFHIKLNQRSTVWY"/>
        <Action selector_logic="boundary AND loop"        aas="ADEFGHIKLNPQRSTVWY"/>
        <Action selector_logic="core AND helix_start"     aas="AFILVW"/>
        <Action selector_logic="core AND helix"           aas="AFILVW"/>
        <Action selector_logic="core AND sheet"           aas="FILVWY"/>
        <Action selector_logic="core AND loop"            aas="AFGILPVW"/>
        <Action selector_logic="helix_cap"                aas="DNSTP"/>
        </DesignRestrictions>
        <LimitAromaChi2 name="arochi" />
        <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1" />
    </TASKOPERATIONS>
    <FILTERS>
        <BuriedUnsatHbonds name="buns_parent" use_reporter_behavior="true" report_all_heavy_atom_unsats="true" scorefxn="sfxn" residue_selector="chA" ignore_surface_res="false" print_out_info_to_pdb="false" confidence="0" use_ddG_style="true" burial_cutoff="0.01" dalphaball_sasa="true" probe_radius="1.1" max_hbond_energy="1.5" burial_cutoff_apo="0.2" />
        <ExposedHydrophobics name="exposed_hydrophobics_parent" />
        <Geometry name="geometry_parent" confidence="0" />
        <Holes name="holes_core_parent" threshold="0.0" residue_selector="core" confidence="0"/>
        <Holes name="holes_all_parent" threshold="0.0" confidence="0"/>
        <SSPrediction name="mismatch_probability_parent" confidence="0" cmd="/software/psipred4/runpsipred_single" use_probability="1" mismatch_probability="1" use_svm="1" />
        <PackStat name="packstat_parent" threshold="0" chain="0" repeats="5"/>
        <SSShapeComplementarity name="sc_all_parent" verbose="1" loops="1" helices="1" />
        <ScoreType name="total_score_pose" scorefxn="sfxn" score_type="total_score" threshold="0" confidence="0" />
        <ResidueCount name="count" />
        <CalculatorFilter name="score_per_res_parent" equation="total_score_full / res" threshold="-2.0" confidence="0">
            <Var name="total_score_full" filter="total_score_pose"/>
            <Var name="res" filter="count"/>
        </CalculatorFilter>
    </FILTERS>
    <SIMPLE_METRICS>
        <SapScoreMetric name="sap_parent" />
    </SIMPLE_METRICS>
    <MOVERS>
        <FastDesign name="fast_design" scorefxn="sfxn" repeats="2" task_operations="design,pack,lock,layer_design,arochi,ex1_ex2">
            <MoveMap name="mm" chi="true" bb="false" jump="false" />
        </FastDesign>
    </MOVERS>
    <PROTOCOLS>
        <Add mover="fast_design"/>
        <Add filter_name="buns_parent" />
        <Add filter_name="exposed_hydrophobics_parent" />
        <Add filter_name="geometry_parent" />
        <Add filter_name="holes_core_parent"/>
        <Add filter_name="holes_all_parent"/>
        <Add filter_name="mismatch_probability_parent" />
        <Add filter_name="packstat_parent"/>
        <Add filter_name="sc_all_parent" />
        <Add filter_name="score_per_res_parent" />
        <Add metrics="sap_parent" labels="sap_parent" />
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    design_score = SingleoutputRosettaScriptsTask(xml)

    final_pposes = []
    for pose in poses:
        scores = dict(pose.scores)
        pose = break_all_disulfides(pose)
        designed_ppose = design_score(pose.clone())
        pose = io.to_pose(designed_ppose)
        scores.update(dict(pose.scores))
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                pose, key, str(value)
            )  # store values as strings for safety
        designed_ppose = io.to_packed(pose)
        final_pposes.append(designed_ppose)
    for ppose in final_pposes:
        yield ppose


@requires_init
def prep_input_scaffold(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose object.
    :param kwargs: kwargs such as "pdb_path".
    :return: Iterator of PackedPose objects with ends trimmed and disulfides redesigned.
    First removes trailing loops, adds metadata, then does design to remove disulfides.
    Design is fixbb fastdesign with beta_nov16 on all cys residues using layerdesign.
    Requires the following init flags:
    -corrections::beta_nov16 true
    -detect_disulf false
    -holes:dalphaball /software/rosetta/DAlphaBall.gcc
    """
    from pathlib import Path
    import sys
    import pandas as pd
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import (
        path_to_pose_or_ppose,
        break_all_disulfides,
        redesign_disulfides,
        remove_terminal_loops,
    )

    # get path to metadata from kwargs
    metadata_csv = kwargs.pop("metadata_csv")
    key = kwargs["pdb_path"]

    metadata = dict(pd.read_csv(metadata_csv, index_col="pdb").loc[key])
    metadata_to_keep = [
        "topo",
        "best_model",
        "best_average_plddts",
        "best_ptm",
        "best_rmsd_to_input",
        "best_average_DAN_plddts",
        "scaffold_type",
    ]
    metadata = {k: v for k, v in metadata.items() if k in metadata_to_keep}
    metadata["pdb"] = key

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=False, pack_result=False
        )
    for pose in poses:
        for trimmed_ppose in remove_terminal_loops(
            packed_pose_in=io.to_packed(pose), metadata=metadata, **kwargs
        ):
            for final_ppose in redesign_disulfides(trimmed_ppose, **kwargs):
                yield final_ppose


@requires_init
def add_metadata_to_input(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose object.
    :param kwargs: kwargs such as "pdb_path".
    :return: Iterator of PackedPose objects with ends trimmed.
    Removes trailing loops and adds metadata. Repeat_len is only meaningful for
    repetitive scaffolds.
    Requires the following init flags:
    -corrections::beta_nov16 true
    -holes:dalphaball /software/rosetta/DAlphaBall.gcc
    """
    from pathlib import Path
    import pandas as pd
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import (
        path_to_pose_or_ppose,
        remove_terminal_loops,
    )

    key = kwargs["pdb_path"]
    metadata = {}
    metadata["pdb"] = key

    skip_trimming = False
    if "metadata_csv" in kwargs:
        metadata_series = pd.read_csv(kwargs["metadata_csv"], index_col="pdb").loc[key, :]
        if metadata_series["skip_trimming"] == True:
            skip_trimming = True

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = [pose for pose in path_to_pose_or_ppose(
            path=key, cluster_scores=False, pack_result=False
        )]

    # skip_trimming = False
    # if "skip_trimming" in kwargs:
    #     skip_trimming_str = kwargs["skip_trimming"]
    #     if "metadata_csv" in kwargs:
    #         if metadata_series["skip_trimming"].lower() == "true":
    #             skip_trimming = True
    #     else:
    #         if skip_trimming_str.lower() == "true":
    #             skip_trimming = True
    if "num_ss_per_repeat" in kwargs:
        num_ss_per_repeat = int(kwargs["num_ss_per_repeat"])
    else:
        num_ss_per_repeat = 2 # DHR, for example
    # get the fixed resis from the metadata csv, then make two combinations: one with
    # the important resis, and one with both important and semiimportant resis
    if "fixed_resis" in kwargs:
        fixed_resis_option = kwargs["fixed_resis"]
        if fixed_resis_option not in ["distribute","exact"]:
            raise ValueError("fixed_resis must be either 'distribute' or 'exact'")
        resis_1 = [int(x) for x in metadata_series["Important"].astype(str).split(' ')]
        resis_2 = [int(x) for x in metadata_series["Semiimportant"].astype(str).split(' ')]
        resis_list = [resis_1] * len(poses)
        if resis_2:
            resis_list += [resis_2] * len(poses)
            poses += poses # double the length of the input pose list to match
    else:
        fixed_resis_option = False
        resis_list = [None] * len(poses)

    xml = """
    <ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="sfxn" weights="beta_nov16" symmetric="0"/>
    </SCOREFXNS>
    <RESIDUE_SELECTORS>
        <Chain name="chA" chains="A" />
    </RESIDUE_SELECTORS>
    <TASKOPERATIONS>
    </TASKOPERATIONS>
    <FILTERS>
        <BuriedUnsatHbonds name="buns_parent" use_reporter_behavior="true" report_all_heavy_atom_unsats="true" scorefxn="sfxn" residue_selector="chA" ignore_surface_res="false" print_out_info_to_pdb="false" confidence="0" use_ddG_style="true" burial_cutoff="0.01" dalphaball_sasa="true" probe_radius="1.1" max_hbond_energy="1.5" burial_cutoff_apo="0.2" />
        <ExposedHydrophobics name="exposed_hydrophobics_parent" />
        <Geometry name="geometry_parent" confidence="0" />
        <Holes name="holes_core_parent" threshold="0.0" residue_selector="core" confidence="0"/>
        <Holes name="holes_all_parent" threshold="0.0" confidence="0"/>
        <SSPrediction name="mismatch_probability_parent" confidence="0" cmd="/software/psipred4/runpsipred_single" use_probability="1" mismatch_probability="1" use_svm="1" />
        <PackStat name="packstat_parent" threshold="0" chain="0" repeats="5"/>
        <SSShapeComplementarity name="sc_all_parent" verbose="1" loops="1" helices="1" />
        <ScoreType name="total_score_pose" scorefxn="sfxn" score_type="total_score" threshold="0" confidence="0" />
        <ResidueCount name="count" />
        <CalculatorFilter name="score_per_res_parent" equation="total_score_full / res" threshold="-2.0" confidence="0">
            <Var name="total_score_full" filter="total_score_pose"/>
            <Var name="res" filter="count"/>
        </CalculatorFilter>
    </FILTERS>
    <SIMPLE_METRICS>
        <SapScoreMetric name="sap_parent" />
    </SIMPLE_METRICS>
    <MOVERS>
    </MOVERS>
    <PROTOCOLS>
        <Add filter_name="buns_parent" />
        <Add filter_name="exposed_hydrophobics_parent" />
        <Add filter_name="geometry_parent" />
        <Add filter_name="holes_core_parent"/>
        <Add filter_name="holes_all_parent"/>
        <Add filter_name="mismatch_probability_parent" />
        <Add filter_name="packstat_parent"/>
        <Add filter_name="sc_all_parent" />
        <Add filter_name="score_per_res_parent" />
        <Add metrics="sap_parent" labels="sap_parent" />
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    score_pose = SingleoutputRosettaScriptsTask(xml)

    final_pposes = []
    for pose, fixed_resis in zip(poses, resis_list):

        # add topology to pose
        ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
        topo = ""
        ss_counter = 0
        repeat_start = 0
        repeat_end = 0
        for i in range(1, pose.chain_end(1)):
            ss_i = ss.get_dssp_secstruct(i)
            if ss_i != ss.get_dssp_secstruct(i + 1):
                if ss_i != 'L':
                    topo += ss_i
                    if ss_counter == 0:
                        repeat_start = i
                    if ss_counter == num_ss_per_repeat:
                        repeat_end = i
                    ss_counter += 1
        ss_i = ss.get_dssp_secstruct(pose.chain_end(1))
        if ss_i != 'L':
            topo += ss_i
        repeat_len = repeat_end - repeat_start
        metadata["topo"] = topo
        metadata["repeat_len"] = repeat_len

        if skip_trimming:
            trimmed_pose = pose
            trimmed_length = trimmed_pose.chain_end(1)
            metadata["trim_n"] = 0
            metadata["trimmed_length"] = str(trimmed_length)
            for key, value in metadata.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(trimmed_pose, key, str(value))
        else:
            for trimmed_ppose in remove_terminal_loops(
                packed_pose_in=io.to_packed(pose), metadata=metadata, **kwargs
            ):
                trimmed_pose = io.to_pose(trimmed_ppose)
                break

        if fixed_resis_option:

            trim_n = metadata["trim_n"]
            fixed_resis = [x-trim_n for x in fixed_resis]

            if fixed_resis_option == "distribute":
                trimmed_length = metadata["trimmed_length"]
                full_fixed_resis = []
                for fixed_resi in fixed_resis:
                    i = fixed_resi - repeat_len
                    while i > 0:
                        full_fixed_resis.append(fixed_resi)
                        i -= repeat_len
                    i = fixed_resi
                    while i <= trimmed_length:
                        full_fixed_resis.append(fixed_resi)
                        i += repeat_len
                fixed_resis = full_fixed_resis

            metadata["fixed_resis"] = ','.join(str(x) for x in fixed_resis)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(trimmed_pose, "fixed_resis", metadata["fixed_resis"])

        scores = dict(trimmed_pose.scores)
        designed_ppose = score_pose(trimmed_pose.clone())
        designed_pose = io.to_pose(designed_ppose)
        scores.update(dict(designed_pose.scores))
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                designed_pose, key, str(value)
            )  # store values as strings for safety
        designed_ppose = io.to_packed(designed_pose)
        final_pposes.append(designed_ppose)
    for ppose in final_pposes:
        yield ppose


def trim_and_resurface_peptide(pose: Pose) -> Pose:
    """
    :param pose: Pose object with peptide to trim and resurface.
    :return: Pose object with trimmed and resurfaced peptide.
    If peptide is longer than 30 residues, it is trimmed to 28 or total length - 12,
    whichever is longer.
    If peptide pI is greater than 6.0 it will be resurfaced, targeting a goal pI of less
    than 5.0.
    Assumes that pyrosetta.init() has been called with `-corrections:beta_nov16`.
    """
    from pathlib import Path
    import sys
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NeighborhoodResidueSelector,
        NotResidueSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )
    from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.design import (
        clear_constraints,
        gen_std_layer_design,
        gen_task_factory,
        interface_among_chains,
        pack_rotamers,
        score_cms,
        score_per_res,
    )

    # preserve the original pose scores
    scores = dict(pose.scores)
    # get the length of the peptide
    chB = pose.clone()
    rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    rechain.chain_order("2")
    rechain.apply(chB)
    chB_length = chB.total_residue()
    # if length is above 28, try to trim the peptide down to 28
    if chB_length > 28:
        # get trimmable regions
        chB_residue_indices = [
            str(i) for i in range(pose.chain_begin(2), pose.chain_end(2) + 1)
        ]
        internal = chB_residue_indices[6:-6]
        # now we decide whether to do a sliding window approach or serial truncation
        if len(internal) > 28:
            # get all possible sub windows of length internal and score them with cms
            window_size = len(internal)
        else:
            # get all possible sub windows of length 28
            window_size = 28
        # get all possible sub windows of length window_size from chB_residue_indices
        sub_windows = [
            chB_residue_indices[i : i + window_size]
            for i in range(len(chB_residue_indices) - window_size + 1)
        ]
        # make a dict keyed by sub_window and valued by pose trimmed to that window
        truncations_dict = {}
        # make a dict keyed by sub_window and valued score of cms
        cms_scores_dict = {}
        # for each sub_window, make a pose and trim it to the sub_window
        for sub_window in sub_windows:
            # make a residue selector that includes all residues in sub_window
            indices = ",".join(sub_window)
            sub_window_sel = ResidueIndexSelector(indices)
            chA_sel, chB_sel, chC_sel = (
                ChainSelector(1),
                ChainSelector(2),
                ChainSelector(3),
            )
            chA_chC_sel = OrResidueSelector(chA_sel, chC_sel)
            to_keep_sel = OrResidueSelector(sub_window_sel, chA_chC_sel)
            to_delete_sel = NotResidueSelector(to_keep_sel)
            # setup the delete region mover
            trimmed_pose = pose.clone()
            trimmer = DeleteRegionMover()
            trimmer.set_rechain(True)
            trimmer.set_add_terminal_types_on_rechain(True)
            trimmer.set_residue_selector(to_delete_sel)
            trimmer.apply(trimmed_pose)
            # fix the pdb_info
            rechain.chain_order("123")
            rechain.apply(trimmed_pose)
            # score the pose
            cms = score_cms(
                pose=trimmed_pose,
                sel_1=chA_sel,
                sel_2=chB_sel,
            )
            # add to the dicts
            truncations_dict[indices] = trimmed_pose
            cms_scores_dict[indices] = cms
        # invert the dict, this destroys any ties but we don't care
        inverted_cms_scores_dict = {v: k for k, v in cms_scores_dict.items()}
        # get the sub_window with the highest cms score
        highest_cms_sub_window = inverted_cms_scores_dict[max(inverted_cms_scores_dict)]
        # get the pose with the highest cms score
        highest_cms_pose = truncations_dict[highest_cms_sub_window]
        # exit the if statement with the highest cms score pose set to pose
        pose = highest_cms_pose
    else:  # don't need to trim if already 30 or less
        pass
    # try to mutate non-interface surface residues to GLU and generally resurface chB
    chB = pose.clone()
    rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    rechain.chain_order("2")
    rechain.apply(chB)
    # get the interface residues
    interface_sel = interface_among_chains([1, 2], vector_mode=True)
    # get the non-interface residues
    non_interface_sel = NotResidueSelector(interface_sel)
    # get the residues to mutate
    non_interface_chB_sel = AndResidueSelector(non_interface_sel, ChainSelector(2))
    design_sel = non_interface_chB_sel
    pack_sel = NeighborhoodResidueSelector(design_sel, 10)
    # make sfxn with constraints
    sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    sfxn_clean = pyrosetta.create_score_function("beta_nov16.wts")
    sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.aa_composition, 1.0)
    sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.netcharge, 1.0)
    sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0)
    # make layer_design dict
    layer_design = gen_std_layer_design()
    # make a task factory
    task_factory = gen_task_factory(
        design_sel=design_sel,
        pack_sel=pack_sel,
        extra_rotamers_level=2,
        limit_arochi=True,
        prune_buns=False,
        upweight_ppi=False,
        restrict_pro_gly=True,
        precompute_ig=False,
        ifcl=True,
        layer_design=layer_design,
    )
    # add a net charge constraint
    chg = pyrosetta.rosetta.protocols.aa_composition.AddNetChargeConstraintMover()
    # make the file contents
    file_contents = """
    DESIRED_CHARGE -4
    PENALTIES_CHARGE_RANGE -7 -1
    PENALTIES 3 0 0 0 2 4 6
    BEFORE_FUNCTION LINEAR
    AFTER_FUNCTION LINEAR
    """
    chg.create_constraint_from_file_contents(file_contents)
    chg.add_residue_selector(ChainSelector(2))
    chg.apply(pose)
    # pack rotamers up to 5 times trying to get a pI under 5
    for _ in range(0, 5):
        pack_rotamers(
            pose,
            task_factory,
            sfxn,
        )
        chB_seq = list(pose.split_by_chain())[1].sequence()
        # recheck pI
        chB_pI = ProteinAnalysis(chB_seq).isoelectric_point()
        # break if below 5, else continue
        if chB_pI <= 5:
            break
        else:
            continue
    scores["B_final_seq"] = list(pose.split_by_chain())[1].sequence()
    # reset scores
    for key, value in scores.items():
        pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
    return pose


@requires_init
def finalize_peptide(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose object.
    :param kwargs: kwargs such as "pdb_path".
    :return: Iterator of PackedPose objects with chB trimmed and resurfaced.
    """

    from operator import lt, gt
    from pathlib import Path
    import sys
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NeighborhoodResidueSelector,
        NotResidueSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )
    from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import (
        path_to_pose_or_ppose,
    )
    from crispy_shifty.protocols.design import score_cms
    from crispy_shifty.protocols.folding import SuperfoldRunner
    from crispy_shifty.protocols.mpnn import MPNNDesign
    from crispy_shifty.utils.io import print_timestamp

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=True, pack_result=False
        )
    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
         # get the length of the peptide
        chB = pose.clone()
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        rechain.chain_order("2")
        rechain.apply(chB)
        chB_length = chB.total_residue()
        # if length is above 28, try to trim the peptide down to 28
        if chB_length > 28:
            # get trimmable regions
            chB_residue_indices = [
                str(i) for i in range(pose.chain_begin(2), pose.chain_end(2) + 1)
            ]
            internal = chB_residue_indices[5:-5]
            # now we decide whether to do a sliding window approach or serial truncation
            if len(internal) > 28:
                # get all possible sub windows of length internal and score them with cms
                window_size = len(internal)
            else:
                # get all possible sub windows of length 28
                window_size = 28
            # get all possible sub windows of length window_size from chB_residue_indices
            sub_windows = [
                chB_residue_indices[i : i + window_size]
                for i in range(len(chB_residue_indices) - window_size + 1)
            ]
            # make a dict keyed by sub_window and valued by pose trimmed to that window
            truncations_dict = {}
            # make a dict keyed by sub_window and valued score of cms
            cms_scores_dict = {}
            # for each sub_window, make a pose and trim it to the sub_window
            for sub_window in sub_windows:
                # make a residue selector that includes all residues in sub_window
                indices = ",".join(sub_window)
                sub_window_sel = ResidueIndexSelector(indices)
                chA_sel, chB_sel, chC_sel = (
                    ChainSelector(1),
                    ChainSelector(2),
                    ChainSelector(3),
                )
                chA_chC_sel = OrResidueSelector(chA_sel, chC_sel)
                to_keep_sel = OrResidueSelector(sub_window_sel, chA_chC_sel)
                to_delete_sel = NotResidueSelector(to_keep_sel)
                # setup the delete region mover
                trimmed_pose = pose.clone()
                trimmer = DeleteRegionMover()
                trimmer.set_rechain(True)
                trimmer.set_add_terminal_types_on_rechain(True)
                trimmer.set_residue_selector(to_delete_sel)
                trimmer.apply(trimmed_pose)
                # fix the pdb_info
                rechain.chain_order("123")
                rechain.apply(trimmed_pose)
                # score the pose
                cms = score_cms(
                    pose=trimmed_pose,
                    sel_1=chA_sel,
                    sel_2=chB_sel,
                )
                # add to the dicts
                truncations_dict[indices] = trimmed_pose
                cms_scores_dict[indices] = cms
            # invert the dict, this destroys any ties but we don't care
            inverted_cms_scores_dict = {v: k for k, v in cms_scores_dict.items()}
            # get the sub_window with the highest cms score
            highest_cms_sub_window = inverted_cms_scores_dict[max(inverted_cms_scores_dict)]
            # get the pose with the highest cms score
            highest_cms_pose = truncations_dict[highest_cms_sub_window]
            # exit the if statement with the highest cms score pose set to pose
            pose = highest_cms_pose
        else:  # don't need to trim if already 28 or less
            pass   
        # now we have a pose with the correct length, we can resurface it
        trimmed_pose = pose.clone()

        print_timestamp("Setting up design selectors", start_time)
        # make a designable residue selector of only the chB residues
        design_sel = ChainSelector(2)
        print_timestamp("Designing interface with MPNN", start_time)
        # construct the MPNNDesign object
        mpnn_design = MPNNDesign(
            design_selector=design_sel,
            num_sequences=24,
            omit_AAs="CX",
            temperature=0.1,
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # get rid of sequences that don't pass protparams filters
        print_timestamp("Filtering sequences on sequence metrics", start_time)
        # make filter_dict
        filter_dict = {
            "pI": -5.0,
        }
        chB_seqs = {k: v for k, v in scores.items() if "mpnn_seq" in k}
        # remove sequences that don't pass the filter
        # for seq_id, seq in chB_seqs.items():

        #     chB_pI = ProteinAnalysis(seq).isoelectric_point()
        #     # remove scores that don't pass the filter from scores and pose datacache
        #     if not chB_pI < filter_dict["pI"]:
        #         scores.pop(seq_id)
        #         pyrosetta.rosetta.core.pose.clearPoseExtraScore(pose, seq_id)
        #     else:
        #         pass
        # print_timestamp("Filtering sequences with AF2", start_time)

        # # load fasta into a dict
        # tmp_fasta_dict = fasta_to_dict(fasta_path)
        # pose_chains = list(pose.split_by_chain())
        # # slice out the bound state, aka chains A and B
        # tmp_pose, X_pose = Pose(), Pose()
        # pyrosetta.rosetta.core.pose.append_pose_to_pose(
        #     tmp_pose, pose_chains[0], new_chain=True
        # )
        # pyrosetta.rosetta.core.pose.append_pose_to_pose(
        #     tmp_pose, pose_chains[1], new_chain=True
        # )
        # # slice out the free state, aka chain C
        # pyrosetta.rosetta.core.pose.append_pose_to_pose(
        #     X_pose, pose_chains[2], new_chain=True
        # )
        # # fix the fasta by splitting on chainbreaks '/' and rejoining the first two
        # tmp_fasta_dict = {
        #     tag: "/".join(seq.split("/")[0:2]) for tag, seq in tmp_fasta_dict.items()
        # }
        # # change the pose to the modified pose
        # pose = tmp_pose.clone()
        # print_timestamp("Setting up for AF2", start_time)
        # runner = SuperfoldRunner(
        #     pose=pose, fasta_path=fasta_path, load_decoys=True, **kwargs
        # )
        # runner.setup_runner(file=fasta_path)
        # # initial_guess, reference_pdb both are the tmp.pdb
        # initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        # reference_pdb = initial_guess
        # flag_update = {
        #     "--initial_guess": initial_guess,
        #     "--reference_pdb": reference_pdb,
        # }
        # # now we have to point to the right fasta file
        # new_fasta_path = str(Path(runner.get_tmpdir()) / "tmp.fa")
        # dict_to_fasta(tmp_fasta_dict, new_fasta_path)
        # runner.set_fasta_path(new_fasta_path)
        # runner.override_input_file(new_fasta_path)
        # runner.update_flags(flag_update)
        # runner.update_command()
        # print_timestamp("Running AF2", start_time)
        # runner.apply(pose)
        # print_timestamp("AF2 complete, updating pose datacache", start_time)
        # # update the scores dict
        # scores.update(pose.scores)
        # # update the pose with the updated scores dict
        # for key, value in scores.items():
        #     pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # # setup prefix, rank_on, filter_dict (in this case we can't get from kwargs)
        # filter_dict = {
        #     "mean_plddt": (gt, 92.0),
        #     "rmsd_to_reference": (lt, 1.5),
        #     "mean_pae_interaction": (lt, 5),
        # }
        # rank_on = "mean_plddt"
        # prefix = "mpnn_seq"
        # print_timestamp("Generating decoys", start_time)
        # for decoy in generate_decoys_from_pose(
        #     pose,
        #     filter_dict=filter_dict,
        #     generate_prediction_decoys=True,
        #     label_first=True,
        #     prefix=prefix,
        #     rank_on=rank_on,
        # ):
        #     # add the free state back into the decoy
        #     pyrosetta.rosetta.core.pose.append_pose_to_pose(
        #         decoy, X_pose, new_chain=True
        #     )
        #     # get the chA sequence
        #     chA_seq = list(decoy.split_by_chain())[0].sequence()
        #     # setup SimpleThreadingMover
        #     stm = pyrosetta.rosetta.protocols.simple_moves.SimpleThreadingMover()
        #     # thread the sequence from chA onto chA
        #     stm.set_sequence(chA_seq, start_position=decoy.chain_begin(3))
        #     stm.apply(decoy)
        #     # rename af2 metrics to have Y_ prefix
        #     decoy_scores = dict(decoy.scores)
        #     for key, value in decoy_scores.items():
        #         if key in af2_metrics:
        #             pyrosetta.rosetta.core.pose.setPoseExtraScore(
        #                 decoy, f"Y_{key}", value
        #             )

        #     packed_decoy = io.to_packed(decoy)
        #     yield packed_decoy



        # for key, value in scores.items():
        #     pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(value))
        # final_ppose = io.to_packed(pose)
        # yield final_ppose
