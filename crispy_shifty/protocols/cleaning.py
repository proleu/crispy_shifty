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
            except UnboundLocalError:
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
            except UnboundLocalError:
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
    Assumes a monomer. Must provide either a packed_pose_in or "pdb_path" kwarg.
    """

    import sys
    from pathlib import Path
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.scoring.dssp import Dssp

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

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
        rechain.chain_order("1")
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
            rosetta_idx_n_term = str(dssp.find("LH") + 1)
            # setup trimming mover
            trimmer = (
                pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
            )
            trimmer.region(str(trimmed_pose.chain_begin(1)), rosetta_idx_n_term)
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
            rosetta_idx_c_term = str(dssp.rfind("HL") + 2)
            # setup trimming mover
            trimmer = (
                pyrosetta.rosetta.protocols.grafting.simple_movers.KeepRegionMover()
            )
            trimmer.start("1")
            trimmer.end(rosetta_idx_c_term)
            trimmer.apply(trimmed_pose)
            rechain.apply(trimmed_pose)
        trimmed_length = len(trimmed_pose.residues)
        if "metadata" in kwargs:
            metadata = kwargs["metadata"]
        else:
            metadata = {}
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
        for trimmed_ppose in remove_terminal_loops(packed_pose_in=io.to_packed(pose), metadata=metadata, **kwargs):
            for final_ppose in redesign_disulfides(trimmed_ppose, **kwargs):
                yield final_ppose


def trim_and_resurface_peptide(pose: Pose) -> Pose:
    """
    TODO
    """
    from pathlib import Path
    import sys
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NotResidueSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )
    
    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.design import interface_among_chains, score_cms
    # get the length and the pI of the peptide
    chB = pose.clone()
    rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    rechain.chain_order("2")
    rechain.apply(chB)
    chB_length = chB.total_residue()
    # if length is above 31, try to trim the peptide down to 31
    if chB_length > 31:
        # get trimmable regions
        # interface_sel = interface_among_chains([1, 2], vector_mode = True)
        # not_interface_sel = NotResidueSelector(interface_sel)
        chB_residue_indices = [str(i) for i in range(pose.chain_begin(2), pose.chain_end(2) + 1)]
        n_term, c_term = chB_residue_indices[:5], chB_residue_indices[-5:]
        # n_term_sel = ResidueIndexSelector(",".join(n_term))
        # c_term_sel = ResidueIndexSelector(",".join(c_term))
        length = chB_length
        while length > 31:
            end_indices = n_term[0], c_term[-1]
            scores_post_del = {}
            pose_del = pose.clone()
            for index in end_indices:
                # setup trimmer
                trimmer = pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
                trimmer.set_rechain(True)
                trimmer.set_add_terminal_types_on_rechain(True)
                trimmer.set_residue_selector(ResidueIndexSelector(index))
                trimmer.apply(pose_del)
                # score pose cms
                scores_post_del[index] = score_cms(pose_del)
            # get the lowest scoring index from the scores




    # TODO

    chB_pI = ProteinAnalysis(chB.sequence()).isoelectric_point()
    # if pI is above 6, try to mutate non-interface surface residues to GLU
    # while pI is above 6, score mutating every surface non interface residue to GLU
    # pick the best mutation
    # recheck pI
    # break if below 6, else continue

@requires_init
def finalize_peptide(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose object.
    :param kwargs: kwargs such as "pdb_path".
    :return: Iterator of PackedPose objects with chB trimmed and resurfaced.
    Requires the following init flags:
    -corrections::beta_nov16 true
    """
    from pathlib import Path
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import (
        path_to_pose_or_ppose,
        trim_and_resurface_peptide,
    )

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=True, pack_result=False
        )
    for pose in poses:
        scores = dict(pose.scores)
        pose = trim_and_resurface_peptide(pose)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                pose, key, str(value)
            )
        final_ppose = io.to_packed(pose)
        yield final_ppose

