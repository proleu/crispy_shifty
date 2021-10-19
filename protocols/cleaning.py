# Python standard library
from typing import *
# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
# Custom library imports

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
        # setup rechain mover to sanitize the trimmed pose
        rechain = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        rechain.chain_order("1")
        # get secondary structure
        pyrosetta.rosetta.core.scoring.dssp.Dssp(pose).insert_ss_into_pose(pose, True)
        dssp = pose.secstruct()
        # get leading loop from ss
        trimmed_pose = pose.clone()
        if dssp[0] == "H":  # in case no leading loop is detected
            pass
        else:  # get beginning index of first occurrence of LH in dssp
            rosetta_idx_n_term  = str(dssp.find("LH")+1)
            # setup trimming mover
            trimmer = pyrosetta.rosetta.protocols.grafting.simple_movers.KeepRegionMover()
            trimmer.start(rosetta_idx_n_term)
            trimmer.end(str(pose.chain_end(1)))
            trimmer.apply(trimmed_pose)
            rechain.apply(trimmed_pose)
        # get trailing loop from ss
        if dssp[-1] == "H":  # in case no trailing loop is detected
            pass
        else:  # get ending index of last occurrence of HL in dssp
            rosetta_idx_c_term = str(dssp.rfind("HL")+2)
            # setup trimming mover
            trimmer = pyrosetta.rosetta.protocols.grafting.simple_movers.KeepRegionMover()
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
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(value))
        final_pposes.append(io.to_packed(pose))
    return final_pposes

def redesign_disulfides(packed_pose_in=None, **kwargs):
    """
    fixbb fastdesign with beta_nov16 on all cys residues using layerdesign.
    Requires the following init flags:
    -corrections::beta_nov16 true
    -detect_disulf false
    -holes:dalphaball /home/bcov/ppi/tutorial_build/main/source/external/DAlpahBall/DAlphaBall.gcc
    -indexed_structure_store:fragment_store /home/bcov/sc/scaffold_comparison/data/ss_grouped_vall_all.h5
    """

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )
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
        <SSShapeComplementarity name="sc_hlx_parent" verbose="1" loops="0" helices="1" />
        <SSShapeComplementarity name="sc_all_parent" verbose="1" loops="1" helices="1" />
        <ScoreType name="total_score_pose" scorefxn="sfxn" score_type="total_score" threshold="0" confidence="0" />
        <ResidueCount name="count" />
        <CalculatorFilter name="score_per_res_parent" equation="total_score_full / res" threshold="-2.0" confidence="0">
            <Var name="total_score_full" filter="total_score_pose"/>
            <Var name="res" filter="count"/>
        </CalculatorFilter>
        <worst9mer name="wnm_hlx_parent" rmsd_lookup_threshold="0.4" confidence="0" only_helices="true" />
        <worst9mer name="wnm_all_parent" rmsd_lookup_threshold="0.4" confidence="0" />
        <worst9mer name="9mer_parent" rmsd_lookup_threshold="0.4" confidence="0" />
    </FILTERS>
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
        <Add filter_name="sc_hlx_parent" />
        <Add filter_name="sc_all_parent" />
        <Add filter_name="score_per_res_parent" />
        <Add filter_name="wnm_hlx_parent"/>
        <Add filter_name="wnm_all_parent"/>
        <Add filter_name="9mer_parent"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    design_score = SingleoutputRosettaScriptsTask(xml)

    final_pposes = []
    for pose in poses:
        scores = dict(pose.scores)
        designed_ppose = design_score(pose.clone())
        pose = io.to_pose(designed_ppose)
        scores.update(dict(pose.scores))
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                pose, key, str(value)
            )  # store values as strings for safety
        designed_ppose = io.to_packed(pose)
        final_pposes.append(designed_ppose)
    return final_pposes
