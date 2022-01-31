# Python standard library
from typing import Iterator, List, Optional, Tuple, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector
from pyrosetta.rosetta.protocols.filters import Filter # TODO
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.scoring import ScoreFunction
from pyrosetta.rosetta.core.kinematics import MoveMap

# Custom library imports


def almost_linkres(
    pose: Pose,
    movemap: MoveMap,
    residue_selectors: Union[List[ResidueSelector], Tuple[ResidueSelector]],
    scorefxn: ScoreFunction,
    task_factory: TaskFactory,
    repeats: int = 1,
) -> None:
    """
    This function does fast design using a linkres-style approach.
    It requires at minimum a pose, movemap, scorefxn, and task_factory.
    The pose will be modified in place with fast_design, and the movemap and scorefxn 
    will be passed directly to fast_design. The task_factory will have a sequence 
    symmetry taskop added before it will be passed to fast_design. The residue_selectors 
    will be used to determine which residues to pseudosymmetrize, and need to specify
    equal numbers of residues for each selector.
    """
    import sys
    from pathlib import Path
    import pyrosetta

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.design import fast_design

    # make a dict of the residue selectors as index strings
    index_selectors = {}
    for i, selector in enumerate(residue_selectors):
        index_selectors[f"sel_{i}"] = ",".join(
            [str(j) for j, pos in list(enumerate(selector.apply(pose), start=1)) if pos]
        )
    pre_xml_string = """
    <RESIDUE_SELECTORS>
        {index_selectors_str}
    </RESIDUE_SELECTORS>
    <TASKOPERATIONS>
        <KeepSequenceSymmetry name="linkres_op" setting="true"/>
    </TASKOPERATIONS>
    <MOVERS>
        <SetupForSequenceSymmetryMover name="almost_linkres" sequence_symmetry_behaviour="linkres_op" >
            <SequenceSymmetry residue_selectors="{selector_keys}" />
        </SetupForSequenceSymmetryMover>
    </MOVERS>
    """
    index_selectors_str = "\n\t\t".join(
        [f"""<Index name="{key}" resnums="{value}" />""" for key, value in index_selectors.items()]
    )
    # autogenerate an xml string
    xml_string = pre_xml_string.format(
        index_selectors_str=index_selectors_str, selector_keys=",".join(index_selectors.keys())
    )
    # setup the xml_object
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        xml_string
    )
    # get the taskop from the xml_object
    linkres_op = objs.get_task_operation("linkres_op")
    # push back the taskop to the task factory
    task_factory.push_back(linkres_op)
    # get the mover from the xml_object
    pre_linkres = objs.get_mover("almost_linkres")
    # apply the mover to the pose
    pre_linkres.apply(pose)
    # run fast_design with the pose, movemap, scorefxn, task_factory
    fast_design(
        pose=pose,
        movemap=movemap,
        scorefxn=scorefxn,
        task_factory=task_factory,
        repeats=repeats,
    )
    return


@requires_init
def two_state_design_paired_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    TODO
    needs beta_nov16 at least
    """

    from pathlib import Path
    from time import time
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        add_metadata_to_pose,
        fast_design,
        interface_among_chains,
        gen_movemap,
        gen_score_filter,
        gen_std_layer_design,
        gen_task_factory,
        score_per_res,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()
    # hardcode precompute_ig
    pyrosetta.rosetta.basic.options.set_boolean_option("packing:precompute_ig", True)
    # setup scorefxns
    clean_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0  # TODO
    )
    print_timestamp("Generated score functions", start_time=start_time)
    # setup movemaps
    fixbb_mm = gen_movemap(jump=True, chi=True, bb=False)
    flexbb_mm = gen_movemap(jump=True, chi=True, bb=True)
    print_timestamp("Generated movemaps", start_time=start_time)
    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )
    # gogogo
    for pose in poses:
        start_time = time()
        # get the scores from the pose
        scores = dict(pose.scores)
        # for the neighborhood residue selector
        pose.update_residue_neighbors()
        # get the chains
        chA, chB, chC = (ChainSelector(i) for i in range(1, 4))
        # get the bound interface
        interface_sel = interface_among_chains(chain_list=[1, 2], vector_mode=True)
        # get the chB interface
        chB_interface_sel = AndResidueSelector(chB, interface_sel)
        offset = pose.chain_end(2)
        # get any residues that differ between chA and chC - starts as a list of tuples
        difference_indices = [
            (i, i + offset)
            for i in range(1, pose.chain_end(1) + 1)
            if pose.residue(i).name() != pose.residue(i + offset).name()
        ]
        # flatten the list of tuples into a sorted list of indices
        difference_indices = sorted(sum(difference_indices, ()))
        # make a residue selector for the difference indices
        difference_sel = ResidueIndexSelector(",".join(map(str, difference_indices)))
        # use OrResidueSelector to combine the two
        design_sel = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(
            chB_interface_sel, difference_sel
        )
        # we want to design the peptide (chB) interface + anything that differs between the states
        # and maybe the neighborhood of that as well ? TODO
        print_timestamp("Generated selectors", start_time=start_time)
        layer_design = gen_std_layer_design()
        task_factory_1 = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=False,
            extra_rotamers_level=2,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=True,
            restrict_pro_gly=True,
            ifcl=True,  # so that it respects precompute_ig if it is passed as an flag
            layer_design=layer_design,
        )
        print_timestamp(
            "Generated interface design task factory", start_time=start_time
        )
        # setup the linked selectors
        residue_selectors = chA, chC
        print_timestamp(
            "Starting 1 round of fixbb msd with neighborhood",
            start_time=start_time,
        )
        almost_linkres(
            pose=pose,
            movemap=fixbb_mm,
            residue_selectors=residue_selectors,
            scorefxn=design_sfxn,
            task_factory=task_factory_1,
            repeats=1,
        )
        print_timestamp(
            "Starting 1 round of flexbb design without neighborhood",
            end="",
            start_time=start_time,
        )
        task_factory_2 = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=False,
            extra_rotamers_level=2,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=True,
            restrict_pro_gly=True,
            ifcl=True,  # so that it respects precompute_ig if it is passed as an flag
            layer_design=layer_design,
        )
        almost_linkres(
            pose=pose,
            movemap=flexbb_mm,
            residue_selectors=residue_selectors,
            scorefxn=design_sfxn,
            task_factory=task_factory_2,
            repeats=1,
        )
        print_timestamp("Scoring...", start_time=start_time)
        score_per_res(pose, clean_sfxn)
        score_filter = gen_score_filter(clean_sfxn)
        add_metadata_to_pose(pose, "path_in", pdb_path)
        end_time = time()
        total_time = end_time - start_time
        print_timestamp(f"Total time: {total_time:.2f} seconds", start_time=start_time)
        add_metadata_to_pose(pose, "time", total_time)
        scores.update(pose.scores)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose
