# Python standard library
from typing import *

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.distributed import requires_init

def loop_match(pose: Pose, length: int):
    """
    Runs ConnectChainsMover. Expects a pose with two chains, A and B.
    """
    import pyrosetta

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <ConnectChainsMover name="connectchains" 
                chain_connections="[A+B]" 
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
        closure_type = 'loop_match'
    except RuntimeError as e: # if ConnectChainsMover cannot find a closure
        print(e)
        closure_type = 'not_closed'
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, "closure_type", closure_type
    )
    return closure_type

def phi_psi_omega_to_abego(phi: float, psi: float, omega: float) -> str:
    """
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

def abego_string(phi_psi_omega: list) -> str:
    """
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    out = ""
    for x in phi_psi_omega:
        out += phi_psi_omega_to_abego(x[0], x[1], x[2])
    return out

def get_torsions(pose: Pose) -> list:
    """
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

def loop_remodel(
    pose: Pose, length: int,
    attempts: int = 10,
    remodel_before_loop: int = 1, 
    remodel_after_loop: int = 1,
    remodel_lengths_by_vector: bool = False
    ):
    """
    Remodel a new loop using Blueprint Builder. Expects a pose with two chains.
    DSSP and SS agnostic in principle but in practice more or less matches.
    """
    import numpy as np
    import os
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    def remodel_helper(pose: Pose, loop_length: int, remodel_before_loop: int = 1, remodel_after_loop: int = 1) -> str:
        import binascii, os
        import pyrosetta
        
        tors = get_torsions(pose)
        abego_str = abego_string(tors)
        dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(pose)
        # name blueprint a random 32 long hex string
        if 'TMPDIR' in os.environ:
            tmp_path = os.environ['TMPDIR']
        else:
            tmp_path = os.getcwd()
        filename = os.path.join(
            tmp_path,
            str(binascii.b2a_hex(os.urandom(16)).decode("utf-8")) + ".bp"
        )
        # write a temporary blueprint file
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path, exist_ok=True)
        with open(filename, "w+") as f:
            end1, begin2 = (
                pose.chain_end(1),
                pose.chain_begin(2),
            )
            end2 = pose.chain_end(2)
            for i in range(1, end1 + 1):
                if i >= end1 - (remodel_before_loop - 1):
                    print(
                        str(i),
                        pose.residue(i).name1(),
                        dssp[i - 1] + "X",
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
            for i in range(loop_length):
                print(
                    "0", "V", "LX", "R", file=f
                )  # DX is bad, causes rare error sometimes
            for i in range(begin2, end2 + 1):
                if i <= begin2 + (remodel_after_loop - 1):
                    print(
                        str(i),
                        pose.residue(i).name1(),
                        dssp[i - 1] + "X",
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

    # computes the number of residues to remodel before and after the loop by finding which residue-residue vectors point towards the helix to loop to
    # probably works best for building a loop between two helices
    # still uses the default lengths to remodel if none of the vectors are good (dot>0)
    if remodel_lengths_by_vector:
        end1, begin2 = (pose.chain_end(1), pose.chain_begin(2))
        max_dot_1 = 0
        max_dot_2 = 0
        vec_12 = pose.residue(begin2).xyz("CA") - pose.residue(end1).xyz("CA")
        for i in range(3):
            vec_1 = pose.residue(end1-i).xyz("CA") - pose.residue(end1-i-1).xyz("CA")
            dot_1 = vec_12.dot(vec_1.normalize()) # normalization accounts for slight differences in Ca-Ca distances dependent on secondary structure
            if dot_1 > max_dot_1:
                max_dot_1 = dot_1
                remodel_before_loop = i + 1
            vec_2 = pose.residue(begin2+i+1).xyz("CA") - pose.residue(begin2+i).xyz("CA")
            dot_2 = vec_12.dot(vec_2.normalize())
            if dot_2 > max_dot_2:
                max_dot_2 = dot_2
                remodel_after_loop = i + 1

    bp_file = remodel_helper(pose, length, remodel_before_loop, remodel_after_loop)

    bp_sfxn = pyrosetta.create_score_function("fldsgn_cen.wts")
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0)

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

    closure_type = 'not_closed'
    for _ in range(attempts):
        bp_mover.apply(pose)
        if pose.num_chains() == 1:
            closure_type = 'loop_remodel'
            break

    os.remove(bp_file)
    
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "remodel_before_loop", str(remodel_before_loop))
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "remodel_after_loop", str(remodel_after_loop))
    return closure_type


@requires_init
def loop_complex(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Generator[PackedPose, PackedPose, None]:

    from time import time
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io

    sys.path.insert(0, "/mnt/home/broerman/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.utils.io import print_timestamp
    from crispy_shifty.protocols.design import gen_std_layer_design, gen_task_factory, packrotamers, struct_profile, clear_constraints, score_wnm, score_ss_sc

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
    all_chains_to_loop = [[int(chain) for chain in chains.split(',')] for chains in kwargs["chains_to_loop"].split(";")]
    all_loop_lengths = [[int(length) for length in lengths.split(',')] for lengths in kwargs["loop_lengths"].split(";")]

    for pose in poses:
        scores = dict(pose.scores)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)
        # get parent length from the score
        # parent_length = int(float(scores["parent_length"]))

        # For convenience, the all_chains_to_loop function argument only includes the chains to loop.
        # Add in all the other chains that aren't being looped.
        chain_loop_configuration = [([i], None) for i in range(1,all_chains_to_loop[0][0])]
        for i in range(len(all_chains_to_loop)-1):
            chain_loop_configuration.append((all_chains_to_loop[i], all_loop_lengths[i]))
            for j in range(all_chains_to_loop[i][-1]+1, all_chains_to_loop[i+1][0]):
                chain_loop_configuration.append(([j], None))
        chain_loop_configuration.append((all_chains_to_loop[-1], all_loop_lengths[-1]))
        for j in range(all_chains_to_loop[-1][-1]+1, pose.num_chains()+1):
            chain_loop_configuration.append(([j], None))

        # careful- all_chains is still 1-indexed, so the chains_to_loop kwarg is 1-indexed also
        all_chains = pose.split_by_chain()

        looped_poses = []
        closure_type = 'not_closed'
        for chains_to_loop, loop_lengths in chain_loop_configuration:
            looped_pose = all_chains[chains_to_loop[0]]

            if len(chains_to_loop) >= 2:
                new_loop_strs = []
                for unlooped_chain, loop_length in zip(chains_to_loop[1:], loop_lengths):
                    loop_start = int(looped_pose.size()) + 1
                    pyrosetta.rosetta.core.pose.append_pose_to_pose(looped_pose, all_chains[unlooped_chain], True)
                    # Rebuild PDBInfo for ConnectChainsMover
                    pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(looped_pose)
                    looped_pose.pdb_info(pdb_info)

                    print_timestamp("Attempting closure by loop match...", start_time, end="")
                    closure_type = loop_match(looped_pose, loop_length)
                    # closure by loop matching was successful, move on to the next set to close or continue to scoring
                    # should I use an additional check like 'pose_to_loop.num_chains() == 1' to determine if the pose is closed?
                    if closure_type != 'not_closed':
                        print('success.')
                    else:
                        print('failed.')

                        print_timestamp("Attempting closure by loop remodel...", start_time, end="")
                        closure_type = loop_remodel(looped_pose, loop_length, 10, 1, 1, True)
                        if closure_type != 'not_closed':
                            print('success.')
                        else:
                            print('failed. Exiting.')
                            # couldn't close this pair; stop trying with the whole set
                            break

                    # is this naive? Phil did something more complicated with residue selectors, looking at the valines.
                    # Wondering if I'm missing some edge cases for which this approach doesn't work.
                    new_loop_strs.append(",".join(str(resi) for resi in range(loop_start, loop_start + loop_length)))

                if closure_type == 'not_closed':
                    # couldn't close this set; stop trying with the all the sets
                    break

                new_loop_str = ",".join(new_loop_strs)
                pyrosetta.rosetta.core.pose.setPoseExtraScore(
                    looped_pose, "new_loop_resis", new_loop_str
                )

            looped_poses.append(looped_pose)

        # if we couldn't close one of the sets in this complex, continue to the next pose and skip scoring and yielding the pose (so nothing is written to disk)
        if closure_type == 'not_closed':
            continue

        # The code will only reach here if all loops are closed.
        # Loop closure is fast but has a somewhat high failure rate, so more efficient to first see if all loops can be closed, 
        # and only design and score if so.
        # First combine all the looped poses into one pose, then design the new loop residues. This ensures the new loop
        # residues are designed in the context of the rest of the pose.
        # looped_poses contains single chains of the unlooped chains, and the looped chains.

        # combine all the looped poses into one pose
        print_timestamp("Collecting poses...", start_time, end="")
        combined_looped_pose = pyrosetta.rosetta.core.pose.Pose()
        new_loop_strs = []
        loop_scores = {}
        for i, looped_pose in enumerate(looped_poses):
            chain_id = str(i + 1)
            for key, value in looped_pose.scores.items():
                loop_scores[key + '_' + chain_id] = value
            pose_end = combined_looped_pose.size()
            if 'new_loop_resis' in looped_pose.scores:
                new_loop_resis = [int(i) + pose_end for i in looped_pose.scores['new_loop_resis'].split(',')]
                new_loop_strs.append(','.join([str(i) for i in new_loop_resis]))
            pyrosetta.rosetta.core.pose.append_pose_to_pose(combined_looped_pose, looped_pose, True)
        new_loop_str = ','.join(new_loop_strs)
        # Rebuild PDBInfo
        pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(combined_looped_pose)
        combined_looped_pose.pdb_info(pdb_info)
        # Add scores from looping
        for key, value in loop_scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key, value)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, 'new_loop_resis', new_loop_str)
        print('complete.')
        print(new_loop_str)
        
        print_timestamp("Designing loops...", start_time, end="")
        layer_design = gen_std_layer_design()
        design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
        design_sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
        )

        new_loop_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(new_loop_str)
        design_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(new_loop_sel, 6, True)
        task_factory = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=2,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=False,
            restrict_pro_gly=False,
            precompute_ig=True,
            ifcl=True,
            layer_design=layer_design,
        )
        struct_profile(combined_looped_pose, design_sel) # Phil's code used eliminate_background=False...
        packrotamers(combined_looped_pose, task_factory, design_sfxn)
        clear_constraints(combined_looped_pose)
        print('complete.')
        
        # Score the packed loops
        print_timestamp("Scoring...", start_time, end="")
        for i, looped_pose in enumerate(combined_looped_pose.split_by_chain()):
            chain_id = str(i + 1)
            total_length = len(looped_pose.residues)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, "total_length_" + chain_id, total_length)
            dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(looped_pose)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, "dssp_" + chain_id, dssp)
            tors = get_torsions(looped_pose)
            abego_str = abego_string(tors)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, "abego_str_" + chain_id, abego_str)

            # should be fast since the database is already loaded from CCM/SPM
            wnm = score_wnm(looped_pose)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, "wnm_" + chain_id, wnm)
            ss_sc = score_ss_sc(looped_pose, False, True, 'loop_sc')
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, "ss_sc_" + chain_id, ss_sc)
        print('complete.')

        # Add any old scores back into the pose
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key, value)

        ppose = io.to_packed(combined_looped_pose)
        yield ppose


# Old, pre-generalization to arbitrary chain configurations
# @requires_init
# def loop_dimer(
#     packed_pose_in: Optional[PackedPose] = None, **kwargs
# ) -> Generator[PackedPose, PackedPose, None]:

#     from copy import deepcopy
#     from time import time
#     import sys
#     import pyrosetta
#     import pyrosetta.distributed.io as io

#     sys.path.insert(0, "/mnt/home/broerman/projects/crispy_shifty")
#     from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
#     from crispy_shifty.utils.io import print_timestamp
#     from crispy_shifty.protocols.design import gen_std_layer_design, gen_task_factory, packrotamers, struct_profile, clear_constraints, score_wnm, score_ss_sc

#     # testing to properly set the TMPDIR on distributed jobs
#     # import os
#     # os.environ['TMPDIR'] = '/scratch'
#     # print(os.environ['TMPDIR'])

#     start_time = time()

#     # generate poses or convert input packed pose into pose
#     if packed_pose_in is not None:
#         poses = [io.to_pose(packed_pose_in)]
#         pdb_path = "none"
#     else:
#         pdb_path = kwargs["pdb_path"]
#         poses = path_to_pose_or_ppose(
#             path=pdb_path, cluster_scores=True, pack_result=False
#         )

#     for pose in poses:
#         scores = dict(pose.scores)
#         pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)
#         # get parent length from the score
#         parent_length = int(float(scores["parent_length"]))

#         looped_poses = []
#         sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
#         for chains_to_loop in ['12', '34']:
#             sw.chain_order(chains_to_loop)
#             looped_pose = deepcopy(pose)
#             sw.apply(looped_pose)

#             loop_length = int(parent_length - looped_pose.chain_end(2))

#             # is this naive? Phil did something more complicated with residue selectors, looking at the valines.
#             # Wondering if I'm missing some edge cases for which this approach doesn't work.
#             loop_start = int(looped_pose.chain_end(1)) + 1
#             new_loop_str = ",".join(str(resi) for resi in range(loop_start, loop_start + loop_length))
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(
#                 looped_pose, "new_loop_resis", new_loop_str
#             )

#             print_timestamp("Attempting closure by loop match...", start_time, end="")
#             closure_type = loop_match(looped_pose, loop_length)
#             # closure by loop matching was successful, move on to the next dimer or continue to scoring
#             # should I use a check like 'pose_to_loop.num_chains() == 1' to determine if the pose is closed?
#             if closure_type != 'not_closed':
#                 print('success.')
#             else:
#                 print('failed.')

#                 print_timestamp("Attempting closure by loop remodel...", start_time, end="")
#                 closure_type = loop_remodel(looped_pose, loop_length, 10, 1, 1, True)
#                 if closure_type != 'not_closed':
#                     print('success.')
#                 else:
#                     print('failed. Exiting.')
#                     # couldn't close this monomer; stop trying with the whole dimer
#                     break

#             looped_poses.append(looped_pose)

#         # if we couldn't close the dimer, continue to the next pose and skip scoring, labeling, and yielding the pose (so nothing is written to disk)
#         if closure_type == 'not_closed':
#             continue

#         # The code will only reach here if both loops are closed.
#         # Loop closure is fast but has a high failure rate, so more efficient to first see if all loops can be closed, 
#         # and only design and score if so.
         
#         layer_design = gen_std_layer_design()
#         design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
#         design_sfxn.set_weight(
#             pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
#         )

#         for looped_pose in looped_poses:

#             print_timestamp("Designing loop...", start_time, end="")
#             new_loop_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(new_loop_str)
#             design_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(new_loop_sel, 6, True)
#             task_factory = gen_task_factory(
#                 design_sel=design_sel,
#                 pack_nbhd=True,
#                 extra_rotamers_level=2,
#                 limit_arochi=True,
#                 prune_buns=True,
#                 upweight_ppi=False,
#                 restrict_pro_gly=False,
#                 ifcl=True, # to respect precompute_ig
#                 layer_design=layer_design,
#             )
#             struct_profile(looped_pose, design_sel) # Phil's code used eliminate_background=False...
#             packrotamers(looped_pose, task_factory, design_sfxn)
#             clear_constraints(looped_pose)
#             print('complete.')
            
#             print_timestamp("Scoring...", start_time, end="")
#             total_length = len(looped_pose.residues)
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "total_length", total_length)
#             dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(looped_pose)
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "dssp", dssp)
#             tors = get_torsions(looped_pose)
#             abego_str = abego_string(tors)
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "abego_str", abego_str)

#             # score_wnm is fine since it's only one chain
#             # should also be fast since the database is already loaded from CCM
#             score_wnm(looped_pose)
#             score_ss_sc(looped_pose, False, True, 'loop_sc')
#             print('complete.')

#         combined_looped_pose = deepcopy(looped_poses[0])
#         pyrosetta.rosetta.core.pose.append_pose_to_pose(combined_looped_pose, looped_poses[1], True)
#         sw.chain_order('12')
#         sw.apply(combined_looped_pose)
#         pyrosetta.rosetta.core.pose.clearPoseExtraScores(combined_looped_pose)

#         for key, value in scores.items():
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key, value)
#         for protomer, looped_pose in zip(['A', 'B'], looped_poses):
#             for key, value in looped_pose.scores.items():
#                 pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key + '_' + protomer, value)

#         ppose = io.to_packed(combined_looped_pose)
#         yield ppose


# Old, pre-design-after-looping
# @requires_init
# def loop_dimer(
#     packed_pose_in: Optional[PackedPose] = None, **kwargs
# ) -> Generator[PackedPose, PackedPose, None]:

#     from copy import deepcopy
#     from time import time
#     import sys
#     import pyrosetta
#     import pyrosetta.distributed.io as io

#     sys.path.insert(0, "/mnt/home/broerman/projects/crispy_shifty")
#     from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
#     from crispy_shifty.utils.io import print_timestamp
#     from crispy_shifty.protocols.design import score_wnm, score_ss_sc

#     # testing to properly set the TMPDIR on distributed jobs
#     # import os
#     # os.environ['TMPDIR'] = '/scratch'
#     # print(os.environ['TMPDIR'])

#     start_time = time()

#     # generate poses or convert input packed pose into pose
#     if packed_pose_in is not None:
#         poses = [io.to_pose(packed_pose_in)]
#         pdb_path = "none"
#     else:
#         pdb_path = kwargs["pdb_path"]
#         poses = path_to_pose_or_ppose(
#             path=pdb_path, cluster_scores=True, pack_result=False
#         )

#     for pose in poses:
#         scores = dict(pose.scores)
#         pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)
#         # get parent length from the score
#         parent_length = int(float(scores["parent_length"]))

#         looped_poses = []
#         sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
#         for chains_to_loop in ['12', '34']:
#             sw.chain_order(chains_to_loop)
#             looped_pose = deepcopy(pose)
#             sw.apply(looped_pose)

#             loop_length = int(parent_length - looped_pose.chain_end(2))

#             # is this naive? Phil did something more complicated with residue selectors, looking at the valines.
#             # Wondering if I'm missing some edge cases for which this approach doesn't work.
#             loop_start = int(looped_pose.chain_end(1)) + 1
#             new_loop_str = ",".join(str(resi) for resi in range(loop_start, loop_start + loop_length))
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(
#                 looped_pose, "new_loop_resis", new_loop_str
#             )

#             print_timestamp("Attempting closure by loop match...", start_time, end="")
#             closure_type = loop_match(looped_pose, loop_length)
#             # closure by loop matching was successful, move on to the next dimer or continue to scoring
#             # should I use a check like 'pose_to_loop.num_chains() == 1' to determine if the pose is closed?
#             if closure_type != 'not_closed':
#                 print('success.')
#                 # continue
#             else:
#                 print('failed.')

#                 print_timestamp("Attempting closure by loop remodel...", start_time, end="")
#                 closure_type = loop_remodel(looped_pose, loop_length, 10, 1, 1, True)
#                 if closure_type != 'not_closed':
#                     print('success.')
#                 else:
#                     print('failed. Exiting.')
#                     # couldn't close this monomer; stop trying with the whole dimer
#                     break

#             # The code will only reach here if the loop is closed, and then I can score individual loops by accessing the looped_pose object
#             print_timestamp("Designing loop...", start_time, end="")
#             new_loop_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(new_loop_str)
#             designable_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(new_loop_sel, 6, True)
            
            
#             print_timestamp("Scoring...", start_time, end="")
#             total_length = len(looped_pose.residues)
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "total_length", total_length)
#             dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(looped_pose)
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "dssp", dssp)
#             tors = get_torsions(looped_pose)
#             abego_str = abego_string(tors)
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, "abego_str", abego_str)

#             # score_wnm is fine since it's only one chain
#             # should also be fast since the database is already loaded from CCM
#             score_wnm(looped_pose)
#             score_ss_sc(looped_pose, False, True, 'loop_sc')
#             print('complete.')

#             looped_poses.append(looped_pose)

#         # if we couldn't close the dimer, continue to the next pose and skip scoring, labeling, and yielding the pose (so nothing is written to disk)
#         if closure_type == 'not_closed':
#             continue

#         combined_looped_pose = deepcopy(looped_poses[0])
#         pyrosetta.rosetta.core.pose.append_pose_to_pose(combined_looped_pose, looped_poses[1], True)
#         sw.chain_order('12')
#         sw.apply(combined_looped_pose)
#         pyrosetta.rosetta.core.pose.clearPoseExtraScores(combined_looped_pose)

#         for key, value in scores.items():
#             pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key, value)
#         for protomer, looped_pose in zip(['A', 'B'], looped_poses):
#             for key, value in looped_pose.scores.items():
#                 pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key + '_' + protomer, value)

#         ppose = io.to_packed(combined_looped_pose)
#         yield ppose