# Python standard library
from typing import *

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector
from pyrosetta.rosetta.protocols.filters import Filter
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.scoring import ScoreFunction

# Custom library imports

def interface_between_selectors(sel_1:ResidueSelector, sel_2:ResidueSelector, vector_mode:bool=False):
    """
    Returns a selector that selects the interface between two selectors.
    """
    if vector_mode:
        from pyrosetta.rosetta.core.select.residue_selector import InterGroupInterfaceByVectorSelector
        int_sel = InterGroupInterfaceByVectorSelector(sel_1, sel_2)
        # tuned for TJ DHRs
        int_sel.nearby_atom_cut(4)
        int_sel.vector_dist_cut(6.5)
        int_sel.cb_dist_cut(8.5)
        return int_sel
    else:
        from pyrosetta.rosetta.core.select.residue_selector import (
            NeighborhoodResidueSelector,
            AndResidueSelector
        )
        sel_1_nbhd = NeighborhoodResidueSelector(sel_1, 8, True)
        sel_2_nbhd = NeighborhoodResidueSelector(sel_2, 8, True)
        return AndResidueSelector(sel_1_nbhd, sel_2_nbhd)

def interface_among_chains(chain_list:list, vector_mode:bool=False):
    """
    Returns a selector that selects the interface between the given chains of a pose.
    """
    from itertools import combinations
    from pyrosetta.rosetta.core.select.residue_selector import (
        OrResidueSelector,
        ChainSelector
    )
    int_sel = OrResidueSelector()
    for chain_1, chain_2 in combinations(chain_list, 2):
        sel_1 = ChainSelector(chain_1)
        sel_2 = ChainSelector(chain_2)
        pair_int_sel = interface_between_selectors(sel_1, sel_2, vector_mode)
        int_sel.add_residue_selector(pair_int_sel)

    return int_sel

def gen_task_factory(design_sel:ResidueSelector,
                     pack_nbhd:bool=False,
                     extra_rotamers_level:int=0,
                     limit_arochi:bool=False,
                     prune_buns:bool=False,
                     upweight_ppi:bool=False,
                     restrict_pro_gly:bool=False,
                     ifcl:bool=False,
                     layer_design:list=None):
    import pyrosetta
    from itertools import product
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        LayerSelector,
        NotResidueSelector,
        PrimarySequenceNeighborhoodSelector,
        SecondaryStructureSelector
    )
    from pyrosetta.rosetta.core.pack.task.operation import (
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictAbsentCanonicalAASExceptNativeRLT,
        RestrictToRepackingRLT
    )

    task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()

    if pack_nbhd:
        pack_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(design_sel, 6, True) # pack around designable area
        pack_op = OperateOnResidueSubset(RestrictToRepackingRLT(), design_sel, True) # everything not designable
        task_factory.push_back(pack_op)

        lock_op = OperateOnResidueSubset(PreventRepackingRLT(), pack_sel, True) # everything not packable
        task_factory.push_back(lock_op)
    else:
        lock_op = OperateOnResidueSubset(PreventRepackingRLT(), design_sel, True)
        task_factory.push_back(lock_op)
    
    if extra_rotamers_level > 0:
        extra_rotamers_op = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
        extra_rotamers_op.ex1(True)
        extra_rotamers_op.ex2(False)
        extra_rotamers_op.ex3(False)
        extra_rotamers_op.ex4(False)
        if extra_rotamers_level > 1:
            extra_rotamers_op.ex2(True)
            if extra_rotamers_level > 2:
                extra_rotamers_op.ex3(True)
                if extra_rotamers_level > 3:
                    extra_rotamers_op.ex4(True)
        task_factory.push_back(extra_rotamers_op)

    if prune_buns:
        prune_buns_op = pyrosetta.rosetta.protocols.task_operations.PruneBuriedUnsatsOperation()
        prune_buns_op.allow_even_trades(False)
        prune_buns_op.atomic_depth_cutoff(3.5)
        prune_buns_op.minimum_hbond_energy(-1.0)
        task_factory.push_back(prune_buns_op)
    
    # add standard task operations
    if limit_arochi:
        arochi_op = pyrosetta.rosetta.protocols.task_operations.LimitAromaChi2Operation()
        arochi_op.chi2max(110)
        arochi_op.chi2min(70)
        arochi_op.include_trp(True)
        task_factory.push_back(arochi_op)

    if upweight_ppi:
        upweight_ppi_op = pyrosetta.rosetta.protocols.pack_interface.ProteinProteinInterfaceUpweighter()
        upweight_ppi_op.set_weight(3.0)
        task_factory.push_back(upweight_ppi_op)

    if restrict_pro_gly:
        pro_gly_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueNameSelector('PRO,GLY')
        pro_gly_op = OperateOnResidueSubset(PreventRepackingRLT(), pro_gly_sel, False)
        task_factory.push_back(pro_gly_op)

    if ifcl:
        ifcl_op = pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline()
        task_factory.push_back(ifcl_op)

    if layer_design is None: # set default layer design
        layer_design = ['DNSTP', # helix_cap
                        'AFILVWYNQSTHP', # core AND helix_start
                        'AFILVWM', # core AND helix
                        'FILVWY', # core AND sheet
                        'AFGILPVWYSM', # core AND loop
                        'ADEHIKLNPQRSTVWY', # boundary AND helix_start
                        'ADEHIKLNQRSTVWYM', # boundary AND helix
                        'DEFHIKLNQRSTVWY', # boundary AND sheet
                        'ADEFGHIKLNPQRSTVWY', # boundary AND loop
                        'DEHKPQR', # surface AND helix_start
                        'EHKQR', # surface AND helix
                        'EHKNQRST', # surface AND sheet
                        'DEGHKNPQRST'] # surface AND loop
    assert(len(layer_design) == 13)

    layer_sels = [] # core, boundary, surface
    for layer in ['core', 'bdry', 'surf']:
        layer_sel = LayerSelector()
        # layer_sel.set_layers(i==0, i==1, i==2) # 1-liner when iterating through a range, but less easy to read
        if layer == 'core':
            layer_sel.set_layers(True, False, False)
        elif layer == 'bdry':
            layer_sel.set_layers(False, True, False)
        elif layer == 'surf':
            layer_sel.set_layers(False, False, True)
        layer_sel.set_use_sc_neighbors(True)
        layer_sels.append(layer_sel)

    ss_sels = [] # alpha, beta, coil
    for ss in ['H', 'E', 'L']:
        ss_sel = SecondaryStructureSelector()
        ss_sel.set_selected_ss(ss)
        ss_sel.set_overlap(0)
        ss_sel.set_minH(3)
        ss_sel.set_minE(2)
        ss_sel.set_use_dssp(True)
        if ss == 'L':
            ss_sel.set_include_terminal_loops(True)
        ss_sels.append(ss_sel)

    helix_cap_sel = AndResidueSelector(ss_sels[2], PrimarySequenceNeighborhoodSelector(1, 0, ss_sels[0], False))
    helix_start_sel = AndResidueSelector(ss_sels[0], PrimarySequenceNeighborhoodSelector(0, 1, helix_cap_sel, False))
    final_ss_sels = [helix_start_sel,
                     AndResidueSelector(ss_sels[0], NotResidueSelector(helix_start_sel)),
                     ss_sels[1],
                     AndResidueSelector(ss_sels[2], NotResidueSelector(helix_cap_sel))]

    region_sels = [helix_cap_sel]
    for layer_sel, ss_sel in product(layer_sels, final_ss_sels):
        region_sels.append(AndResidueSelector(layer_sel, ss_sel))

    for layer_aas, region_sel in zip(layer_design, region_sels):
        task_op = RestrictAbsentCanonicalAASExceptNativeRLT()
        task_op.aas_to_keep(layer_aas)
        region_op = OperateOnResidueSubset(task_op, region_sel, False)
        task_factory.push_back(region_op)

        # testing - prints the pymol selections corresponding to the amino acids set for the layer
        # from crispy_shifty.utils.io import pymol_selection
        # from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
        # import pyrosetta.distributed.io as io
        # pdb_path = '/home/broerman/projects/crispy_shifty/projects/crispy_shifty_dimers/01_make_states/decoys/0000/CSD_01_make_states_4c315eabb34a449a886b9e9bd9b8227a.pdb.bz2'
        # for pose in path_to_pose_or_ppose(path=pdb_path, cluster_scores=True, pack_result=False):
        #     print(pymol_selection(pose, region_sel, 'region ' + layer_aas))

    return task_factory

def fastdesign(pose:Pose, task_factory:TaskFactory, scorefxn:ScoreFunction, flexbb:bool=False, repeats:int=1):
    """
    Runs FastDesign with the given task factory and score function.
    Setting flexbb to False prevents backbone movement for all chains
    """
    # took ~37 minutes to run with fixbb and one repeat on test case X23_3_20_3_ct7_fc.pdb
    import pyrosetta
    # using an xml to create the fastdesign mover since it's easier to load in a relax script and to specify the minimization algorithm
    # chose lbfgs_armigo_nonmonotone for the minimization algorithm based on https://new.rosettacommons.org/docs/wiki/rosetta_basics/structural_concepts/minimization-overview
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <FastDesign name="fastdesign" repeats="{repeats}" ramp_down_constraints="false" 
                batch="false" cartesian="false" bondangle="false" bondlength="false" 
                min_type="lbfgs_armijo_nonmonotone" relaxscript="InterfaceDesign2019">
            </FastDesign>
        </MOVERS>
        """
    )
    fdes_mover = objs.get_mover("fastdesign")
    fdes_mover.set_task_factory(task_factory)
    fdes_mover.set_scorefxn(scorefxn)

    fdes_mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    fdes_mm.set_bb(flexbb)
    fdes_mm.set_chi(True)
    fdes_mm.set_jump(True)
    fdes_mover.set_movemap(fdes_mm)

    fdes_mover.apply(pose)

def struct_profile(pose:Pose, design_sel:ResidueSelector):
    import pyrosetta
    # used the defaults from rosettascripts, only changing things changed in the original one-state xml
    # need to specify all these options since there are limited constructors available for structprofilemover
    sp_mover = pyrosetta.rosetta.protocols.simple_moves.StructProfileMover(rmsThreshold=0.6,
                                                                           burialThreshold=3.0,
                                                                           consider_topN_frags=100,
                                                                           burialWt=0.0,
                                                                           only_loops=False,
                                                                           censorByBurial=False,
                                                                           allowed_deviation=0.1,
                                                                           allowed_deviation_loops=0.1,
                                                                           eliminate_background=True,
                                                                           psiblast_style_pssm=False,
                                                                           outputProfile=False,
                                                                           add_csts_to_pose=True,
                                                                           ignore_terminal_res=True,
                                                                           fragment_store_path='',
                                                                           fragment_store_format='hashed',
                                                                           fragment_store_compression='all')
    sp_mover.set_residue_selector(design_sel)
    sp_mover.apply(pose)

    # rmsThreshold: float, 
    # burialThreshold: float, 
    # consider_topN_frags: int, 
    # burialWt: float, 
    # only_loops: bool, 
    # censorByBurial: bool, 
    # allowed_deviation: float, 
    # allowed_deviation_loops: float, 
    # eliminate_background: bool, 
    # psiblast_style_pssm: bool, 
    # outputProfile: bool, 
    # add_csts_to_pose: bool, 
    # ignore_terminal_res: bool, 
    # fragment_store_path: str, 
    # fragment_store_format: str, 
    # fragment_store_compression: str)
    
def clear_constraints(pose:Pose):
    """
    Removes all constraints from the pose.
    """
    import pyrosetta
    cst_mover = pyrosetta.rosetta.protocols.constraint_movers.ClearConstraintsMover()
    cst_mover.apply(pose)

def gen_sasa_filter(pose:Pose, name:str='sasa'):
    pass
    # skip this for now, since it's annoying that it can only compute across jumps
    # import pyrosetta
    # sasa_filter = pyrosetta.rosetta.protocols.simple_filters.InterfaceSasaFilter()

def gen_score_filter(scorefxn:ScoreFunction, name:str='score'):
    import pyrosetta
    score_filter = pyrosetta.rosetta.protocols.score_filters.ScoreTypeFilter(scorefxn, pyrosetta.rosetta.core.scoring.ScoreType.total_score, 0)
    score_filter.set_user_defined_name(name)
    return score_filter
    
def score_on_chain_subset(pose:Pose, filter:Filter, chain_list:list):
    import pyrosetta
    chain_str = ''.join(str(i) for i in chain_list)
    name = filter.get_user_defined_name() + '_' + chain_str
    sw_mover = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    sw_mover.chain_order(chain_str)
    mb_filter = pyrosetta.rosetta.protocols.filters.MoveBeforeFilter(mover=sw_mover, filter=filter)
    mb_filter.set_user_defined_name(name)
    value = mb_filter.report_sm(pose) # maybe could use score instead of report_sm, but couldn't figure out how to set scorename_ of the filter so the values are written to the pdb...
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, value)

def score_cms(pose:Pose, sel_1:ResidueSelector, sel_2:ResidueSelector, name:str='cms'):
    import pyrosetta
    cms_filter = pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter(selector1_=sel_1, selector2_=sel_2)
    cms_filter.set_user_defined_name(name)
    cms = cms_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, cms)

def score_sc(pose: Pose, sel_1:ResidueSelector, sel_2:ResidueSelector, name:str='sc_int'):
    import pyrosetta
    sc_filter = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
    sc_filter.selector1(sel_1)
    sc_filter.selector2(sel_2)
    sc_filter.set_user_defined_name(name)
    sc_filter.write_int_area(True)
    sc_filter.write_median_distance(True)
    sc = sc_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, sc)

def score_ss_sc(pose:Pose, helices:bool=True, loops:bool=True, name:str='ss_sc'):
    import pyrosetta
    ss_sc_filter = pyrosetta.rosetta.protocols.denovo_design.filters.SSShapeComplementarityFilter()
    ss_sc_filter.set_calc_helices(helices)
    ss_sc_filter.set_calc_loops(loops)
    ss_sc_filter.set_user_defined_name(name)
    ss_sc = ss_sc_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, ss_sc)

def score_wnm(pose:Pose, name:str='wnm'):
    # loading the database takes 4.5 minutes and ~5-6 GB of memory, but once loaded, remains for the rest of the python session
    import pyrosetta
    # using an xml to create the worst9mer filter because I couldn't figure out how to use pyrosetta without completely crashing python
    # this is probably because there are many internal variables in the filter which only get set when parsing an xml
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        """
        <FILTERS>
            <Worst9mer name="wnm" rmsd_lookup_threshold="0.4" confidence="0" />
        </FILTERS>
        """
    )
    wnm_filter = objs.get_filter("wnm")
    wnm_filter.set_user_defined_name(name)
    for chain_num in range(1, pose.num_chains() + 1):
        score_on_chain_subset(pose, wnm_filter, [chain_num])

    # import pyrosetta
    # # using an xml to run the worst9mer filter because I couldn't figure out how to use pyrosetta without completely crashing python
    # import pyrosetta.distributed.io as io
    # from pyrosetta.distributed.tasks.rosetta_scripts import SingleoutputRosettaScriptsTask
    # wnm_rs = SingleoutputRosettaScriptsTask(
    #     """
    #     <ROSETTASCRIPTS>
    #         <FILTERS>
    #             <Worst9mer name="wnm" rmsd_lookup_threshold="0.4" confidence="0" />
    #         </FILTERS>
    #         <PROTOCOLS>
    #             <Add filter_name="wnm" />
    #         </PROTOCOLS>
    #     </ROSETTASCRIPTS>
    #     """
    # )
    # for i, chain in enumerate(pose.split_by_chain()):
    #     chain = io.to_pose(wnm_rs(chain))
    #     wnm = chain.scores['wnm']
    #     pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name + '_' + str(i), wnm)

    # import pyrosetta
    # wnm_filter = pyrosetta.rosetta.protocols.simple_filters.Worst9merFilter()
    # for chain_num in range(1, pose.num_chains() + 1):
    #     chain_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(chain_num)
    #     wnm_filter.set_user_defined_name(name + '_' + str(chain_num))
    #     wnm_filter.set_residue_selector(chain_sel)
    #     wnm = wnm_filter.report_sm(pose)
    #     pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name + '_' + str(chain_num), wnm)

    # import pyrosetta
    # wnm_filter = pyrosetta.rosetta.protocols.simple_filters.Worst9merFilter()
    # wnm_filter.set_user_defined_name(name)
    # for chain_num in range(1, pose.num_chains() + 1):
    #     score_on_chain_subset(pose, wnm_filter, [chain_num])

def score_wnm_helix(pose:Pose, name:str='wnm_hlx'):
    # loading the database takes ~5-6 GB of memory, but once loaded, remains for the rest of the python session
    import pyrosetta
    # using an xml to create the worst9mer filter because I couldn't figure out how to use pyrosetta without completely crashing python
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        """
        <FILTERS>
            <Worst9mer name="wnm_hlx" rmsd_lookup_threshold="0.4" confidence="0" only_helices="true" />
        </FILTERS>
        """
    )
    wnm_hlx_filter = objs.get_filter("wnm_hlx")
    wnm_hlx_filter.set_user_defined_name(name)
    wnm = wnm_hlx_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, wnm)

    # import pyrosetta
    # hlx_sel = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector()
    # hlx_sel.set_selected_ss('H')
    # hlx_sel.set_overlap(0)
    # hlx_sel.set_minH(3)
    # hlx_sel.set_use_dssp(True)
    # wnm_hlx_filter = pyrosetta.rosetta.protocols.simple_filters.Worst9merFilter()
    # wnm_hlx_filter.set_user_defined_name(name)
    # wnm_hlx_filter.set_residue_selector(hlx_sel)
    # wnm = wnm_filter.report_sm(pose)
    # pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, wnm)

def score_per_res(pose: Pose, scorefxn:ScoreFunction, name: str = 'score'):
    import pyrosetta
    score_filter = gen_score_filter(scorefxn, name)
    score = score_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, score)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name + '_per_res', score/pose.size())

@requires_init
def one_state_design_unlooped_dimer(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Generator[PackedPose, PackedPose, None]:

    from time import time
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io

    # sys.path.insert(0, "/mnt/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    start_time = time()
    def print_timestamp(print_str, end='\n', *args):
        time_min = (time() - start_time)/60
        print(f'{time_min:.2f} min: {print_str}', end=end)
        for arg in args:
            print(arg)

    design_sel = interface_among_chains(chain_list=[1, 2, 3, 4], vector_mode=True)
    print_timestamp('Generated interface selector')
    task_factory = gen_task_factory(design_sel=design_sel,
                                    pack_nbhd=False,
                                    extra_rotamers_level=2,
                                    limit_arochi=True,
                                    prune_buns=True,
                                    upweight_ppi=True,
                                    restrict_pro_gly=True,
                                    ifcl=False,
                                    layer_design=None)
    print_timestamp('Generated interface design task factory')
    clean_sfxn = pyrosetta.create_score_function('beta_nov16.wts')
    design_sfxn = pyrosetta.create_score_function('beta_nov16.wts')
    design_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0)
    print_timestamp('Generated score functions')

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = 'none'
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors() # for the neighborhood residue selector

        # testing
        from crispy_shifty.utils.io import pymol_selection
        print(pymol_selection(pose, design_sel, 'design_sel'))

        print_timestamp('Generating structure profile...', end='')
        struct_profile(pose, design_sel)
        print('complete.')

        print_timestamp('Starting 1 round of fixed backbone design...', end='')
        fastdesign(pose, task_factory, scorefxn=design_sfxn, flexbb=False, repeats=1)
        print('complete.')
        print_timestamp('Starting 2 rounds of flexible backbone design...', end='')
        fastdesign(pose, task_factory, scorefxn=design_sfxn, flexbb=True, repeats=2) # comment out for testing
        print('complete.')

        print_timestamp('Clearing constraints...', end='')
        clear_constraints(pose)
        print('complete.')
        
        print_timestamp('Scoring contact molecular surface and shape complementarity...', end='')
        an_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(1)
        ac_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(2)
        bn_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(3)
        bc_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(4)
        dhr_sel = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(an_sel, bc_sel)
        selector_pairs = [(an_sel, bc_sel),
                          (ac_sel, dhr_sel),
                          (bn_sel, dhr_sel),
                          (ac_sel, bn_sel)]
        pair_names = ['dhr', 'dhr_ac', 'dhr_bn', 'ac_bn']
        for (sel_1, sel_2), name in zip(selector_pairs, pair_names):
            # score_cms(pose, sel_1, sel_2, 'cms_' + name) # need the residue selectors to be exposed in contactmolecularsurfacefilter; until then, use the annoying rosettascript below
            score_sc(pose, sel_1, sel_2, 'sc_' + name)
        print('complete.')

        print_timestamp('Scoring secondary structure shape complementarity...', end='')
        score_ss_sc(pose)
        print('complete.')
        print_timestamp('Scoring per_chain worst9mer...', end='')
        score_wnm(pose)
        print('complete.')
        print_timestamp('Scoring helical worst9mer...', end='')
        score_wnm_helix(pose)
        print('complete.')

        print_timestamp('Scoring...', end='')
        score_per_res(pose, clean_sfxn)
        score_filter = gen_score_filter(clean_sfxn)
        chain_lists = [[1], [2], [3], [4], [1,4], [2,3], [1,2,4], [1,3,4]]
        for chain_list in chain_lists:
            score_on_chain_subset(pose, score_filter, chain_list)
        print('complete.')

        # For now, use this annoying rosettascript to calculate contact molecular surface. Later, uncomment the line above.
        from pyrosetta.distributed.tasks.rosetta_scripts import SingleoutputRosettaScriptsTask
        cms_rs = SingleoutputRosettaScriptsTask(
            """
            <ROSETTASCRIPTS>
                <RESIDUE_SELECTORS>
                    <Chain name="chAN" chains="A"/>
                    <Chain name="chAC" chains="B"/>
                    <Chain name="chBN" chains="C"/>
                    <Chain name="chBC" chains="D"/>
                    <Or name="recon_DHR" selectors="chAN,chBC" />
                </RESIDUE_SELECTORS>
                <FILTERS>
                    <ContactMolecularSurface name="cms_dhr" target_selector="chAN" binder_selector="chAC" confidence="0" />
                    <ContactMolecularSurface name="cms_dhr_ac" target_selector="recon_DHR" binder_selector="chAC" confidence="0" />
                    <ContactMolecularSurface name="cms_dhr_bn" target_selector="recon_DHR" binder_selector="chBN" confidence="0" />
                    <ContactMolecularSurface name="cms_ac_bn" target_selector="chAC" binder_selector="chBN" confidence="0" />
                </FILTERS>
                <PROTOCOLS>
                    <Add filter_name="cms_dhr" />
                    <Add filter_name="cms_dhr_ac" />
                    <Add filter_name="cms_dhr_bn" />
                    <Add filter_name="cms_ac_bn" />
                </PROTOCOLS>
            </ROSETTASCRIPTS>
            """
        )
        print_timestamp('Scoring contact molecular surface...', end='')
        pose = io.to_pose(cms_rs(pose))
        print('complete.')

        # Here's another annoying rosettascript for worst9mer, since it apparently wants to crash python every time I run it from pyrosetta...
        # wnm_rs = SingleoutputRosettaScriptsTask(
        #     """
        #     <ROSETTASCRIPTS>
        #         <RESIDUE_SELECTORS>
        #             <Chain name="an_sel" chains="A"/>
        #             <Chain name="ac_sel" chains="B"/>
        #             <Chain name="bn_sel" chains="C"/>
        #             <Chain name="bc_sel" chains="D"/>
        #         </RESIDUE_SELECTORS>
        #         <FILTERS>
        #             <Worst9mer name="wnm_an" residue_selector="an_sel" rmsd_lookup_threshold="0.4" confidence="0" />
        #             <Worst9mer name="wnm_ac" residue_selector="ac_sel" rmsd_lookup_threshold="0.4" confidence="0" />
        #             <Worst9mer name="wnm_bn" residue_selector="bn_sel" rmsd_lookup_threshold="0.4" confidence="0" />
        #             <Worst9mer name="wnm_bc" residue_selector="bc_sel" rmsd_lookup_threshold="0.4" confidence="0" />
        #         </FILTERS>
        #         <PROTOCOLS>
        #             <Add filter_name="wnm_an" />
        #             <Add filter_name="wnm_ac" />
        #             <Add filter_name="wnm_bn" />
        #             <Add filter_name="wnm_bc" />
        #         </PROTOCOLS>
        #     </ROSETTASCRIPTS>
        #     """
        # )
        # pose = io.to_pose(wnm_rs(pose))

        ppose = io.to_packed(pose)
        yield ppose
