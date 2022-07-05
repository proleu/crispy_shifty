# Python standard library
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from pyrosetta.distributed import requires_init

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector


########## Azobenzene Crosslinking ##########
# The main function for azobenzene crosslinking.  Use with the following init:
# pyrosetta.init(
#    " ".join(
#        [
#            "-beta",
#            "-corrections:gen_potential true",
#            "-out:level 100",
#            "-in:file:extra_res_fa AZC.params",
#            "-in:file:extra_res_fa AZT.params",
#            "-gen_bonded_params_file scoring/score_functions/generic_potential/generic_bonded.aamerge.txt",
#            "-genbonded_score_hybrid true",
#        ]
#    )
# )
# params files can be found in crispy_shifty/params
# Minimal usage: add_azo(pose, selection, residue_name)
# pose - input pose
# selection - ResidueIndexSelector or str specifying two residues
# residue_name - What is the crosslinker called?
# Other options:
# add_constraints - bool, should constraints be added to the crosslinker?
# filter_by_sidechain_distance - float, negative for no filtering, positive to set a threshold in angstroms for crosslinking to be attempted.  Use this in a try/except block as it will throw an AssertionError
# filter_by_cst_energy - float, negative for no filtering, positive to set a threshold in Rosetta energy units (for constraint scores only) for crosslinking to be accepted.  Use in a try/except block
# filter_by_total_score - same as filter_by_cst_energy, but total_score instead of constraint scores
# force_cys - bool, True to mutate selected residues to Cys, False to error if one of the selected residues is not Cys
# sc_fast_relax_rounds - int, number of round of fast relax for the linked sidechains and linker only
# final_fast_relax_rounds - int, number of round of whole-structure fast relax to do after relaxing sidechains and linker
# custom_movemap - override the default movemap for the final relax (untested)
# rmsd_filter - None to not calculate rmsd.  {"sele":StrOfIndices, "super_sele":StrOfIndices, "type":str(pyrosetta.rosetta.core.scoring.rmsd_atoms member), "save":bool, "cutoff":float}
@requires_init
def add_azo(
    selection,
    residue_name: str,
    pose_in=None,
    add_constraints: bool = False,
    filter_by_sidechain_distance: float = -1,
    fbsd_ub: float = -1,
    fbsd_lb: float = -1,
    filter_by_cst_energy: float = -1,
    filter_by_total_score: float = None,
    force_cys: bool = False,
    sc_fast_relax_rounds: int = 1,
    tors_fast_relax_rounds: int = 0,
    cart_fast_relax_rounds: int = 0,
    custom_movemap=None,
    rmsd_filter: float = -1,
    rmsd_sele: str = None,
    super_sele: str = None,
    rmsd_type: str = "rmsd_protein_bb_ca",
    save_rmsd: bool = True,
    invert_rmsd: bool = False,
    invert_fbsd: bool = False,
    pass_rmsd: bool = False,
    ramp_cart_bonded: bool = True,
    pdb_path: str = None,
):
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    import pyrosetta

    # Force correct typing
    residue_name = str(residue_name)
    add_constraints = bool(add_constraints)
    filter_by_sidechain_distance = float(filter_by_sidechain_distance)
    fbsd_ub = float(fbsd_ub)
    fbsd_lb = float(fbsd_lb)
    filter_by_cst_energy = float(filter_by_cst_energy)
    if filter_by_total_score is not None:
        filter_by_total_score = float(filter_by_total_score)
    force_cys = bool(force_cys)
    sc_fast_relax_rounds = int(sc_fast_relax_rounds)
    tors_fast_relax_rounds = int(tors_fast_relax_rounds)
    cart_fast_relax_rounds = int(cart_fast_relax_rounds)
    rmsd_filter = float(rmsd_filter)
    #rmsd_sele = str(rmsd_sele)
    #super_sele = str(super_sele)
    rmsd_type = str(rmsd_type)
    save_rmsd = bool(save_rmsd)
    invert_rmsd = bool(invert_rmsd)
    invert_fbsd = bool(invert_fbsd)
    pass_rmsd = bool(pass_rmsd)
    ramp_cart_bonded = bool(ramp_cart_bonded)
    # Define a bunch of functions
    def get_linker_index(pose, res_indices):
        assert len(res_indices) == 2, "Incorrect number of residues specified"
        nconn1 = pose.residue(res_indices[0]).n_possible_residue_connections()
        nconn2 = pose.residue(res_indices[1]).n_possible_residue_connections()
        azo_index = pose.residue(res_indices[0]).residue_connection_partner(nconn1)
        return azo_index

    def filter_by_sc_dist(pose, selection, cutoff, ub=None, lb=None):
        cutoff_sq = cutoff ** 2
        res_indices = [i + 1 for i, b in enumerate(selection.apply(pose)) if b]
        assert len(res_indices) == 2, "Incorrect number of residues specified"
        d1 = (
            pose.residue(res_indices[0])
            .xyz("CB")
            .distance_squared(pose.residue(res_indices[1]).xyz("CB"))
        )
        print(f"Sidechain distance: {d1**0.5}", flush=True)
        if ub is None or lb is None:
            return d1 < cutoff_sq
        else:
            assert (
                ub > lb
            ), "Upper limit of sidechain distance is less than the lower limit!"
            return lb ** 2 < d1 < ub ** 2

    def add_linker_bonds_asymmetric(pose, res_indices, linker_index):
        bond1 = pyrosetta.rosetta.protocols.simple_moves.DeclareBond()
        bond2 = pyrosetta.rosetta.protocols.simple_moves.DeclareBond()
        bond1.set(linker_index, "C13", res_indices[0], "SG", False)
        bond2.set(linker_index, "C17", res_indices[1], "SG", False)
        bond1.apply(pose)
        bond2.apply(pose)

    def add_linker_csts(pose, selection, residue_name):
        res_indices = [i + 1 for i, b in enumerate(selection.apply(pose)) if b]
        azo_index = get_linker_index(pose, res_indices)
        # Distance constraints
        dist_cst_str = "HARMONIC 0.0 0.01"
        dist_csts = (
            pyrosetta.rosetta.protocols.cyclic_peptide.CreateDistanceConstraint()
        )
        res1 = pyrosetta.rosetta.utility.vector1_unsigned_long()
        res2 = pyrosetta.rosetta.utility.vector1_unsigned_long()
        atm1 = pyrosetta.rosetta.utility.vector1_std_string()
        atm2 = pyrosetta.rosetta.utility.vector1_std_string()
        cst_fxn = pyrosetta.rosetta.utility.vector1_std_string()
        res1.append(res_indices[0])
        res1.append(res_indices[1])
        atm2.append("V1")
        atm2.append("V2")
        res1.append(res_indices[0])
        res1.append(res_indices[1])
        atm2.append("C13")
        atm2.append("C17")
        for i in range(0, 2):
            atm1.append("SG")
            res2.append(azo_index)
            cst_fxn.append(dist_cst_str)
        for i in range(2, 4):
            atm1.append("V1")
            res2.append(azo_index)
            cst_fxn.append(dist_cst_str)
        print("R1\tA1\tR2\tA2\tFUNC", flush=True)
        for i in range(1, len(atm1) + 1):
            print(
                f"{res1[i]}\t{atm1[i]}\t{res2[i]}\t{atm2[i]}\t{cst_fxn[i]}", flush=True
            )
        dist_csts.set(res1, atm1, res2, atm2, cst_fxn)
        dist_csts.apply(pose)
        # Torsion constraints
        tors_csts = pyrosetta.rosetta.protocols.cyclic_peptide.CreateTorsionConstraint()
        res1 = pyrosetta.rosetta.utility.vector1_unsigned_long()
        res2 = pyrosetta.rosetta.utility.vector1_unsigned_long()
        res3 = pyrosetta.rosetta.utility.vector1_unsigned_long()
        res4 = pyrosetta.rosetta.utility.vector1_unsigned_long()
        atm1 = pyrosetta.rosetta.utility.vector1_std_string()
        atm2 = pyrosetta.rosetta.utility.vector1_std_string()
        atm3 = pyrosetta.rosetta.utility.vector1_std_string()
        atm4 = pyrosetta.rosetta.utility.vector1_std_string()
        cst_fxn = pyrosetta.rosetta.utility.vector1_std_string()
        # C-SG-CB-CA
        res1.append(azo_index)
        res2.append(res_indices[0])
        res3.append(res_indices[0])
        res4.append(res_indices[0])
        atm1.append("C13")
        atm2.append("SG")
        atm3.append("CB")
        atm4.append("CA")
        cst_fxn.append("AMBERPERIODIC 0 3 2")
        res1.append(azo_index)
        res2.append(res_indices[1])
        res3.append(res_indices[1])
        res4.append(res_indices[1])
        atm1.append("C17")
        atm2.append("SG")
        atm3.append("CB")
        atm4.append("CA")
        cst_fxn.append("AMBERPERIODIC 0 3 2")
        # N-C-C-SG
        res1.append(azo_index)
        res2.append(azo_index)
        res3.append(azo_index)
        res4.append(res_indices[0])
        atm1.append("N4")
        atm2.append("C12")
        atm3.append("C13")
        atm4.append("SG")
        cst_fxn.append("AMBERPERIODIC 0 2 2")
        res1.append(azo_index)
        res2.append(azo_index)
        res3.append(azo_index)
        res4.append(res_indices[1])
        atm1.append("N1")
        atm2.append("C1")
        atm3.append("C17")
        atm4.append("SG")
        cst_fxn.append("AMBERPERIODIC 0 2 2")
        # C-C-SG-CB
        res1.append(azo_index)
        res2.append(azo_index)
        res3.append(res_indices[0])
        res4.append(res_indices[0])
        atm1.append("C12")
        atm2.append("C13")
        atm3.append("SG")
        atm4.append("CB")
        cst_fxn.append("AMBERPERIODIC 0 3 2")
        res1.append(azo_index)
        res2.append(azo_index)
        res3.append(res_indices[1])
        res4.append(res_indices[1])
        atm1.append("C1")
        atm2.append("C17")
        atm3.append("SG")
        atm4.append("CB")
        cst_fxn.append("AMBERPERIODIC 0 3 2")
        # C5-N2-N3-C6
        res1.append(azo_index)
        res2.append(azo_index)
        res3.append(azo_index)
        res4.append(azo_index)
        atm1.append("C5")
        atm2.append("N2")
        atm3.append("N3")
        atm4.append("C6")
        if residue_name == "AZT":
            angle = pyrosetta.rosetta.numeric.conversions.radians(180)
            cst_fxn.append(f"CIRCULARHARMONIC {angle} 0.1")
        elif residue_name == "AZC":
            cst_fxn.append("CIRCULARHARMONIC 0 0.1")
        else:
            sys.exit("Residue name not set up for constriants")
        tors_csts.set(res1, atm1, res2, atm2, res3, atm3, res4, atm4, cst_fxn)
        tors_csts.apply(pose)

    def get_jump_index_for_crosslinker(pose, linker_index):
        foldtree = pose.fold_tree()
        assert foldtree.is_jump_point(linker_index), "The linker is not a jump point"
        return foldtree.get_jump_that_builds_residue(linker_index)

    def pack_and_min(
        pose,
        selection,
        whole_structure: bool = False,
        rounds=1,
        cartesian=False,
        ramp_cart_bonded=True,
        custom_movemap=None,
    ):
        # Set up for cart relax
        if ramp_cart_bonded:
            import numpy as np

            cb_weights = list(np.logspace(3, 0, rounds))
        else:
            cb_weights = [1]
        helper = (
            pyrosetta.rosetta.protocols.cyclic_peptide.crosslinker.CrosslinkerMoverHelper()
        )
        sfxn = pyrosetta.create_score_function("beta_genpot_cst.wts")

        def apply_cart(pose, sfxn, cb_weights, helper, movemap=None):
            for cbw in cb_weights:
                print(
                    f"Performing 1 round of cartesian FastRelax with cart_bonded={cbw}",
                    flush=True,
                )
                print(movemap, flush=True)
                helper.pre_relax_round_update_steps(
                    pose, selection.apply(pose), whole_structure, False, True
                )
                sfxn.set_weight(pyrosetta.rosetta.core.scoring.pro_close, 0)
                sfxn.set_weight(pyrosetta.rosetta.core.scoring.cart_bonded, cbw)
                frlx = pyrosetta.rosetta.protocols.relax.FastRelax(sfxn, 1)
                if movemap is not None:
                    frlx.set_movemap(movemap)
                frlx.cartesian(True)
                print("applied cart", flush=True)
                frlx.apply(pose)
                print("relaxed", flush=True)
                helper.post_relax_round_update_steps(
                    pose, selection.apply(pose), whole_structure, False, True
                )
                print("cart done", flush=True)

        # Set up for any relax
        frlx = pyrosetta.rosetta.protocols.relax.FastRelax(sfxn, 1)
        movemap = None
        # Set up movemap for whole structure or sc/crosslinker
        if whole_structure and custom_movemap is not None:
            movemap = custom_movemap
        if not whole_structure:
            res_indices = [i + 1 for i, b in enumerate(selection.apply(pose)) if b]
            azo_index = get_linker_index(pose, res_indices)
            movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
            movemap.set_bb(False)
            movemap.set_chi(False)
            movemap.set_jump(False)
            for res in res_indices:
                movemap.set_chi(res, True)
            movemap.set_chi(azo_index, True)
            movemap.set_jump(get_jump_index_for_crosslinker(pose, azo_index), True)
            # frlx.set_movemap(movemap)
            # frlx.set_movemap_disables_packing_of_fixed_chi_positions(True)
        # Cart relax
        if cartesian:
            apply_cart(pose, sfxn, cb_weights, helper, movemap)
        # Tors relax
        else:
            for i in range(0, rounds):
                if not whole_structure:
                    print(
                        "Performing 1 round of sidechain/crosslinker FastRelax",
                        flush=True,
                    )
                else:
                    print("Performing 1 round of torsional FastRelax", flush=True)
                helper.pre_relax_round_update_steps(
                    pose, selection.apply(pose), whole_structure, False, True
                )
                if movemap is not None:
                    frlx.set_movemap(movemap)
                    frlx.set_movemap_disables_packing_of_fixed_chi_positions(True)
                frlx.apply(pose)
                helper.post_relax_round_update_steps(
                    pose, selection.apply(pose), whole_structure, False, True
                )

    def cst_energy_filter(pose, selection, cutoff):
        score_pose = pose.clone()
        pyrosetta.rosetta.protocols.constraint_movers.ClearConstraintsMover().apply(
            score_pose
        )
        add_linker_csts(score_pose, selection, residue_name)
        sfxn = pyrosetta.rosetta.core.scoring.ScoreFunction()
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
        sfxn(score_pose)
        cst_energy = score_pose.energies().total_energy()
        return cst_energy > cutoff

    def total_score_filter(pose, cutoff):
        sfxn = pyrosetta.create_score_function("beta_genpot_cst.wts")
        sfxn(pose)
        total_eng = pose.energies().total_energy()
        return total_eng > cutoff

    def filter_by_rmsd(
        pose,
        pose_in,
        selection,
        rmsd_filter,
        rmsd_sele,
        super_sele,
        rmsd_type,
        save_rmsd,
    ):
        from pyrosetta.rosetta.core.select.residue_selector import (
            AndResidueSelector as AndResidueSelector,
        )
        from pyrosetta.rosetta.core.select.residue_selector import (
            NotResidueSelector as NotResidueSelector,
        )
        from pyrosetta.rosetta.core.select.residue_selector import (
            ResidueIndexSelector as ResidueIndexSelector,
        )
        from pyrosetta.rosetta.core.select.residue_selector import (
            TrueResidueSelector as TrueResidueSelector,
        )
        print(type(rmsd_sele))
        if rmsd_sele is None:
            rmsd_sele = AndResidueSelector(
                TrueResidueSelector(), NotResidueSelector(selection)
            )
        else:
            rmsd_sele = AndResidueSelector(
                ResidueIndexSelector(rmsd_sele), NotResidueSelector(selection)
            )
        if super_sele is None:
            super_sele = rmsd_sele
        else:
            super_sele = AndResidueSelector(
                ResidueIndexSelector(super_sele),
                NotResidueSelector(selection),
            )
        rmsd_resis=",".join([str(i+1) for i,b in enumerate(rmsd_sele.apply(pose_in)) if b])
        super_resis=",".join([str(i+1) for i,b in enumerate(super_sele.apply(pose_in)) if b])
        print(rmsd_resis,flush=True)
        print(super_resis,flush=True)
        rmsd_sele=ResidueIndexSelector(rmsd_resis)
        super_sele=ResidueIndexSelector(super_resis)
        #rmsd_sele=AndResidueSelector(rmsd_sele,NotResidueSelector(ResidueIndexSelector(azo_index)))
        print(super_sele.apply(pose), flush=True)
        print(rmsd_sele.apply(pose), flush=True)
        print(super_sele, flush=True)
        print(rmsd_sele, flush=True)
        print("making rmsd metric", flush=True)
        rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
        rmsd_metric.set_residue_selector(rmsd_sele)
        rmsd_metric.set_residue_selector_reference(rmsd_sele)
        rmsd_metric.set_residue_selector_super(super_sele)
        rmsd_metric.set_residue_selector_super_reference(super_sele)
        print(f"setting rmsd_type: pyrosetta.rosetta.core.scoring.rmsd_atoms.{rmsd_type}", flush=True)
        rmsd_metric.set_rmsd_type(
            eval(f"pyrosetta.rosetta.core.scoring.rmsd_atoms.{rmsd_type}")
        )
        rmsd_metric.set_run_superimpose(True)
        rmsd_metric.set_comparison_pose(pose_in)
        print("calculating", flush=True)
        rmsd = rmsd_metric.calculate(pose)
        if save_rmsd:
            print("saving rmsd",flush=True)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                pose, "crosslinking_rmsd", rmsd
            )
        print(f"RMSD: {rmsd}", flush=True)
        return rmsd > rmsd_filter

    if isinstance(pose_in, pyrosetta.distributed.packed_pose.core.PackedPose):
        pose_in = pyrosetta.distributed.packed_pose.core.to_pose(pose_in)
    elif pose_in is None:
        for pose_in in path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        ):
            break
    pose = pose_in.clone()
    if type(selection) == str:
        selection_str=selection
        selection = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
            selection
        )
    res_indices = [i + 1 for i, b in enumerate(selection.apply(pose)) if b]
    assert len(res_indices) == 2, "Incorrect number of residues specified"
    cys1 = res_indices[0]
    cys2 = res_indices[1]
    if force_cys:
        for res in res_indices:
            mut0 = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
            mut0.set_res_name("CYS")
            mut0.set_target(res)
            mut0.apply(pose)

    def is_cys(res):
        return pose.residue_type(res).aa() == pyrosetta.rosetta.core.chemical.aa_cys

    def is_dcs(res):
        return pose.residue_type(res).aa() == pyrosetta.rosetta.core.chemical.aa_dcs

    for res in res_indices:
        assert is_cys(res) or is_dcs(res), f"ERROR: Residue {res} is not a cysteine"

    # Filter by sidechain distance
    check_ub_lb = fbsd_ub > 0 and fbsd_lb > 0
    if filter_by_sidechain_distance > 0 and not check_ub_lb:
        if invert_fbsd:
            assert not filter_by_sc_dist(
                pose, selection, filter_by_sidechain_distance
            ), f"CB of positions {res_indices[0]} and {res_indices[1]} are greater than {filter_by_sidechain_distance}A apart."
        else:
            assert filter_by_sc_dist(
                pose, selection, filter_by_sidechain_distance
            ), f"CB of positions {res_indices[0]} and {res_indices[1]} are greater than {filter_by_sidechain_distance}A apart."
    elif check_ub_lb:
        if invert_fbsd:
            assert not filter_by_sc_dist(
                pose, selection, 1, fbsd_ub, fbsd_lb
            ), f"CB of positions {res_indices[0]} and {res_indices[1]} are between {fbsd_lb} and {fbsd_ub} apart."
        else:
            assert filter_by_sc_dist(
                pose, selection, 1, fbsd_ub, fbsd_lb
            ), f"CB of positions {res_indices[0]} and {res_indices[1]} are not between {fbsd_lb} and {fbsd_ub} apart."

    # Mutate CYS -> CYX
    mut1 = pyrosetta.rosetta.protocols.simple_moves.ModifyVariantTypeMover()
    index_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    for index in res_indices:
        index_sel.append_index(index)
    mut1.set_residue_selector(index_sel)
    mut1.set_additional_type_to_add("SIDECHAIN_CONJUGATION")
    mut1.apply(pose)

    # Create AZO
    standard_residues = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
    new_rsd = pyrosetta.rosetta.core.conformation.ResidueFactory().create_residue(
        standard_residues.residue_type_set("fa_standard").name_map(residue_name)
    )
    azo_pose = pyrosetta.rosetta.core.pose.Pose()
    azo_pose.append_residue_by_jump(new_rsd, 1)

    # Align the AZO pose to the input pose
    alignment_atoms = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    alignment_atoms[
        pyrosetta.rosetta.core.id.AtomID(new_rsd.type().atom_index("V1"), 1)
    ] = pyrosetta.rosetta.core.id.AtomID(pose.residue_type(cys1).atom_index("SG"), cys1)
    alignment_atoms[
        pyrosetta.rosetta.core.id.AtomID(new_rsd.type().atom_index("V2"), 1)
    ] = pyrosetta.rosetta.core.id.AtomID(pose.residue_type(cys2).atom_index("SG"), cys2)
    alignment_atoms[
        pyrosetta.rosetta.core.id.AtomID(new_rsd.type().atom_index("C13"), 1)
    ] = pyrosetta.rosetta.core.id.AtomID(pose.residue_type(cys1).atom_index("V1"), cys1)
    alignment_atoms[
        pyrosetta.rosetta.core.id.AtomID(new_rsd.type().atom_index("C17"), 1)
    ] = pyrosetta.rosetta.core.id.AtomID(pose.residue_type(cys2).atom_index("V1"), cys2)
    pyrosetta.rosetta.core.scoring.superimpose_pose(azo_pose, pose, alignment_atoms)

    # Merge poses
    pose.append_residue_by_jump(azo_pose.residue(1), cys1)
    azo_res = pose.total_residue()

    # Declare covalent bonds
    add_linker_bonds_asymmetric(pose, res_indices, azo_res)

    # Add constraints
    if add_constraints:
        add_linker_csts(pose, selection, residue_name)

    # Pack linker and sidechains
    if sc_fast_relax_rounds > 0:
        pack_and_min(pose, selection, False, sc_fast_relax_rounds)

    # Filter by constraint energy
    if filter_by_cst_energy > 0:
        assert not cst_energy_filter(
            pose, selection, filter_by_cst_energy
        ), "Failed constraint energy filter after relaxing linker and sidechains"

    # Final fast relax
    if tors_fast_relax_rounds > 0:
        pack_and_min(
            pose, selection, True, tors_fast_relax_rounds, False, custom_movemap
        )

    if cart_fast_relax_rounds > 0:
        pack_and_min(
            pose, selection, True, cart_fast_relax_rounds, True, custom_movemap
        )

    # Filter by constraint energy
    if filter_by_cst_energy > 0:
        print("Filtering by constraint energy", flush=True)
        assert not cst_energy_filter(
            pose, selection, filter_by_cst_energy
        ), "Failed constraint energy filter after final fast relax"

    # Filter by total score
    if filter_by_total_score is not None:
        print("Filtering by total score", flush=True)
        assert not total_score_filter(pose, filter_by_total_score)

    # Filter by rmsd
    if rmsd_filter > 0:
        res_indices = [i + 1 for i, b in enumerate(selection.apply(pose)) if b]
        azo_index = get_linker_index(pose, res_indices)
        print("Calculating RMSD", flush=True)
        rmsd_passed = filter_by_rmsd(
                pose,
                pose_in,
                selection,
                rmsd_filter,
                rmsd_sele,
                super_sele,
                rmsd_type,
                save_rmsd,
            )
        if not pass_rmsd:
            if invert_rmsd:
                assert rmsd_passed, "Failed RMSD filter"
            else:
                assert not rmsd_passed, "Failed RMSD filter"
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            pose, "crosslinked_residues", selection_str
        )
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            pose, "crosslinker_name", residue_name
        )
    return pose


# Adds azobenzene to chainB, cart minimizes chain B only
@requires_init
def add_azo_chainB(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    import copy
    import sys

    import pyrosetta
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    required = ["pdb_path", "selection", "residue_name"]
    for req in required:
        if req not in kwargs.keys():
            sys.exit(f"Missing required argument {req}")

    optional = [
        "ramp_cart_bonded",
        "pass_rmsd",
        "fbsd_ub",
        "fbsd_lb",
        "invert_rmsd",
        "invert_fbsd",
        "add_constraints",
        "filter_by_sidechain_distance",
        "filter_by_cst_energy",
        "filter_by_total_score",
        "force_cys",
        "sc_fast_relax_rounds",
        "tors_fast_relax_rounds",
        "cart_fast_relax_rounds",
        "custom_movemap",
        "rmsd_filter",
        "rmsd_sele",
        "super_sele",
        "rmsd_type",
        "save_rmsd",
    ]
    provided = []
    kwargs_bak = copy.deepcopy(kwargs)
    for key, val in kwargs_bak.items():
        if key in optional:
            provided.append(key)
        else:
            if key not in required:
                print(f"Warning: Ignoring unrecognized option {key}")
                del kwargs[key]
    pdb_path = kwargs["pdb_path"]
    del kwargs["pdb_path"]
    for pose in path_to_pose_or_ppose(
        path=pdb_path, cluster_scores=True, pack_result=False
    ):
        custom_movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
        custom_movemap.set_bb(False)
        custom_movemap.set_chi(True)
        custom_movemap.set_jump(True)
        chB = [i + 1 for i, b in enumerate(ChainSelector(2).apply(pose)) if b]
        for res in chB:
            custom_movemap.set_bb(True)
        kwargs["pose_in"] = pose
        kwargs["custom_movemap"] = custom_movemap
        if "pass_rmsd" in kwargs.keys():
            pass_rmsd = bool(kwargs["pass_rmsd"])
            print(f"Updating pass_rmsd from kwargs: {pass_rmsd}")
        else:
            pass_rmsd = False    
        if "rmsd_filter" not in kwargs.keys():
            kwargs["rmsd_filter"] = 10

        if "rmsd_sele" not in kwargs.keys():
            kwargs["rmsd_sele"] = ",".join(
                [
                    str(i + 1)
                    for i, b in enumerate(
                        pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
                            2
                        ).apply(
                            pose
                        )
                    )
                    if b
                ]
            )
        if "super_sele" not in kwargs.keys():
            kwargs["super_sele"] = ",".join(
                [
                    str(i + 1)
                    for i, b in enumerate(
                        pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
                            1
                        ).apply(
                            pose
                        )
                    )
                    if b
                ]
            )
        if "rmsd_type" not in kwargs.keys():
            kwargs["rmsd_type"] = "rmsd_protein_bb_heavy"
        if "save_rmsd" not in kwargs.keys():
            kwargs["save_rmsd"] = True
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            pose, "crosslinked_residues", kwargs["selection"]
        )
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            pose, "crosslinker_name", kwargs["residue_name"]
        )
        pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "input_pdb", pdb_path)
        return add_azo(**kwargs)
