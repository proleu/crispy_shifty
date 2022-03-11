from pyrosetta.distributed import requires_init

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
# Minimal usage: add_azo(pose, selection)
# pose - input pose
# selection - ResidueIndexSelector or str specifying two residues
# Other options:
# add_constraints - bool, should constraints be added to the crosslinker?
# filter_by_sidechain_distance - float, negative for no filtering, positive to set a threshold in angstroms for crosslinking to be attempted.  Use this in a try/except block as it will throw an AssertionError
# filter_by_cst_energy - float, negative for no filtering, positive to set a threshold in Rosetta energy units (for constraint scores only) for crosslinking to be accepted.  Use in a try/except block
# filter_by_total_score - same as filter_by_cst_energy, but total_score instead of constraint scores
# force_cys - bool, True to mutate selected residues to Cys, False to error if one of the selected residues is not Cys
# sc_fast_relax_rounds - int, number of round of fast relax for the linked sidechains and linker only
# final_fast_relax_rounds - int, number of round of whole-structure fast relax to do after relaxing sidechains and linker
# cartesian - bool, should the final fast relax be in cartesian space?
# custom_movemap - override the default movemap for the final relax (untested)
# rmsd_filter - None to not calculate rmsd.  {"sele":StrOfIndices, "super_sele":StrOfIndices, "type":str(pyrosetta.rosetta.core.scoring.rmsd_atoms member), "save":bool, "cutoff":float}
def add_azo(
    pose_in,
    selection,
    residue_name: str,
    add_constraints: bool = False,
    filter_by_sidechain_distance: float = -1,
    filter_by_cst_energy: float = -1,
    filter_by_total_score: float = None,
    force_cys: bool = False,
    sc_fast_relax_rounds: int = 1,
    final_fast_relax_rounds: int = 0,
    cartesian: bool = False,
    custom_movemap=None,
    rmsd_filter: dict = None,
):
    import pyrosetta

    def get_linker_index(pose, res_indices):
        assert len(res_indices) == 2, "Incorrect number of residues specified"
        nconn1 = pose.residue(res_indices[0]).n_possible_residue_connections()
        nconn2 = pose.residue(res_indices[1]).n_possible_residue_connections()
        azo_index = pose.residue(res_indices[0]).residue_connection_partner(nconn1)
        return azo_index

    def filter_by_sc_dist(pose, selection, cutoff):
        cutoff_sq = cutoff ** 2
        res_indices = [i + 1 for i, b in enumerate(selection.apply(pose)) if b]
        assert len(res_indices) == 2, "Incorrect number of residues specified"
        d1 = (
            pose.residue(res_indices[0])
            .xyz("CB")
            .distance_squared(pose.residue(res_indices[1]).xyz("CB"))
        )
        return d1 > cutoff_sq

    def add_linker_bonds_asymmetric(pose, res_indices, linker_index):
        bond1 = pyrosetta.rosetta.protocols.cyclic_peptide.DeclareBond()
        bond2 = pyrosetta.rosetta.protocols.cyclic_peptide.DeclareBond()
        bond1.set(linker_index, "C13", res_indices[0], "SG", False)
        bond2.set(linker_index, "C17", res_indices[1], "SG", False)
        bond1.apply(pose)
        bond2.apply(pose)

    def add_linker_csts(pose, selection):
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
        custom_movemap=None,
    ):
        helper = (
            pyrosetta.rosetta.protocols.cyclic_peptide.crosslinker.CrosslinkerMoverHelper()
        )
        sfxn = pyrosetta.create_score_function("beta_genpot_cst.wts")
        if whole_structure and cartesian:
            sfxn = pyrosetta.create_score_function("beta_genpot_cart.wts")
            sfxn.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
            sfxn.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
            sfxn.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
        frlx = pyrosetta.rosetta.protocols.relax.FastRelax(sfxn, 1)
        if whole_structure and cartesian:
            frlx.cartesian(True)
        for i in range(0, rounds):
            helper.pre_relax_round_update_steps(
                pose, selection.apply(pose), whole_structure, False, True
            )
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
            if whole_structure and custom_movemap is not None:
                movemap = custom_movemap
            frlx.apply(pose)
            helper.post_relax_round_update_steps(
                pose, selection.apply(pose), whole_structure, False, True
            )

    def cst_energy_filter(pose, selection, cutoff):
        score_pose = pose.clone()
        pyrosetta.rosetta.protocols.constraint_movers.ClearConstraintsMover().apply(
            score_pose
        )
        add_linker_csts(score_pose, selection)
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

    def filter_by_rmsd(pose, pose_in, selection, **kwargs):
        from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector as TrueResidueSelector
        from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector as ResidueIndexSelector
        from pyrosetta.rosetta.core.select.residue_selector import AndResidueSelector as AndResidueSelector
        from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector as NotResidueSelector

        if "type" not in kwargs.keys():
            kwargs["type"] = "rmsd_protein_bb_ca"
        if "sele" not in kwargs.keys():
            kwargs["sele"] = AndResidueSelector(
                TrueResidueSelector(), NotResidueSelector(selection)
            )
        else:
            kwargs["sele"] = AndResidueSelector(
                ResidueIndexSelector(kwargs["sele"]), NotResidueSelector(selection)
            )
        if "super_sele" not in kwargs.keys():
            kwargs["super_sele"] = kwargs["sele"]
        else:
            kwargs["super_sele"] = AndResidueSelector(
                ResidueIndexSelector(kwargs["super_sele"]),
                NotResidueSelector(selection),
            )
        if "save" not in kwargs.keys():
            kwargs["save"] = False
        rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
        rmsd_metric.set_residue_selector(kwargs["sele"])
        rmsd_metric.set_residue_selector_reference(kwargs["sele"])
        rmsd_metric.set_residue_selector_super(kwargs["super_sele"])
        rmsd_metric.set_residue_selector_super_reference(kwargs["super_sele"])
        rmsd_metric.set_rmsd_type(
            eval(f"pyrosetta.rosetta.core.scoring.rmsd_atoms.{kwargs['type']}")
        )
        rmsd_metric.set_run_superimpose(True)
        rmsd_metric.set_comparison_pose(pose_in)
        rmsd = rmsd_metric.calculate(pose)
        if kwargs["save"]:
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                pose, "crosslinking_rmsd", rmsd
            )
        if "cutoff" in kwargs.keys():
            return rmsd > kwargs["cutoff"]
        else:
            return False

    pose = pose_in.clone()
    if type(selection) == str:
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
    if filter_by_sidechain_distance > 0:
        assert not filter_by_sc_dist(
            pose, selection, filter_by_sidechain_distance
        ), f"CB of positions {res_indices[0]} and {res_indices[1]} are greater than {filter_by_sidechain_distance}A apart."

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
        add_linker_csts(pose, selection)

    # Pack linker and sidechains
    if sc_fast_relax_rounds > 0:
        pack_and_min(pose, selection, False, sc_fast_relax_rounds)

    # Filter by constraint energy
    if filter_by_cst_energy > 0:
        assert not cst_energy_filter(
            pose, selection, filter_by_cst_energy
        ), "Failed constraint energy filter after relaxing linker and sidechains"

    # Final fast relax
    if final_fast_relax_rounds > 0:
        pack_and_min(
            pose, selection, True, final_fast_relax_rounds, cartesian, custom_movemap
        )

    # Filter by constraint energy
    if filter_by_cst_energy > 0:
        assert not cst_energy_filter(
            pose, selection, filter_by_cst_energy
        ), "Failed constraint energy filter after final fast relax"

    # Filter by total score
    if filter_by_total_score is not None:
        assert not total_score_filter(pose, filter_by_total_score)

    # Filter by rmsd
    if rmsd_filter is not None:
        assert not filter_by_rmsd(
            pose, pose_in, selection, **rmsd_filter
        ), "Failed RMSD filter"

    return pose
