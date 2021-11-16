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
    except RuntimeError:
        closure_type = 'not_closed'
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, "closure_type", closure_type
    )
    return closure_type


def strict_remodel(pose: Pose, length: int):
    """
    Remodel a new loop using Blueprint Builder. Expects a pose with two chains.
    DSSP and SS agnostic in principle but in practice more or less matches.
    """
    import os
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.packed_pose.core import PackedPose
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )
    from pyrosetta.rosetta.core.select import get_residues_from_subset
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

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
        return "X"

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

    def strict_remodel_helper(pose: Pose, loop_length: int) -> str:
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
        with open(filename, "w+") as f:
            end1, begin2 = (
                pose.chain_end(1),
                pose.chain_begin(2),
            )
            end2 = pose.chain_end(2)
            for i in range(1, end1 + 1):
                if i == end1:
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
                if i == begin2:
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

    def find_vv(seq):
        indices = []
        seq_minus_one = seq[:-1]
        for i, char in enumerate(seq_minus_one):
            if (char == seq[i + 1]) and (char == "V"):
                indices.append(i + 1)
            else:
                pass
        # rosetta sequence indexes begin at 1
        true_indices = [str(x + 1) for x in indices]
        return true_indices

    def append_b_to_a(pose_a: Pose, pose_b: Pose, end_a: int, start_b: int) -> Pose:
        """
        Make a new pose, containing pose_a up to end_a, then pose_b starting from start_b
        Assumes pose_a has only one chain.
        """
        import pyrosetta
        from pyrosetta.rosetta.core.pose import Pose

        newpose = Pose()
        for i in range(1, end_a + 1):
            newpose.append_residue_by_bond(pose_a.residue(i))
        newpose.append_residue_by_jump(
            pose_b.residue(start_b), newpose.chain_end(1), "CA", "CA", 1
        )
        for i in range(start_b + 1, len(pose_b.residues) + 1):
            newpose.append_residue_by_bond(pose_b.residue(i))
        return newpose

    bp = strict_remodel_helper(pose, length)

    bp_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0)

    xml = """
    <ROSETTASCRIPTS>
        <SCOREFXNS>
            <ScoreFunction name="sfxn1" weights="fldsgn_cen">
                <Reweight scoretype="hbond_sr_bb" weight="1.0" />
                <Reweight scoretype="hbond_lr_bb" weight="1.0" />
                <Reweight scoretype="atom_pair_constraint" weight="1.0" />
                <Reweight scoretype="angle_constraint" weight="1.0" />
                <Reweight scoretype="dihedral_constraint" weight="1.0" />
            </ScoreFunction>
        </SCOREFXNS>
        <RESIDUE_SELECTORS>          
        </RESIDUE_SELECTORS>
        <TASKOPERATIONS>
        </TASKOPERATIONS>
        <SIMPLE_METRICS>
        </SIMPLE_METRICS>
        <MOVERS>
            <BluePrintBDR name="bdr" 
            blueprint="{bp}" 
            use_abego_bias="0" 
            use_sequence_bias="0" 
            rmdl_attempts="20"
            scorefxn="sfxn1"/>
        </MOVERS>
        <FILTERS>
        </FILTERS>
        <PROTOCOLS>
            <Add mover_name="bdr"/>
        </PROTOCOLS>
    </ROSETTASCRIPTS>
    """.format(
        bp=bp
    )
    strict_remodel = SingleoutputRosettaScriptsTask(xml)
    maybe_closed_ppose = None
    for i in range(10):
        print(f"attempt: {i}")
        if maybe_closed_ppose is not None:  # check if it worked
            break  # stop retrying if it did
        else:  # try again if it didn't. returns None if fail
            maybe_closed_ppose = strict_remodel(packed_pose_in.pose.clone())
    os.remove(bp)  # cleanup tree
    if maybe_closed_ppose is not None:
        closure_type = "strict_remodel"
        # hacky rechain
        maybe_closed_pose = append_b_to_a(
            pose_a=maybe_closed_ppose.pose.clone(),
            pose_b=packed_pose_in.pose.clone(),
            end_a=packed_pose_in.pose.chain_end(2),
            start_b=packed_pose_in.pose.chain_begin(3),
        )
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                maybe_closed_pose, key, str(value)
            )
        # update closure_type
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            maybe_closed_pose, "closure_type", closure_type
        )

    else:  # return the original input if BlueprintBDR still didn't close
        maybe_closed_pose = packed_pose_in.pose.clone()


@requires_init
def loop_match(packed_pose_in: PackedPose, **kwargs) -> PackedPose:
    """
    Match loop length, total length and DSSP with parent. Strictest method of closure.
    """
    import bz2
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )

    def append_b_to_a(pose_a: Pose, pose_b: Pose, end_a: int, start_b: int) -> Pose:
        """
        Make a new pose, containing pose_a up to end_a, then pose_b starting from start_b
        Assumes pose_a has only one chain.
        """
        import pyrosetta
        from pyrosetta.rosetta.core.pose import Pose

        newpose = Pose()
        for i in range(1, end_a + 1):
            newpose.append_residue_by_bond(pose_a.residue(i))
        newpose.append_residue_by_jump(
            pose_b.residue(start_b), newpose.chain_end(1), "CA", "CA", 1
        )
        for i in range(start_b + 1, len(pose_b.residues) + 1):
            newpose.append_residue_by_bond(pose_b.residue(i))
        return newpose

    if packed_pose_in == None:
        file = kwargs["-s"]
        with open(file, "rb") as f:
            packed_pose_in = io.pose_from_pdbstring(bz2.decompress(f.read()).decode())
        scores = pyrosetta.distributed.cluster.get_scores_dict(file)["scores"]
    else:
        raise RuntimeError("Need to supply an input")
    # get parent from packed_pose_in, get loop length from parent length - packed_pose_in length
    parent_length = int(scores["parent_length"])
    length = int(parent_length - packed_pose_in.pose.chain_end(2))
    xml = """
    <ROSETTASCRIPTS>
        <SCOREFXNS>
            <ScoreFunction name="sfxn" weights="beta_nov16" /> 
        </SCOREFXNS>
        <RESIDUE_SELECTORS>          
        </RESIDUE_SELECTORS>
        <TASKOPERATIONS>
        </TASKOPERATIONS>
        <SIMPLE_METRICS>
        </SIMPLE_METRICS>
        <MOVERS>
            <ConnectChainsMover name="closer" 
                chain_connections="[A+B]" 
                loopLengthRange="{length},{length}" 
                resAdjustmentRangeSide1="0,0" 
                resAdjustmentRangeSide2="0,0" 
                RMSthreshold="1.0"/>
        </MOVERS>
        <FILTERS>
        </FILTERS>
        <PROTOCOLS>
            <Add mover_name="closer"/>
        </PROTOCOLS>
    </ROSETTASCRIPTS>
    """.format(
        length=length
    )
    closer = SingleoutputRosettaScriptsTask(xml)
    try:
        maybe_closed_ppose = closer(packed_pose_in.pose.clone())
        # hacky rechain
        maybe_closed_pose = append_b_to_a(
            pose_a=maybe_closed_ppose.pose.clone(),
            pose_b=packed_pose_in.pose.clone(),
            end_a=packed_pose_in.pose.chain_end(2),
            start_b=packed_pose_in.pose.chain_begin(3),
        )
        closure_type = "loop_match"
    except RuntimeError:
        maybe_closed_pose = io.to_pose(packed_pose_in.pose.clone())
        closure_type = "not_closed"
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        maybe_closed_pose, "closure_type", closure_type
    )
    for key, value in scores.items():
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            maybe_closed_pose, key, str(value)
        )
    final_ppose = io.to_packed(maybe_closed_pose)
    return final_ppose


def strict_remodel(packed_pose_in: PackedPose, **kwargs) -> PackedPose:
    """
    DSSP and SS agnostic in principle but in practice more or less matches.
    """
    import os
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.packed_pose.core import PackedPose
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )
    from pyrosetta.rosetta.core.select import get_residues_from_subset
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

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
        return "X"

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

    def strict_remodel_helper(packed_pose_in: PackedPose, loop_length: int) -> str:
        import binascii, os
        import pyrosetta
        from pyrosetta.rosetta.core.pose import Pose

        pose = packed_pose_in.pose.clone()
        tors = get_torsions(pose)
        abego_str = abego_string(tors)
        dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(pose)
        # name blueprint a random 32 long hex string
        filename = str(binascii.b2a_hex(os.urandom(16)).decode("utf-8")) + ".bp"
        # write a temporary blueprint file
        with open(filename, "w+") as f:
            end1, begin2 = (
                packed_pose_in.pose.chain_end(1),
                packed_pose_in.pose.chain_begin(2),
            )
            end3 = packed_pose_in.pose.chain_end(3)
            for i in range(1, end1 + 1):
                if i == end1:
                    print(
                        str(i),
                        packed_pose_in.pose.residue(i).name1(),
                        dssp[i - 1] + "X",
                        "R",
                        file=f,
                    )
                else:
                    print(
                        str(i),
                        packed_pose_in.pose.residue(i).name1(),
                        dssp[i - 1] + abego_str[i - 1],
                        ".",
                        file=f,
                    )
            for i in range(loop_length):
                print(
                    "0", "V", "LX", "R", file=f
                )  # DX is bad, causes rare error sometimes
            for i in range(begin2, end3 + 1):
                if i == begin2:
                    print(
                        str(i),
                        packed_pose_in.pose.residue(i).name1(),
                        dssp[i - 1] + "X",
                        "R",
                        file=f,
                    )
                else:
                    print(
                        str(i),
                        packed_pose_in.pose.residue(i).name1(),
                        dssp[i - 1] + abego_str[i - 1],
                        ".",
                        file=f,
                    )

        return filename

    def find_vv(seq):
        indices = []
        seq_minus_one = seq[:-1]
        for i, char in enumerate(seq_minus_one):
            if (char == seq[i + 1]) and (char == "V"):
                indices.append(i + 1)
            else:
                pass
        # rosetta sequence indexes begin at 1
        true_indices = [str(x + 1) for x in indices]
        return true_indices

    def append_b_to_a(pose_a: Pose, pose_b: Pose, end_a: int, start_b: int) -> Pose:
        """
        Make a new pose, containing pose_a up to end_a, then pose_b starting from start_b
        Assumes pose_a has only one chain.
        """
        import pyrosetta
        from pyrosetta.rosetta.core.pose import Pose

        newpose = Pose()
        for i in range(1, end_a + 1):
            newpose.append_residue_by_bond(pose_a.residue(i))
        newpose.append_residue_by_jump(
            pose_b.residue(start_b), newpose.chain_end(1), "CA", "CA", 1
        )
        for i in range(start_b + 1, len(pose_b.residues) + 1):
            newpose.append_residue_by_bond(pose_b.residue(i))
        return newpose

    # ensure pose still needs to be closed, skip to scoring and labeling if it has
    if packed_pose_in.pose.num_chains() == 2:
        maybe_closed_pose = packed_pose_in.pose.clone()
    else:
        scores = packed_pose_in.pose.scores
        # get parent from packed_pose_in, get loop length from parent length - packed_pose_in length
        parent_length = int(scores["parent_length"])
        length = int(parent_length - packed_pose_in.pose.chain_end(2))
        bp = strict_remodel_helper(packed_pose_in, length)
        xml = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn1" weights="fldsgn_cen">
                    <Reweight scoretype="hbond_sr_bb" weight="1.0" />
                    <Reweight scoretype="hbond_lr_bb" weight="1.0" />
                    <Reweight scoretype="atom_pair_constraint" weight="1.0" />
                    <Reweight scoretype="angle_constraint" weight="1.0" />
                    <Reweight scoretype="dihedral_constraint" weight="1.0" />
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>          
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
            </TASKOPERATIONS>
            <SIMPLE_METRICS>
            </SIMPLE_METRICS>
            <MOVERS>
                <BluePrintBDR name="bdr" 
                blueprint="{bp}" 
                use_abego_bias="0" 
                use_sequence_bias="0" 
                rmdl_attempts="20"
                scorefxn="sfxn1"/>
            </MOVERS>
            <FILTERS>
            </FILTERS>
            <PROTOCOLS>
                <Add mover_name="bdr"/>
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            bp=bp
        )
        strict_remodel = SingleoutputRosettaScriptsTask(xml)
        maybe_closed_ppose = None
        for i in range(10):
            print(f"attempt: {i}")
            if maybe_closed_ppose is not None:  # check if it worked
                break  # stop retrying if it did
            else:  # try again if it didn't. returns None if fail
                maybe_closed_ppose = strict_remodel(packed_pose_in.pose.clone())
        os.remove(bp)  # cleanup tree
        if maybe_closed_ppose is not None:
            closure_type = "strict_remodel"
            # hacky rechain
            maybe_closed_pose = append_b_to_a(
                pose_a=maybe_closed_ppose.pose.clone(),
                pose_b=packed_pose_in.pose.clone(),
                end_a=packed_pose_in.pose.chain_end(2),
                start_b=packed_pose_in.pose.chain_begin(3),
            )
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(
                    maybe_closed_pose, key, str(value)
                )
            # update closure_type
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                maybe_closed_pose, "closure_type", closure_type
            )

        else:  # return the original input if BlueprintBDR still didn't close

            maybe_closed_pose = packed_pose_in.pose.clone()

    # switch scores to new score dict
    scores = maybe_closed_pose.scores
    # ensure pose has been closed, if not don't label new loop
    if maybe_closed_pose.num_chains() != 2:
        new_loop_str = "0,0"
        labeled_pose = maybe_closed_pose.clone()
    else:
        seq = str(maybe_closed_pose.sequence())
        vv_indices = ",".join(find_vv(seq))
        pre_break_helix = int(scores["pre_break_helix"])
        # get helix indices for the pre and post break helices. assumes middle loop of chA is the new one
        lower = pre_break_helix
        xml = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SSElement name="middle" selection="{pre},H,S" to_selection="-{post},H,E" chain="A" reassign_short_terminal_loop="2" />       
                <Index name="polyval_all" resnums="{vv_indices}" />
                <And name="polyval" selectors="middle,polyval_all" />
                <PrimarySequenceNeighborhood name="entire_val" selector="polyval" lower="5" upper="5" />
                <SecondaryStructure name="loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
                <And name="new_loop_center" selectors="entire_val,loop" />
                <PrimarySequenceNeighborhood name="entire_new_loop_broad" selector="new_loop_center" lower="5" upper="5" />
                <ResidueName name="isval" residue_name3="VAL" />
                <And name="entire_new_loop" selectors="entire_new_loop_broad,isval" />
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
            </TASKOPERATIONS>
            <SIMPLE_METRICS>
            </SIMPLE_METRICS>
            <MOVERS>
                <SwitchChainOrder name="rechain" chain_order="12"/>
            </MOVERS>
            <FILTERS>
            </FILTERS>
            <PROTOCOLS>
                <Add mover="rechain" />
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            pre=pre_break_helix, post=pre_break_helix, vv_indices=vv_indices
        )
        labeled = SingleoutputRosettaScriptsTask(xml)
        xml_obj = XmlObjects.create_from_string(xml)
        entire_new_loop_sel = xml_obj.get_residue_selector("entire_new_loop")
        labeled_ppose = labeled(maybe_closed_pose.clone())
        labeled_pose = io.to_pose(labeled_ppose)
        new_loop_resis = list(
            get_residues_from_subset(entire_new_loop_sel.apply(labeled_pose))
        )
        new_loop_str = ",".join(str(resi) for resi in new_loop_resis)

    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        labeled_pose, "new_loop_resis", new_loop_str
    )
    total_length = len(labeled_pose.residues)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        labeled_pose, "total_length", total_length
    )
    dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(labeled_pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(labeled_pose, "dssp", dssp)
    tors = get_torsions(labeled_pose)
    abego_str = abego_string(tors)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(labeled_pose, "abego_str", abego_str)
    for key, value in scores.items():
        pyrosetta.rosetta.core.pose.setPoseExtraScore(labeled_pose, key, str(value))
    final_ppose = io.to_packed(labeled_pose.clone())

    return final_ppose