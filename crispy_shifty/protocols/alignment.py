# 3rd party library imports
# Rosetta library imports
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

def score_rmsd(
    pose: Pose, refpose: Pose, sel: ResidueSelector = None, refsel: ResidueSelector = None, 
    rmsd_type: pyrosetta.rosetta.core.scoring.rmsd_atoms = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca
):
    # Adam Broerman
    rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
    rmsd_metric.set_comparison_pose(refpose)
    if sel == None:
        sel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    rmsd_metric.set_residue_selector(sel)
    if refsel == None:
        refsel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    rmsd_metric.set_residue_selector_reference(refsel)
    rmsd_metric.set_rmsd_type(rmsd_type) # Default is rmsd_all_heavy
    rmsd_metric.set_run_superimpose(True)
    rmsd = rmsd_metric.calculate(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(rmsd))
    return rmsd