import glob
import os
import shutil

def filename(path):
    return os.path.splitext(os.path.basename(path))[0]

import pyrosetta
pyrosetta.init('-out:mute all -ex1 -ex2aro')

from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector
from pyrosetta.rosetta.core.select.residue_selector import AndResidueSelector
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.core.select.residue_selector import SecondaryStructureSelector
from pyrosetta.rosetta.core.select.residue_selector import ResiduePDBInfoHasLabelSelector
from pyrosetta.rosetta.core.select.residue_selector import PrimarySequenceNeighborhoodSelector
from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector
from pyrosetta.rosetta.core.select.residue_selector import ResidueSpanSelector

from pyrosetta.rosetta.protocols.symmetry import DetectSymmetry
from pyrosetta.rosetta.protocols.symmetry import SetupForSymmetryMover

from atom.the_stapler import TheNativeDisulfideStapler

# Preset ResidueSelectors
default_residue_selectors = [TrueResidueSelector(), TrueResidueSelector()]
interface_residue_selectors = [ChainSelector('A'), ChainSelector('B')]
interface_or_internal_residue_selectors = [ChainSelector('A'), ChainSelector('A,B')]
only_binder_residue_selectors = [ChainSelector('B'), ChainSelector('B')]
not_on_loops = [SecondaryStructureSelector('HE'), SecondaryStructureSelector('HE')]
not_on_loops_across_interface = [AndResidueSelector(SecondaryStructureSelector('HE'),ChainSelector('A')),
                                 AndResidueSelector(SecondaryStructureSelector('HE'),ChainSelector('B'))]
# FloP residue selectors
new_loop = ResiduePDBInfoHasLabelSelector("new_loop")
before_loop = PrimarySequenceNeighborhoodSelector(500,0,new_loop)
after_loop = PrimarySequenceNeighborhoodSelector(0,500,new_loop)
between_parts = [AndResidueSelector(before_loop, NotResidueSelector(new_loop)), AndResidueSelector(after_loop, NotResidueSelector(new_loop))]
#between_parts = [AndResidueSelector(ChainSelector('A'), ResidueSpanSelector(46,94)), AndResidueSelector(ChainSelector('A'), ResidueSpanSelector(97,140))]
objs_sel = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        '''
        <RESIDUE_SELECTORS>
            <SSElement name="part1" selection="n_term" to_selection="4,H,E" chain="A" reassign_short_terminal_loop="2" />
            <SSElement name="part2" selection="5,H,S" to_selection="c_term" chain="A" reassign_short_terminal_loop="2" />
        </RESIDUE_SELECTORS>
        ''')
part1 = objs_sel.get_residue_selector('part1')
part2 = objs_sel.get_residue_selector('part2')
by_helix = [part1,part2]
# Initialize the native disulfide stapler with defaults.
the_native_disulfide_stapler = TheNativeDisulfideStapler(score_function='ref2015_cart',
                                                         residue_selectors=by_helix,
                                                         disulfide_energy_cutoff=-0.8, flexible_neighborhood_distance=10.0,
                                                         scanning_sequence_distance_minimum=4, coordinate_constraint_coefficient=None)
# NOTE: If you are not on the Digs cluster, you will need to change the path to the database file.
#       The syntax for that is: hashtable_file_name=/path/to/file/default_native_disulfide_full_10_15_512.hashtable.gz
# NOTE: Add coordinate constraint coefficient ~2.0-10.0 if you are running on a pose that is not designed.
# NOTE: ~90% of all native disulfides have an energy of less than -0.8 given the standard 1.25 weight (Ref2015).
# NOTE: Various Preset ResidueSelector pairs have been provided for common use cases.

pdbs_to_staple = sorted(
    # glob.glob('/home/broerman/projects/CSD/round_2/af2/af2_models/*A_*.pdb') + \
    glob.glob('/home/broerman/projects/CSD/round_2/af2/af2_models/*[!A]B_*.pdb') + \
    glob.glob('/home/broerman/projects/CSD/round_2/af2/split_Y/*.pdb')
)

for pdb in pdbs_to_staple:
    pdb_filename = filename(pdb)

    pose = pyrosetta.pose_from_file(pdb)

    # Preset Movers for Symmetry
    # DetectSymmetry().apply(pose)
    # SetupForSymmetryMover('C2.sym').apply(pose)

    for i, crosslinked_pose in enumerate(the_native_disulfide_stapler.apply(pose)):
        crosslinked_pose.dump_pdb(os.path.join('/home/broerman/projects/CSD/round_2/disulfides/staples', f'{pdb_filename}_staple_{i}.pdb'))
