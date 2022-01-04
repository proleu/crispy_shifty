# Python standard library
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


class MPNNRunner(ABC):
    """
    Abstract base class for MPNN runners.
    TODO: dunder methods? probs not
    TODO: test update_flags method
    TODO: fasta mode
    """

    import os, pwd, uuid, shutil, subprocess
    import pyrosetta.distributed.io as io

    def __init__(
        self,
        batch_size: Optional[int] = 8,
        num_sequences: Optional[int] = 64,
        omit_AAs: Optional[str] = "X",
        temperature: Optional[float] = 0.1,
        selector: Optional[ResidueSelector] = None,
        chains_to_mask: Optional[List[str]] = None,
    ):
        """
        Initialize the base class for MPNN runners with common attributes.
        :param: batch_size: number of sequences to generate per batch.
        :param: num_sequences: number of sequences to generate in total.
        :param: omit_AAs: concatenated string of amino acids to omit from the sequence.
        :param: temperature: temperature to use for the MPNN.
        :param: selector: ResidueSelector that specifies residues to design.
        :param: chains_to_mask: list of chains to mask, these will be designable.
        If no `chains_to_mask` is provided, the runner will run on (mask) all chains.
        If a `chains_to_mask` is provided, the runner will run on (mask) only that chain.
        If no `selector` is provided, all residues on all masked chains will be designed.
        The chain letters in your PDB must be correct. TODO, might want to add a check for this.
        """
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.omit_AAs = omit_AAs
        self.temperature = temperature
        self.selector = selector
        self.chains_to_mask = chains_to_mask
        # setup standard command line flags for MPNN with default values
        self.flags = {
            "--backbone_noise": "0.05",
            "--checkpoint_path": "/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch51_step255000.pt",
            "--decoding_order": "random",
            "--hidden_dim": "192",
            "--max_length": "10000",
            "--num_connections": "64",
            "--num_layers": "3",
            "--protein_features": "full",
        }
        # add the flags that are required by MPNN and provided by the user
        self.flags.update(
            {
                "--batch_size": str(self.batch_size),
                "--num_seq_per_target": str(self.num_sequences),
                "--omit_AAs": self.omit_AAs,
                "--sampling_temp": str(self.temperature),
            }
        )
        # unset, needed: --jsonl_path --chain_id_jsonl --fixed_positions_jsonl --out_folder
        # unset, optional: --bias_AA_jsonl --omit_AA_jsonl --tied_positions_jsonl
        self.allowed_flags = [
            # flags that have default values that are provided by the runner:
            "--backbone_noise",
            "--checkpoint_path",
            "--decoding_order",
            "--hidden_dim",
            "--max_length",
            "--num_connections",
            "--num_layers",
            "--protein_features",
            # flags that are set by MPNNRunner constructor:
            "--batch_size",
            "--num_seq_per_target",
            "--omit_AAs",
            "--sampling_temp",
            # flags that are required and are set by MPNNRunner or children:
            "--jsonl_path",
            "--chain_id_jsonl",
            "--fixed_positions_jsonl",
            "--out_folder",
            # flags that are optional and are set by MPNNRunner or children:
            "--bias_AA_jsonl",
            "--omit_AA_jsonl",
            "--tied_positions_jsonl",
            "--pssm_bias_flag",
            "--pssm_jsonl",
            "--pssm_log_odds_flag",
            "--pssm_multi",
            "--pssm_threshold",
        ]
        # this updates to mpnn_run_tied.py if there is the --tied_positions_jsonl flag
        self.script = "/projects/crispy_shifty/mpnn/mpnn_run.py"
        self.tmpdir = None  # this will be updated by the setup_tmpdir method.
        self.is_setup = False  # this will be updated by the setup_runner method.

    def get_flags(self) -> Dict[str, str]:
        """
        :return: dictionary of flags.
        """
        return self.flags

    def get_script(self) -> str:
        """
        :return: script path.
        """
        return self.script

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the MPNNRunner.
        """
        import os, pwd, uuid

        if os.environ.get("TMPDIR") is not None:
            tmpdir_root = os.environ.get("TMPDIR")
        else:
            tmpdir_root = f"/net/scratch/{pwd.getpwuid(os.getuid()).pw_name}"

        self.tmpdir = os.path.join(tmpdir_root, uuid.uuid4().hex)
        os.makedirs(self.tmpdir, exist_ok=True)
        return

    def teardown_tmpdir(self) -> None:
        """
        :return: None
        Remove the temporary directory for the MPNNRunner.
        """
        import shutil

        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)
        return

    def update_flags(self, update_dict: Dict[str, str]) -> None:
        """
        :param: update_dict: dictionary of flags to update.
        :return: None
        Update the flags dictionary with the provided dictionary.
        Validate the flags before updating.
        """

        for flag in update_dict.keys():
            if flag not in self.allowed_flags:
                raise ValueError(
                    f"Flag {flag} is not allowed. Allowed flags are {self.allowed_flags}"
                )
        self.flags.update(update_dict)
        return

    def update_script(self) -> None:
        """
        :return: None
        Update the script path based on whether the --tied_positions_jsonl flag is set.
        """
        if "--tied_positions_jsonl" in self.flags.keys():
            self.script = "/projects/crispy_shifty/mpnn/mpnn_run_tied.py"
        else:
            self.script = "/projects/crispy_shifty/mpnn/mpnn_run.py"
        return

    def setup_runner(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Setup the MPNNRunner.
        Output sequences and scores are written temporarily to the tmpdir.
        They are then read in, and the sequences are appended to the pose datacache.
        The tmpdir is then removed.
        """
        import json, os, subprocess, sys
        import pyrosetta.distributed.io as io

        sys.path.insert(0, "/projects/crispy_shifty")
        from crispy_shifty.utils.io import cmd # TODO

        # setup the tmpdir
        self.setup_tmpdir()
        out_path = self.tmpdir
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(out_path, "tmp.pdb")
        pdbstring = io.to_pdbstring(pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        # make the jsonl file for the PDB biounits
        biounit_path = os.path.join(out_path, "biounits.jsonl")
        run_cmd = " ".join(
            [
                "/projects/crispy_shifty/mpnn/parse_multiple_chains.py",
                f"--pdb_folder {out_path}",
                f"--out_path {biounit_path}",
            ]
        )
        print(cmd(run_cmd))  # TODO can print this to debug
        # make a number to letter dictionary that starts at 1
        num_to_letter = {
            i: chr(i - 1 + ord("A")) for i in range(1, pose.num_chains() + 1)
        }
        # make the jsonl file for the chain_ids
        chain_id_path = os.path.join(out_path, "chain_id.jsonl")
        chain_dict = {}
        # make lists of masked and visible chains
        masked, visible = [], []
        # first make a list of all chain letters in the pose
        all_chains = [num_to_letter[i] for i in range(1, pose.num_chains() + 1)]
        # if chains_to_mask is provided, update the masked and visible lists
        if self.chains_to_mask is not None:
            # loop over the chains in the pose and add them to the appropriate list
            for chain in all_chains:
                if chain in self.chains_to_mask:
                    masked.append(i)
                else:
                    visible.append(i)
        else:
            # if chains_to_mask is not provided, mask all chains
            masked = all_chains
        chain_dict["tmp"] = [masked, visible]
        # write the chain_dict to a jsonl file
        with open(chain_id_path, "w") as f:
            f.write(
                json.dumps(chain_dict)
            )  # TODO can print this to debug, probably don't need newline
        # make the jsonl file for the fixed_positions
        fixed_positions_path = os.path.join(out_path, "fixed_positions.jsonl")
        fixed_positions_dict = {"tmp": {chain: [] for chain in all_chains}}
        # get a boolean mask of the residues in the selector
        if self.selector is not None:
            designable_filter = list(self.selector.apply(pose))
        else:  # if no selector is provided, make all residues designable
            designable_filter = [True] * pose.size()
        # check the residue selector specifies designability across the entire pose
        try:
            assert len(designable_filter) == pose.total_residue()
        except AssertionError:
            print(
                "Residue selector must specify designability for all residues.\n",
                f"Selector: {len(list(self.selector.apply(pose)))}\n",
                f"Pose: {pose.size()}",
            )
            raise
        # make a dict mapping of residue numbers to whether they are designable
        designable_dict = dict(zip(range(1, pose.size() + 1), designable_filter))
        # loop over the actual residues in the pose and add them to the fixed_positions_dict
        for i, residue in enumerate(pose.residues, start=1):
            residue_chain = num_to_letter[residue.chain()]
            # if the residue is in a chain that is not masked, it won't be designed anyway
            if residue_chain in visible:
                continue
            # if the residue is not designable, add it to the fixed_positions_dict
            elif not designable_dict[i]:
                fixed_positions_dict["tmp"][residue_chain].append(i)
            else:
                continue
        # write the fixed_positions_dict to a jsonl file
        with open(fixed_positions_path, "w") as f:
            f.write(
                json.dumps(fixed_positions_dict)
            )  # TODO can print this to debug, probably don't need newline
        # update the flags for the biounit, chain_id, and fixed_positions paths
        flag_update = {
            "--jsonl_path": biounit_path,
            "--chain_id_jsonl": chain_id_path,
            "--fixed_positions_jsonl": fixed_positions_path,
        }
        self.update_flags(flag_update)
        self.is_setup = True
        return

    @abstractmethod
    def apply(self) -> None:
        """
        This function needs to be implemented by the child class of MPNNRunner.
        """
        pass


class MPNNDesign(MPNNRunner):
    """
    Class for running MPNN on a single interface selection or chain.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initialize the base class for MPNN runners with common attributes.
        """
        super().__init__(*args, **kwargs)

    def apply(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Run MPNN on the provided pose.
        Setup the MPNNRunner using the provided pose.
        Run MPNN in a subprocess using the provided flags and tmpdir.
        Read in and parse the output fasta file to get the sequences.
        Each sequence designed by MPNN is then appended to the pose datacache.
        """
        import os, subprocess, sys
        import pyrosetta.distributed.io as io

        sys.path.insert(0, "/projects/crispy_shifty")
        from crispy_shifty.protocols.mpnn import fasta_to_dict, thread_full_sequence
        from crispy_shifty.utils.io import cmd

        # setup runner
        self.setup_runner(pose)
        self.update_flags({"--out_folder": self.tmpdir})
        self.update_script()

        # run mpnn by calling self.script and providing the flags
        run_cmd = (
            self.script + " " + " ".join([f"{k} {v}" for k, v in self.flags.items()])
        )
        out_err = cmd(run_cmd)
        print(run_cmd)
        print(out_err)
        # TODO can print this to debug
        alignments_path = os.path.join(self.tmpdir, "alignments/tmp.fa")
        # parse the alignments fasta into a dictionary
        alignments = fasta_to_dict(alignments_path)
        for i, (tag, seq) in enumerate(alignments.items()):
            index = str(i).zfill(4)
            # print(f"{i}: {seq}") TODO
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, f"mpnn_seq_{index}", seq)
        # clean up the temporary files
        self.teardown_tmpdir()
        return

    def dump_fasta(self, pose: Pose, out_path: str) -> None:
        """
        :param: pose: Pose object that contains the designed sequences.
        :param: out_path: Path to write the fasta file to.
        :return: None
        Dump the pose mpnn_seq_* sequences to a single fasta.
        """
        import sys

        sys.path.insert(0, "/projects/crispy_shifty")
        from crispy_shifty.protocols.mpnn import dict_to_fasta 

        # get the mpnn_seq_* sequences from the pose
        seqs_to_write = {tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag}
        # write the sequences to a fasta
        dict_to_fasta(seqs_to_write, out_path)

        return

    def generate_poses(self, pose: Pose) -> Iterator[Pose]:
        """
        :param: pose: Pose object to generate poses from.
        :return: Iterator of Pose objects.
        Generate poses from the provided pose with the newly designed sequences.
        """
        import sys

        sys.path.insert(0, "/projects/crispy_shifty")
        from crispy_shifty.protocols.mpnn import thread_full_sequence

        # get the mpnn_seq_* sequences from the pose
        seqs_dict = {tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag}
        # generate the poses from the seqs_dict
        for _, seq in seqs_dict.items():
            # print(f"{tag}: {seq}")
            # thread the full sequence
            threaded_pose = thread_full_sequence(pose, seq)
            yield threaded_pose

    
def dict_to_fasta(
    seqs_dict: Dict[str, str],
    out_path: str,
) -> None:
    """
    :param seqs_dict: dictionary of sequences to write to a fasta file.
    :param out_path: path to write the fasta file to.
    :return: None
    Write a fasta file to the provided path with the provided sequence dict.
    """
    import os

    # make the output path if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        pass
    # write the sequences to a fasta file
    with open(out_path, "w") as f:
        for i, seq in seqs_dict.items():
            f.write(f">{i}\n{seq}\n")
    return

def fasta_to_dict(fasta:str) -> Dict[str, str]:
    """
    :param fasta: fasta filepath to read from
    :return: dictionary of tags and sequences
    Read in a fasta file and return a dictionary of tags and sequences.
    """
    seqs_dict = {}

    with open(fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                tag = line.strip().split()[0][1:]
                seq = ""
            else:
                seq += line.strip()
            seqs_dict[tag] = seq

    return seqs_dict


def thread_full_sequence(
    pose: Pose,
    sequence: str,
    start_res: int = 1,
) -> Pose:
    """
    :param: pose: Pose to thread sequence onto.
    :param: sequence: Sequence to thread onto pose.
    :return: Pose with threaded sequence.
    Threads a full sequence onto a pose after cloning the input pose.
    Doesn't require the chainbreak '/' to be cleaned.
    """
    from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover

    try:
        assert sequence.count("/") == 0
    except AssertionError:
        print("Cleaning chain breaks in sequence.")
        sequence = sequence.replace("/", "")

    pose = pose.clone()
    stm = SimpleThreadingMover()
    stm.set_sequence(sequence, start_res)
    stm.apply(pose)

    return pose


@requires_init
def mpnn_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNDesign.
    :return: an iterator of PackedPose objects.
    TODO: Assumes that pyrosetta.init() has been called with `-corrections:beta_nov16` ?
    """

    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

    sys.path.insert(0, "/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
    from crispy_shifty.protocols.mpnn import MPNNDesign
    from crispy_shifty.utils.io import print_timestamp

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

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up design selector", start_time)
        # make a designable residue selector of only the interface residues
        chain_a = ChainSelector(1)
        chain_b = ChainSelector(2)
        interface_selector = interface_between_selectors(chain_a, chain_b)
        print_timestamp("Designing interface with MPNN", start_time)
        # construct the MPNNDesign object
        mpnn_design = MPNNDesign(
            selector=interface_selector,
            # TODO **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose
