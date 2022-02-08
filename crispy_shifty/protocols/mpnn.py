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
    from pathlib import Path

    # make the output path if it doesn't exist
    if not os.path.exists(Path(out_path).parent):
        os.makedirs(Path(out_path).parent)
    else:
        pass
    # write the sequences to a fasta file
    with open(out_path, "w") as f:
        for i, seq in seqs_dict.items():
            f.write(f">{i}\n{seq}\n")
    return


def fasta_to_dict(fasta: str, new_tags: bool = False) -> Dict[str, str]:
    """
    :param fasta: fasta filepath to read from.
    :param new_tags: if False, use the sequence tag as the key, if True, use the index.
    :return: dictionary of tags and sequences.
    Read in a fasta file and return a dictionary of tags and sequences.
    """
    seqs_dict = {}

    with open(fasta, "r") as f:
        i = 0
        for line in f:
            if line.startswith(">"):  # this is a header line
                if new_tags:
                    tag = str(i)
                else:
                    tag = line.strip().replace(">", "")
                seqs_dict[tag] = ""
                i += 1
            else:  # we can assume this is a sequence line, add the sequence
                seqs_dict[tag] += line.strip()

    return seqs_dict


@requires_init
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


class MPNNRunner(ABC):
    """
    Abstract base class for MPNN runners.
    """

    import os, pwd, uuid, shutil, subprocess
    import pyrosetta.distributed.io as io

    def __init__(
        self,
        batch_size: Optional[int] = 8,
        num_sequences: Optional[int] = 64,
        omit_AAs: Optional[str] = "X",
        temperature: Optional[float] = 0.1,
        design_selector: Optional[ResidueSelector] = None,
        chains_to_mask: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the base class for MPNN runners with common attributes.
        :param: batch_size: number of sequences to generate per batch.
        :param: num_sequences: number of sequences to generate in total.
        :param: omit_AAs: concatenated string of amino acids to omit from the sequence.
        :param: temperature: temperature to use for the MPNN.
        :param: design_selector: ResidueSelector that specifies residues to design.
        :param: chains_to_mask: list of chains to mask, these will be designable.
        If no `chains_to_mask` is provided, the runner will run on (mask) all chains.
        If a `chains_to_mask` is provided, the runner will run on (mask) only that chain.
        If no `design_selector` is provided, all residues on all masked chains will be designed.
        The chain letters in your PDB must be correct.
        """

        from pathlib import Path

        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.omit_AAs = omit_AAs
        self.temperature = temperature
        self.design_selector = design_selector
        self.chains_to_mask = chains_to_mask
        # setup standard command line flags for MPNN with default values
        self.flags = {
            "--backbone_noise": "0.0",
            "--checkpoint_path": "/home/justas/projects/lab_github/mpnn/model_weigths/p17/epoch150_step235456.pt",  # TODO
            "--hidden_dim": "128",
            "--max_length": "20000",
            "--num_connections": "64",
            "--num_layers": "3",
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
        self.allowed_flags = [
            # flags that have default values that are provided by the runner:
            "--backbone_noise",
            "--checkpoint_path",
            "--hidden_dim",
            "--max_length",
            "--num_connections",
            "--num_layers",
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
            "--pdb_path",
            "--pdb_path_chains",
            "--pssm_bias_flag",
            "--pssm_jsonl",
            "--pssm_log_odds_flag",
            "--pssm_multi",
            "--pssm_threshold",
            "--save_score",
        ]
        # TODO git root?
        self.script = f"{str(Path(__file__).resolve().parent.parent.parent / 'proteinmpnn' / 'protein_mpnn_run.py')}"
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
        Create a temporary directory for the MPNNRunner. Checks for various best
        practice locations for the tmpdir in the following order: TMPDIR, PSCRATCH,
        CSCRATCH, /net/scratch. Uses the cwd if none of these are available.
        """
        import os, pwd, uuid

        if os.environ.get("TMPDIR") is not None:
            tmpdir_root = os.environ.get("TMPDIR")
        elif os.environ.get("PSCRATCH") is not None:
            tmpdir_root = os.environ.get("PSCRATCH")
        elif os.environ.get("CSCRATCH") is not None:
            tmpdir_root = os.environ.get("CSCRATCH")
        elif os.path.exists("/net/scratch"):
            tmpdir_root = f"/net/scratch/{pwd.getpwuid(os.getuid()).pw_name}"
        else:
            tmpdir_root = os.getcwd()

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

    def setup_runner(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Setup the MPNNRunner. Make a tmpdir and write input files to the tmpdir.
        Output sequences and scores will be written temporarily to the tmpdir as well.
        """
        import json, os, subprocess, sys
        import git
        from pathlib import Path
        import pyrosetta
        import pyrosetta.distributed.io as io

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.utils.io import cmd

        # setup the tmpdir
        self.setup_tmpdir()
        # check that the first residue is on chain A, if not we should try to fix it
        if pose.pdb_info().chain(1) != "A":
            # ensure the chain numbers are correct by using SwitchChainOrderMover
            sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
            sc.chain_order("".join([str(c) for c in range(1, pose.num_chains() + 1)]))
            sc.apply(pose)
        else:
            pass
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(self.tmpdir, "tmp.pdb")
        pdbstring = io.to_pdbstring(pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        # make the jsonl file for the PDB biounits
        biounit_path = os.path.join(self.tmpdir, "biounits.jsonl")
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        python = str(Path(root) / "envs" / "crispy" / "bin" / "python")
        if os.path.exists(python):
            pass
        else:  # crispy env must be installed in envs/crispy or must be used on DIGS
            python = "/projects/crispy_shifty/envs/crispy/bin/python"
        run_cmd = " ".join(
            [
                f"{python} {str(Path(__file__).resolve().parent.parent.parent / 'proteinmpnn' / 'helper_scripts' / 'parse_multiple_chains.py')}",
                f"--input_path {self.tmpdir}",
                f"--output_path {biounit_path}",
            ]
        )
        cmd(run_cmd)
        # make a number to letter dictionary that starts at 1
        num_to_letter = {
            i: chr(i - 1 + ord("A")) for i in range(1, pose.num_chains() + 1)
        }
        # make the jsonl file for the chain_ids
        chain_id_path = os.path.join(self.tmpdir, "chain_id.jsonl")
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
            f.write(json.dumps(chain_dict))
        # make the jsonl file for the fixed_positions
        fixed_positions_path = os.path.join(self.tmpdir, "fixed_positions.jsonl")
        fixed_positions_dict = {"tmp": {chain: [] for chain in all_chains}}
        # get a boolean mask of the residues in the design_selector
        if self.design_selector is not None:
            designable_filter = list(self.design_selector.apply(pose))
        else:  # if no design_selector is provided, make all residues designable
            designable_filter = [True] * pose.size()
        # check the residue design_selector specifies designability across the entire pose
        try:
            assert len(designable_filter) == pose.total_residue()
        except AssertionError:
            print(
                "Residue selector must specify designability for all residues.\n",
                f"Selector: {len(list(self.design_selector.apply(pose)))}\n",
                f"Pose: {pose.size()}",
            )
            raise
        # make a dict mapping of residue numbers to whether they are designable
        designable_dict = dict(zip(range(1, pose.size() + 1), designable_filter))
        # loop over the chains and the residues in the pose
        i = 1  # total residue counter
        for chain_number, chain in enumerate(all_chains, start=1):
            j = 1  # chain residue counter
            for res in range(
                pose.chain_begin(chain_number), pose.chain_end(chain_number) + 1
            ):
                # if the residue is on a masked chain but not designable, add it
                if not designable_dict[i] and chain in masked:
                    fixed_positions_dict["tmp"][chain].append(j)
                else:
                    pass
                j += 1
                i += 1
        # write the fixed_positions_dict to a jsonl file
        with open(fixed_positions_path, "w") as f:
            f.write(json.dumps(fixed_positions_dict))
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
    TODO either in here or base class determine how to go from task factory to omit_AA_jsonl
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNRunner.
        :param: kwargs: keyword arguments to pass to MPNNRunner.
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
        Cleanup the tmpdir.
        """
        import os, subprocess, sys
        import git
        from pathlib import Path
        import pyrosetta
        from pyrosetta.rosetta.core.pose import setPoseExtraScore
        import pyrosetta.distributed.io as io

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import fasta_to_dict, thread_full_sequence
        from crispy_shifty.utils.io import cmd

        # setup runner
        self.setup_runner(pose)
        self.update_flags({"--out_folder": self.tmpdir})

        # run mpnn by calling self.script and providing the flags
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        python = str(Path(root) / "envs" / "crispy" / "bin" / "python")
        if os.path.exists(python):
            pass
        else:  # crispy env must be installed in envs/crispy or must be used on DIGS
            python = "/projects/crispy_shifty/envs/crispy/bin/python"
        run_cmd = (
            f"{python} {self.script}"
            + " "
            + " ".join([f"{k} {v}" for k, v in self.flags.items()])
        )
        out_err = cmd(run_cmd)
        print(out_err)
        alignments_path = os.path.join(self.tmpdir, "seqs/tmp.fa")
        # parse the alignments fasta into a dictionary
        alignments = fasta_to_dict(alignments_path, new_tags=True)
        for i, (tag, seq) in enumerate(alignments.items()):
            index = str(i).zfill(4)
            setPoseExtraScore(pose, f"mpnn_seq_{index}", seq)
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
        from pathlib import Path
        import sys

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import dict_to_fasta

        # get the mpnn_seq_* sequences from the pose
        seqs_to_write = {
            tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag
        }
        # write the sequences to a fasta
        dict_to_fasta(seqs_to_write, out_path)

        return

    def generate_all_poses(
        self, pose: Pose, include_native: bool = False
    ) -> Iterator[Pose]:
        """
        :param: pose: Pose object to generate poses from.
        :param: include_native: Whether to generate the original native sequence.
        :return: Iterator of Pose objects.
        Generate poses from the provided pose with the newly designed sequences.
        Maintain the scores of the provided pose in the new poses.
        """
        from pathlib import Path
        import sys

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import thread_full_sequence

        # get the mpnn_seq_* sequences from the pose
        seqs_dict = {tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag}
        # get the non-mpnn_seq_* scores from the pose
        scores_dict = {
            key: val for key, val in pose.scores.items() if "mpnn_seq_" not in key
        }
        # generate the poses from the seqs_dict
        for tag, seq in seqs_dict.items():
            if include_native:
                pass
            else:  # don't include the original pose
                if tag == "mpnn_seq_0000":
                    continue
                else:
                    pass
            # thread the full sequence
            threaded_pose = thread_full_sequence(pose, seq)
            # set the scores
            for key, val in scores_dict.items():
                setPoseExtraScore(threaded_pose, key, val)
            yield threaded_pose


class MPNNMultistateDesign(MPNNDesign):
    """
    Class for running MPNN to do multistate or sequence-symmetric design.
    """

    def __init__(
        self,
        residue_selectors: List[ResidueSelector],
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNDesign.
        :param: kwargs: keyword arguments to pass to MPNNDesign.
        Initialize the base class for MPNN runners with common attributes.
        """
        super().__init__(*args, **kwargs)
        self.residue_selectors = residue_selectors

    def setup_runner(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Setup the MPNNRunner. Make a tmpdir and write input files to the tmpdir.
        Output sequences and scores will be written temporarily to the tmpdir as well.
        """
        import json, os
        from collections import defaultdict

        # take advantage of the superclass's setup_runner() to do most of the work
        super().setup_runner(pose)
        self.is_setup = False
        # make the jsonl file for the tied postions
        tied_positions_path = os.path.join(self.tmpdir, "tied_positions.jsonl")
        # make a dict keyed by pose residue indices with chains as values
        chains_dict = {
            i: pose.pdb_info().chain(i) for i in range(1, pose.total_residue() + 1)
        }
        # make a dict mapping chain letters back to numbers
        chain_numbers = {
            letter: number
            for number, letter in enumerate(sorted(set(chains_dict.values())), start=1)
        }
        # get a boolean mask of the residues in the design_selector
        if self.design_selector is not None:
            designable_filter = list(self.design_selector.apply(pose))
        else:  # if no design_selector is provided, make all residues designable
            designable_filter = [True] * pose.size()
        # make a dict mapping of residue numbers to whether they are designable
        designable_dict = dict(zip(range(1, pose.size() + 1), designable_filter))
        tied_positions_dict = {"tmp": []}
        # now need the one indexed indices of all the True residues in the residue_selectors
        residue_indices_lists = []
        for sel in self.residue_selectors:
            residue_indices_list = []
            for i, selected in enumerate(sel.apply(pose), start=1):
                if selected:
                    residue_indices_list.append(i)
                else:
                    pass
            residue_indices_lists.append(residue_indices_list)
        # flatten the list of lists into a single list of tuples by unpacking the above
        tied_positions_list = list(zip(*residue_indices_lists))
        for tied_positions in tied_positions_list:
            # get the chains for the tied positions
            tied_position_dict = defaultdict(list)
            # we only tie a position if all the residues are up for design
            designable = True
            for tied_position in tied_positions:
                # check if the residue is designable
                if designable_dict[tied_position]:
                    pass
                else:
                    designable = False
                # get the residue index and chain
                chain = chains_dict[tied_position]
                # we need to offset the residue index by the chain now
                residue_index = (
                    tied_position - pose.chain_begin(chain_numbers[chain]) + 1
                )
                # add the residue index and chain to the dict
                tied_position_dict[chain].append(residue_index)
            # skip this tied position if any of the residues are not designable
            if not designable:
                continue
            else:
                pass
            # the output json should have a single entry with a list of dicts of the tied positions
            tied_positions_dict["tmp"].append(dict(tied_position_dict))

        # write the tied_positions_dict to a jsonl file
        with open(tied_positions_path, "w") as f:
            f.write(json.dumps(tied_positions_dict))
        # update the flags for the biounit, chain_id, and fixed_positions paths
        flag_update = {
            "--tied_positions_jsonl": tied_positions_path,
        }
        self.update_flags(flag_update)
        self.is_setup = True
        return


@requires_init
def mpnn_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    from pathlib import Path
    import sys
    from itertools import product
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        NeighborhoodResidueSelector,
        TrueResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

    if "mpnn_temperature" in kwargs:
        if kwargs["mpnn_temperature"] == "scan":
            mpnn_temperatures = [0.1, 0.2, 0.5]
        else:
            mpnn_temperature = float(kwargs["mpnn_temperature"])
            assert (
                0.0 <= mpnn_temperature <= 1.0
            ), "mpnn_temperature must be between 0 and 1"
    else:
        mpnn_temperatures = [0.1]
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    full_selector = TrueResidueSelector()
    selector_options = {
        "full": full_selector,
        "interface": interface_selector,
        "neighborhood": neighborhood_selector,
    }
    # make the inverse dict of selector options
    selector_inverse_options = {value: key for key, value in selector_options.items()}
    if "mpnn_design_area" in kwargs:
        if kwargs["mpnn_design_area"] == "scan":
            mpnn_design_areas = [
                selector_options[key] for key in ["full", "interface", "neighborhood"]
            ]
        else:
            try:
                mpnn_design_areas = [selector_options[kwargs["mpnn_design_area"]]]
            except:
                raise ValueError(
                    "mpnn_design_area must be one of the following: full, interface, neighborhood"
                )
    else:
        mpnn_design_areas = [selector_options["interface"]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()

        # TODO loop over the product of mpnn_temperatures and mpnn_design_selectors
        mpnn_conditions = list(product(mpnn_temperatures, mpnn_design_areas))
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area) in enumerate(list(mpnn_conditions)):
            pose = original_pose.clone()
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )
            print_timestamp("Designing interface with MPNN", start_time)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=mpnn_design_area,
                omit_AAs="CX",
                temperature=mpnn_temperature,
                **kwargs,
            )
            # design the pose
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_temperature": mpnn_temperature,
                    "mpnn_design_area": selector_inverse_options[mpnn_design_area],
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def mpnn_paired_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
    from crispy_shifty.protocols.mpnn import MPNNMultistateDesign
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
        chA = ChainSelector(1)
        chB = ChainSelector(2)
        chC = ChainSelector(3)
        # make a list of linked residue selectors, we want to link chA and chC
        residue_selectors = [chA, chC]
        # get the length of state Y
        offset = pose.chain_end(2)
        # get a boolean mask of the residues in chA
        chA_filter = list(chA.apply(pose))
        # make a list of the corresponding residues in state X that are interface in Y
        X_interface_residues = [
            i + offset for i, designable in enumerate(chA_filter, start=1) if designable
        ]
        X_interface_residues_str = ",".join(str(i) for i in X_interface_residues)
        # make design selectors
        interface_selector = interface_between_selectors(chA, chB)
        X_selector = ResidueIndexSelector(X_interface_residues_str)
        design_selector = OrResidueSelector(interface_selector, X_selector)

        print_timestamp("Multistate design with MPNN", start_time)
        # construct the MPNNMultistateDesign object
        mpnn_design = MPNNMultistateDesign(
            design_selector=design_selector,
            residue_selectors=residue_selectors,
            omit_AAs="CX",
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # generate the original pose, with the sequences written to the datacache
        ppose = io.to_packed(pose)
        yield ppose
