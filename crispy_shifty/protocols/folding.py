# Python standard library
from typing import Dict, Iterator, List, Optional, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


class SuperfoldRunner():
    """
    Class for running AF2 on any cluster with @rdkibler's Superfold.
    """

    import os, pwd, uuid, shutil, subprocess
    import pyrosetta.distributed.io as io

    def __init__(
        self,
        pose: Union[Pose, PackedPose],
        amber_relax: bool = False,
        fasta_path: Optional[str] = None,
        initial_guess: Optional[Union[bool, str]] = None,
        max_recycles: Optional[int] = 3,
        model_type: Optional[str] = "monomer_ptm",
        models = Optional[Union[int, List[int], str]] = "all",
        recycle_tol: Optional[float] = 0.0,
        reference_pdb: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the class with the provided attributes.
        TODO params and docstring.
        """

        import os
        from pathlib import Path

        self.pose = pose
        self.amber_relax = amber_relax
        self.fasta_path = fasta_path
        self.initial_guess = initial_guess
        self.max_recycles = max_recycles
        self.model_type = model_type
        self.models = models
        self.recycle_tol = recycle_tol
        self.reference_pdb = reference_pdb
        # setup standard flags for superfold
        self.flags = {
            "--mock_msa_depth": "1",
            "--nstruct": "1",
            "--num_ensemble": "1",
            "--pad_lengths": " ", # store_true flag
            "--pct_seq_mask": "0.15",
            "--seed_start": "0",
            "--turbo": " ", # store_true flag
            "--version": "monomer",
        }
        # add the flags provided by the user
        # the initial_guess flag is a special case because it is a boolean or string
        initial_guess_flag = "--initial_guess"
        if self.initial_guess is not None:
            if initial_guess == True:
                self.flags[initial_guess_flag] = " "
            else:
                self.flags[initial_guess_flag] = self.initial_guess
        else:
            pass
        self.flags.update(
            {
                # store_true flag
                "--amber_relax": " " if self.amber_relax else "": "",
                "--max_recycles": str(self.max_recycles),
                # cast models to a list (they may already be a list, that's fine)
                # and concatenate the list with spaces as a string
                "--models": " ".join([str(x) for x in list(self.models)]),
                "--recycle_tol": str(self.recycle_tol),
                "--reference_pdb": self.reference_pdb if self.reference_pdb is not None else "": "",
                "--type": self.model_type,
            }
        )
        # 21 total flags plus input_files
        self.allowed_flags = [
            "" # placeholder
            # flags that have default values 
            "--mock_msa_depth",
            "--nstruct",
            "--num_ensemble",
            "--pad_lengths",
            "--pct_seq_mask",
            "--seed_start",
            "--turbo",
            "--version",
            # flags that are set by the constructor
            "--amber_relax",
            "--initial_guess",
            "--max_recycles",
            "--models",
            "--out_dir",
            "--recycle_tol",
            "--reference_pdb",
            "--type",
            # flags that are optional
            "--enable_dropout",
            "--output_pae",
            "--overwrite",
            "--save_intermediates",
            "--show_images",
        ]
        self.script = str(Path(__file__).parent.parent / "superfold" / "run_crispy.py") # TODO
        self.tmpdir = None  # this will be updated by the setup_tmpdir method.
        self.command = None  # this will be updated by the setup_runner method.
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

    def get_tmpdir(self) -> str:
        """
        :return: temporary directory path.
        """
        return self.tmpdir

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the SuperfoldRunner. Checks for various best
        practice locations for the tmpdir in the following order: PSCRATCH, TMPDIR, 
        /net/scratch. Uses the cwd if none of these are available.
        """
        import os, pwd, uuid

        if os.environ.get("PSCRATCH") is not None:
            tmpdir_root = os.environ.get("PSCRATCH")
        elif os.environ.get("TMPDIR") is not None:
            tmpdir_root = os.environ.get("TMPDIR")
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
        Remove the temporary directory for the SuperfoldRunner.
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

    def setup_runner(self, file: str) -> None:
        """
        :return: None
        Setup the SuperfoldRunner.
        Create a temporary directory for the SuperfoldRunner.
        Dump the pose temporarily to a PDB file in the temporary directory.
        Setup the command line arguments for the SuperfoldRunner.
        Run the SuperfoldRunner, and store the results in the temporary directory.
        Read the results from the temporary directory and store them in the pose.
        Remove the temporary directory.
        """
        import json, os, subprocess, sys
        import git
        from pathlib import Path
        import pyrosetta.distributed.io as io

        # # insert the root of the repo into the sys.path
        # sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        # from crispy_shifty.utils.io import cmd_no_stderr

        # setup the tmpdir
        self.setup_tmpdir()
        out_path = self.tmpdir
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(out_path, "tmp.pdb")
        pdbstring = io.to_pdbstring(self.pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        # use git to find the root of the repo
        repo = git.Repo(str(Path(_file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        python = str(Path(root) / "envs"/ "crispy" / "bin" / "python")
        if os.path.exists(python):
            pass
        else: # crispy env must be installed in envs/crispy or must be used on DIGS
            python = "/projects/crispy_shifty/envs/crispy/bin/python"
        # update the flags with the path to the tmpdir
        self.update_flags({"--out_dir": out_path})
        run_cmd = " ".join(
            [
                f"{python} {self.script}",
                f"{file}",
                " ".join([f"{k} {v}" for k, v in self.flags.items()]),
            ]
        )
        self.command = run_cmd
        print(f"Running command: {run_cmd}") # TODO
        self.is_setup = True
        return

#     def apply(self, pose: Pose) -> None:
#         """
#         :param: pose: Pose object to run MPNN on.
#         :return: None
#         Run MPNN on the provided pose.
#         Setup the MPNNRunner using the provided pose.
#         Run MPNN in a subprocess using the provided flags and tmpdir.
#         Read in and parse the output fasta file to get the sequences.
#         Each sequence designed by MPNN is then appended to the pose datacache.
#         """
#         import os, subprocess, sys
#         import git
#         from pathlib import Path
#         import pyrosetta
#         from pyrosetta.rosetta.core.pose import setPoseExtraScore
#         import pyrosetta.distributed.io as io
# 
#         # insert the root of the repo into the sys.path
#         sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#         from crispy_shifty.protocols.mpnn import fasta_to_dict, thread_full_sequence
#         from crispy_shifty.utils.io import cmd
# 
#         # setup runner
#         self.setup_runner(pose)
#         self.update_flags({"--out_folder": self.tmpdir})
#         self.update_script()
# 
#         # run mpnn by calling self.script and providing the flags
#         # use git to find the root of the repo
#         repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
#         root = repo.git.rev_parse("--show-toplevel")
#         python = str(Path(root) / "envs"/ "crispy" / "bin" / "python")
#         if os.path.exists(python):
#             pass
#         else: # crispy env must be installed in envs/crispy or must be used on DIGS
#             python = "/projects/crispy_shifty/envs/crispy/bin/python"
#         run_cmd = (
#             f"{python} {self.script}"
#             + " "
#             + " ".join([f"{k} {v}" for k, v in self.flags.items()])
#         )
#         out_err = cmd(run_cmd)
#         print(out_err)
#         alignments_path = os.path.join(self.tmpdir, "alignments/tmp.fa")
#         # parse the alignments fasta into a dictionary
#         alignments = fasta_to_dict(alignments_path, new_tags=True)
#         for i, (tag, seq) in enumerate(alignments.items()):
#             index = str(i).zfill(4)
#             setPoseExtraScore(pose, f"mpnn_seq_{index}", seq)
#         # clean up the temporary files
#         self.teardown_tmpdir()
#         return


@requires_init
def fold_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the superfold script.
    :param: kwargs: keyword arguments to be passed to the superfold script.
    :return: an iterator of PackedPose objects.
    """

    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.utils.io import cmd, print_timestamp

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

    # constants TODO

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up for AF2", start_time)
        # TODO run the superfold script and get the best decoy, set pose = best decoy
        runner = SuperfoldRunner(pose=pose, **kwargs)
        # TODO initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        runner.setup_runner(file=kwargs["fasta_path"])

        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose
