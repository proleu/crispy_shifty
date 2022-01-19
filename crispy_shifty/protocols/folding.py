# Python standard library
from operator import eq, ge, gt, le, lt, ne
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


def process_results_json(path: str) -> Tuple[str, str, Dict[Any, Any]]:
    """
    :param: path: The path to the JSON to process.
    :return: A tuple containing the name/tag of the prediction target, the name of the 
    model/seed that generated the results and the dictionary of results.
    Load a JSON as a dict. Return the name of the prediction target, the name of the
    model/seed that generated the results and the dictionary of results.
    """
    import json
    
    with open(path, "r") as f:
        scores = json.load(f)

    model = scores["model"]
    seed = scores["seed"]
    if "ptm" in scores["type"]:
        ptm = "_ptm"
    else:
        ptm = ""
    # results json filenames have the format: 
    # {pymol_name}_model_{model}{""|"ptm"}_seed_{seed}_prediction_results.json
    # if we work backwards, after loading the json, we can get the pymol_name
    filename = path.split("/")[-1]
    model_seed = f"model_{model}{ptm}_seed_{seed}"
    pymol_name = filename.replace(f"_{model_seed}_prediction_results.json", "")
    return pymol_name, model_seed, scores


class SuperfoldRunner:
    """
    Class for running AF2 on any cluster with @rdkibler's Superfold.
    """

    import os, pwd, uuid, shutil, subprocess
    import pyrosetta.distributed.io as io

    def __init__(
        self,
        pose: Union[Pose, PackedPose],
        input_file: Optional[str] = None,
        amber_relax: bool = False,
        fasta_path: Optional[str] = None,
        initial_guess: Optional[Union[bool, str]] = None,
        max_recycles: Optional[int] = 3,
        model_type: Optional[str] = "monomer_ptm",
        models: Optional[Union[int, List[int], str]] = "all",
        recycle_tol: Optional[float] = 0.0,
        reference_pdb: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the class with the provided attributes.
        TODO params and docstring.
        """

        import git
        import os
        from pathlib import Path

        self.pose = pose
        self.input_file = input_file
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
            "--pct_seq_mask": "0.15",
            "--seed_start": "0",
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
        # the amber_relax and reference_pdb flags are special cases as well
        if self.amber_relax:
            # store_true flag
            self.flags["--amber_relax"] = " "
        else:
            pass
        if self.reference_pdb is not None:
            self.flags["--reference_pdb"] = self.reference_pdb
        else:
            pass
        self.flags.update(
            {
                "--max_recycles": str(self.max_recycles),
                # cast models to a list (they may already be a list, that's fine)
                # and concatenate the list with spaces as a string
                "--models": " ".join([str(x) for x in list(self.models)]),
                "--recycle_tol": str(self.recycle_tol),
                "--type": self.model_type,
            }
        )
        # 19 total flags plus input_files
        self.allowed_flags = [
            # flags that have default values
            "--mock_msa_depth",
            "--nstruct",
            "--num_ensemble",
            "--pct_seq_mask",
            "--seed_start",
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
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        self.python = str(Path(root) / "envs" / "crispy" / "bin" / "python")
        if os.path.exists(self.python):
            pass
        else:  # crispy env must be installed in envs/crispy or must be used on DIGS
            self.python = "/projects/crispy_shifty/envs/crispy/bin/python"
        self.script = str(
            Path(__file__).parent.parent.parent / "superfold" / "run_superfold_devel.py"
        )
        self.tmpdir = None  # this will be updated by the setup_tmpdir method.
        self.command = None  # this will be updated by the setup_runner method.
        self.is_setup = False  # this will be updated by the setup_runner method.

    def get_command(self) -> str:
        """
        :return: command to run.
        """
        return self.command

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

    def update_command(self) -> None:
        """
        :return: None
        Update the command to run.
        """
        self.command = " ".join(
            [
                f"{self.python} {self.script}",
                f"{self.input_file}",
                " ".join([f"{k} {v}" for k, v in self.flags.items()]),
            ]
        )

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

    def setup_runner(
        self, file: Optional[str] = None, flag_update: Optional[Dict[str, str]] = None
    ) -> None:
        """
        :param: file: path to input file. If None, use the dumped tmp.pdb.
        :param: flag_update: dictionary of flags to update, if any.
        :return: None
        Setup the SuperfoldRunner.
        Create a temporary directory for the SuperfoldRunner.
        Dump the pose temporarily to a PDB file in the temporary directory.
        Update the flags dictionary with the provided dictionary if any.
        Setup the command line arguments for the SuperfoldRunner.
        """
        import json, os, subprocess, sys
        import pyrosetta.distributed.io as io

        # setup the tmpdir
        self.setup_tmpdir()
        out_path = self.tmpdir
        # set input_file
        if file is not None:
            self.input_file = file
        else:
            self.input_file = os.path.join(out_path, "tmp.pdb")
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(out_path, "tmp.pdb")
        pdbstring = io.to_pdbstring(self.pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        # update the flags with the path to the tmpdir
        self.update_flags({"--out_dir": out_path})
        if flag_update is not None:
            self.update_flags(flag_update)
        else:
            pass
        self.update_command()
        self.is_setup = True
        return

    def apply(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run Superfold on.
        :return: None
        Run Superfold on the provided pose in a subprocess.
        Read the results from the temporary directory and store them in the pose.
        Remove the temporary directory.
        """
        import json, os, subprocess, sys
        from collections import defaultdict
        from glob import glob
        from pathlib import Path
        import pyrosetta
        from pyrosetta.rosetta.core.pose import setPoseExtraScore
        import pyrosetta.distributed.io as io

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.utils.io import cmd

        assert self.is_setup, "SuperfoldRunner is not setup."

        # run the command in a subprocess
        out_err = cmd(self.command)
        print(out_err)
        json_files = glob(os.path.join(self.tmpdir, "*_prediction_results.json"))
        # read the json files and update a dict of dicts of dicts of scores
        # the outer dict is keyed by the pymol_name, values are all model/seed results
        # the inner dict is keyed by the model and seed, and the value is the scores 
        results = defaultdict(dict)
        for json_file in json_files:
            pymol_name, model_seed, scores = process_results_json(json_file)
            results[pymol_name].update({model_seed: scores})
        # update the pose with the scores
        for pymol_name, model_scores in results.items():
            setPoseExtraScore(pose, pymol_name, json.dumps(model_scores))
        # clean up the temporary files
        self.teardown_tmpdir()
        return


# TODO: utility functions for generating decoy outputs from the SuperfoldRunner
def update_seq_entry_with_results():
    """
    :return: None TODO
    Update the sequence entry with the results from Superfold.
    """
    return


def generate_decoys_from_pose(pose: Pose, filter_dict: Dict[str, Tuple[Union[eq, ge, gt, le, lt, ne], Union[int, float, str]]]) -> Iterator[Pose]:
    """
    :param: pose: Pose object to generate decoys from.
    :param: filter_dict: dictionary of filters to apply to the decoys. This is supplied 
    as a dictionary of the form {'score_name': (operator, value)} where the operator is
    one of the following: eq, ge, gt, le, lt, ne.
    :return: iterator of poses
    The decoys are generated by applying the filters to the pose.
    TODO
    """
    return



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

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up for AF2", start_time)
        # TODO run the superfold script and get the best decoy, 
        # TODO set pose = best decoy?
        runner = SuperfoldRunner(pose=pose, **kwargs)
        runner.setup_runner(file=kwargs["fasta_path"])
        # TODO initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        flag_update = {
            "--initial_guess": initial_guess,
            "--reference_pdb": reference_pdb,
        }
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict # TODO we actually want to add the seq to the existing dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose
