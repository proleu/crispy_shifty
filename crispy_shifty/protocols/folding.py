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
        models: Optional[Union[int, List[int], List[str], str]] = "all",
        recycle_tol: Optional[float] = 0.0,
        reference_pdb: Optional[str] = None,
        **kwargs,
    ):
        """
        :param: pose: The pose to run Superfold on.
        :param: input_file: The path to the input file. If none, the pose will be used
        on its own.
        :param: amber_relax: Whether to run AMBER relaxation.
        :param: fasta_path: The path to the FASTA file, if any.
        :param: initial_guess: Whether to use an initial guess. If True, the pose will
        be used as an initial guess. If a string, the string will be used as the path
        to the initial guess.
        :param: max_recycles: The maximum number of cycles to run Superfold.
        :param: model_type: The type of model to run.
        :param: models: The models to run.
        :param: recycle_tol: The tolerance for recycling. If the difference between
        mean plddt changes less than this value since the last recycle, the model will
        stop early.
        :param: reference_pdb: The path to the reference PDB for RMSD calculation.
        Initialize the class with the provided attributes.
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
        # the models flag is a special case as well because it is either an int, a list
        # of ints, a list of strings, or a string
        if type(self.models) == int:  # convert to string
            self.flags["--models"] = str(self.models)
        elif type(self.models) == list:  # check if list of ints
            if type(self.models[0]) == int:  # convert to string
                self.flags["--models"] = " ".join(str(x) for x in self.models)
            elif type(self.models[0]) == str:  # join with space
                self.flags["--models"] = " ".join(self.models)
            else:
                raise TypeError(
                    "The models param must be either a list of ints/strings or a single int/string."
                )
        elif type(self.models) == str:  # probably good to go
            self.flags["--models"] = self.models
        else:
            raise TypeError(
                "The models param must be either a list of ints/strings or a single int/string."
            )
        self.flags.update(
            {
                "--max_recycles": str(self.max_recycles),
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
            "--keep_chain_order",
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
    
    def get_fasta_path(self) -> str:
        """
        :return: fasta path.
        """
        return self.fasta_path

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

    def override_input_file(self, input_file: str) -> None:
        """
        Override the input_file attribute. 
        :param: input_file: The new input_file.
        :return: None
        """
        self.input_file = input_file
        return None

    def set_fasta_path(self, fasta_path: str) -> None:
        """
        :param: fasta_path: The path to the fasta file.
        :return: None.
        """
        self.fasta_path = fasta_path
        return None

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the SuperfoldRunner. Checks for various best
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
        from crispy_shifty.protocols.mpnn import fasta_to_dict
        from crispy_shifty.utils.io import cmd

        assert self.is_setup, "SuperfoldRunner is not setup."

        scores = dict(pose.scores)
        # run the command in a subprocess
        out_err = cmd(self.command)
        print(out_err)
        json_files = glob(os.path.join(self.tmpdir, "*_prediction_results.json"))
        # read the json files and update a dict of dicts of dicts of scores
        # the outer dict is keyed by the pymol_name, values are all model/seed results
        # the inner dict is keyed by the model and seed, and the value is the scores
        results = defaultdict(dict)
        for json_file in json_files:
            pymol_name, model_seed, result = process_results_json(json_file)
            results[pymol_name].update({model_seed: result})
        # turn results back into a regular dict
        results = dict(results)
        # check if there were already sequences in the pose datacache
        seqs = {k: {"seq": v} for k, v in scores.items() if k in results.keys()}
        # check if a fasta was provided
        if self.fasta_path is not None:
            # check if the fasta is the same as the input file
            if self.fasta_path == self.input_file:
                # if so, make a dict of the sequences in the fasta
                tag_seq_dict = fasta_to_dict(self.fasta_path)
                # and nest the sequences in that dict
                tag_seq_dict = {k: {"seq": v} for k, v in tag_seq_dict.items()}
            else:
                raise NotImplementedError(
                    "Fasta path is not the same as the input file."
                )

            if len(seqs) > 0:  # check that the seqs dict matches the fasta dict
                if seqs == tag_seq_dict:
                    pass
                else:
                    seqs = tag_seq_dict  # we want the seqs we did predictions on
            else:
                seqs = tag_seq_dict  # we want the seqs we did predictions on

        elif len(seqs) == 0 and len(results) == 1:
            # then this was a single sequence run
            seqs = {"tmp": {"seq": pose.sequence}}

        else:
            raise NotImplementedError("I am not sure how this behaves with silents")

        # update the results with the sequences and update the pose with those results
        for tag, result in results.items():
            result.update(seqs[tag])
            setPoseExtraScore(pose, tag, json.dumps(result))
        # clean up the temporary files
        self.teardown_tmpdir()
        return


def generate_decoys_from_pose(
    pose: Pose,
    filter_dict: Dict[
        str, Tuple[Union[eq, ge, gt, le, lt, ne], Union[int, float, str]]
    ] = {},
    label_first: Optional[bool] = False,
    prefix: Optional[str] = "tmp",
    rank_on: Optional[
        Union["mean_plddt", "pTMscore", "rmsd_to_input", "rmsd_to_reference"]
    ] = "mean_plddt",
    **kwargs,
) -> Iterator[Pose]:
    """
    :param: pose: Pose object to generate decoys from.
    :param: filter_dict: dictionary of filters to apply to the decoys. This is supplied
    as a dictionary of the form {'score_name': (operator, value)} where the operator is
    one of the following: eq, ge, gt, le, lt, ne. Example: {'mean_plddt': (ge, 0.9)} .
    Note that this defaults to an empty dict, which will simply return all decoys.
    prefix: for poses with many tags/pymol_names in results, get all results with the
    same prefix
    rank_on: the score to rank multiple model results for the same tag/pymol_name on.
    kwargs: keyword arguments (if they are not in the named arguments they will be ignored)
    :return: iterator of poses
    The decoys are generated by applying the filters to the model results in the pose.
    """
    import json, sys
    from pathlib import Path
    import pyrosetta
    from pyrosetta.rosetta.core.pose import setPoseExtraScore

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.mpnn import thread_full_sequence

    # get the scores from the pose that have the prefix
    scores = {k: v for k, v in pose.scores.items() if prefix in k}
    # get the scores from the pose that don't have the prefix
    pose_scores = {k: v for k, v in pose.scores.items() if prefix not in k}
    # the format we expect is
    # {'tag1': {'model_seed': json_string of scores dict, 'seq': sequence}, tag2: ...}
    # for each tag, check that the format is correct then:
    # 1. get the sequence
    # 2. get the (possibly multiple) model/seed results by loading the json_string
    # 3. sort the model/seed results by the rank_on score and take the top result only
    # 4. apply the filter(s) that result
    for i, (tag, result) in enumerate(sorted(scores.items())):
        # try to load the json string
        try:
            results = json.loads(result)
        # fail gracefully if it can't be loaded
        except json.decoder.JSONDecodeError:
            print(f"Could not load json string for {tag}")
            continue
        # anything that doesn't have a seq or results is a problem, once loaded
        if "seq" not in results.keys():
            raise ValueError(
                f"{tag} does not have a sequence in the pose datacache. "
                "This is required for generating decoys."
            )
        else:
            sequence = results.pop("seq")
        # remaining should be model/seed results
        if "model_" not in list(results.keys())[0]:
            raise ValueError(
                f"{tag} does not have a model/seed in the pose datacache. "
                "This is required for generating decoys."
            )
        else:
            pass
        # sort the model/seed results by the rank_on score and take the top result only
        # this is a bit tricky because higher is better for mean_plddt and pTMscore
        # and lower is better for rmsd_to_input and rmsd_to_reference
        if rank_on == "mean_plddt" or rank_on == "pTMscore":
            # higher is better
            model_seed_results = sorted(
                results.items(), key=lambda x: x[1][rank_on], reverse=True
            )
        elif rank_on == "rmsd_to_input" or rank_on == "rmsd_to_reference":
            # lower is better
            model_seed_results = sorted(
                results.items(), key=lambda x: x[1][rank_on], reverse=False
            )
        else:
            raise ValueError(
                f"{rank_on} is not a valid rank_on score. "
                "This is required for generating decoys."
            )
        # get the top result
        top_result = model_seed_results[0][1]
        # setup flag: the decoy hasn't been discarded yet
        keep_decoy = True
        # apply the filter(s)
        for score_name, (operator, value) in filter_dict.items():
            if operator(top_result[score_name], value):
                pass
            else:
                # if the filter fails, don't keep the decoy
                keep_decoy = False
                continue
        # if the decoy passes all filters, yield it
        if keep_decoy:
            decoy = thread_full_sequence(
                pose,
                sequence,
            )
            # add the scores from the top result to the decoy
            pose_scores.update(top_result)
            if label_first:
                if i == 0:
                    pose_scores["designed_by"] = "rosetta"
                else:
                    pose_scores["designed_by"] = "mpnn"
            else:
                pass
            for k, v in pose_scores.items():
                setPoseExtraScore(decoy, k, v)
            yield decoy
        else:
            pass
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

    from operator import lt, gt
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
    # hacky split pdb_path into pdb_path and fasta_path
    pdb_path = kwargs.pop("pdb_path")
    pdb_path, fasta_path = tuple(pdb_path.split("____"))

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        # skip the kwargs check
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up for AF2", start_time)
        runner = SuperfoldRunner(pose=pose, fasta_path=fasta_path, **kwargs)
        runner.setup_runner(file=fasta_path)
        # initial_guess, reference_pdb both are the tmp.pdb
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
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # setup prefix, rank_on, filter_dict (in this case we can't get from kwargs)
        # TODO, for the pilot run I will not filter the decoys
        # filter_dict = {
        #     "mean_plddt": (gt, 90.0),
        #     "rmsd_to_reference": (lt, 1.75),
        #     "mean_pae_interaction": (lt, 7.5),
        # }
        filter_dict = {}
        rank_on = "mean_plddt"
        prefix = "mpnn_seq"
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            label_first=True,
            prefix=prefix,
            rank_on=rank_on,
        ):
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy


@requires_init
def fold_paired_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the superfold script.
    :param: kwargs: keyword arguments to be passed to the superfold script.
    :return: an iterator of PackedPose objects.
    """

    from operator import lt, gt
    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # hacky split pdb_path into pdb_path and fasta_path
    pdb_path = kwargs.pop("pdb_path")
    pdb_path, fasta_path = tuple(pdb_path.split("____"))

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        # skip the kwargs check
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        # need to split this pose into either Y or X depending on kwargs
        # TODO
        # load fasta into a dict
        tmp_fasta_dict = fasta_to_dict(fasta_path)
        pose_chains = list(pose.split_by_chain())
        if kwargs["predict"] == "Y":
            # slice out the bound state, aka chains A and B
            tmp_pose = Pose()
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                tmp_pose, pose_chains[0], new_chain=True
            )
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                tmp_pose, pose_chains[1], new_chain=True
            )
            # fix the fasta by splitting on chainbreaks '/' and rejoining the first two
            tmp_fasta_dict = {
                tag: "/".join(seq.split("/")[0:2]) for tag, seq in tmp_fasta_dict.items()
            }
        elif kwargs["predict"] == "X":
            # slice out the free state, aka chain C
            tmp_pose = Pose()
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                tmp_pose, pose_chains[2], new_chain=True
            )
            # fix the fasta by splitting on chainbreaks '/' and getting the last one
            tmp_fasta_dict = {
                tag: "/".join(seq.split("/")[-1]) for tag, seq in tmp_fasta_dict.items()
            }
        else:
            raise ValueError("predict kwarg must be either Y (bound) or X (free)")

        pose = tmp_pose.clone()
        print_timestamp("Setting up for AF2", start_time)
        runner = SuperfoldRunner(pose=pose, fasta_path=fasta_path, **kwargs)
        runner.setup_runner(file=fasta_path)
        # initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        flag_update = {
            "--initial_guess": initial_guess,
            "--reference_pdb": reference_pdb,
        }
        # TODO now we have to point to the right fasta file
        new_fasta_path = str(Path(runner.get_tmpdir()) / "tmp.fa")
        dict_to_fasta(tmp_fasta_dict, new_fasta_path)
        runner.set_fasta_path(new_fasta_path)
        runner.override_input_file(new_fasta_path)
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # setup prefix, rank_on, filter_dict (in this case we can't get from kwargs)
        # TODO, for the pilot run I will not filter the decoys
        # filter_dict = {
        #     "mean_plddt": (gt, 90.0),
        #     "rmsd_to_reference": (lt, 1.75),
        #     "mean_pae_interaction": (lt, 7.5),
        # }
        filter_dict = {}
        rank_on = "mean_plddt"
        prefix = "mpnn_seq"
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            label_first=True,
            prefix=prefix,
            rank_on=rank_on,
        ):
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy
