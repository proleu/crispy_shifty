# Python standard library
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# 3rd party library imports

# Rosetta library imports
from pyrosetta.distributed import requires_init
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose

class InpaintingRunner(ABC):
    """
    Class for running ProteinInpainting on any cluster.
    """

    def __init__(
        self,
        pose: Union[Pose, PackedPose],
        input_file: Optional[str] = None,
        contigs: Optional[str] = None,
        inpaint_seq: Optional[str] = None,
        inpaint_str: Optional[str] = None,
        res_translate: Optional[str] = None,
        tie_translate: Optional[str] = None,
        block_rotate: Optional[str] = None,
        multi_templates: Optional[str] = None,
        num_designs: Optional[int] = 100,
        topo_pdb: Optional[str] = None,
        topo_conf: Optional[str] = None,
        topo_contigs: Optional[str] = None,
        save_original_pose: Optional[bool] = False,
        **kwargs,
    ):
        """
        """

        import os
        from pathlib import Path

        import git

        self.pose = pose
        self.input_file = input_file
        self.contigs = contigs
        self.inpaint_seq = inpaint_seq
        self.inpaint_str = inpaint_str
        self.res_translate = res_translate
        self.tie_translate = tie_translate
        self.block_rotate = block_rotate
        self.multi_templates = multi_templates
        self.num_designs = num_designs
        self.topo_pdb = topo_pdb
        self.topo_conf = topo_conf
        self.topo_contigs = topo_contigs
        self.save_original_pose = save_original_pose

        self.scores = dict(self.pose.scores)

        # setup standard flags for superfold
        all_flags = {
            "--contigs": self.contigs,
            "--inpaint_seq": self.inpaint_seq,
            "--inpaint_str": self.inpaint_str,
            "--res_translate": self.res_translate,
            "--tie_translate": self.tie_translate,
            "--block_rotate": self.block_rotate,
            "--multi_templates": self.multi_templates,
            "--num_designs": self.num_designs,
            "--topo_pdb": self.topo_pdb,
            "--topo_conf": self.topo_conf,
            "--topo_contigs": self.topo_contigs,
        }
        self.flags = {k: v for k, v in all_flags.items() if v is not None}

        self.allowed_flags = [
            "--pdb",
            "--out",
            "--contigs",
            "--inpaint_seq",
            "--inpaint_str",
            "--res_translate",
            "--tie_translate",
            "--block_rotate",
            "--multi_templates",
            "--num_designs",
            "--topo_pdb",
            "--topo_conf",
            "--topo_contigs",
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
            Path(__file__).parent.parent.parent / "proteininpainting" / "inpaint.py"
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

    def set_script(self, script: str) -> None:
        """
        :param: script: The path to the script.
        :return: None.
        """
        self.script = script
        self.update_command()
        return

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the InpaintingRunner. Checks for various best
        practice locations for the tmpdir in the following order: TMPDIR, PSCRATCH,
        CSCRATCH, /net/scratch. Uses the cwd if none of these are available.
        """
        import os
        import pwd
        import uuid

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
        Remove the temporary directory for the InpaintingRunner.
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
                f"--pdb {self.input_file}",
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
        Setup the InpaintingRunner.
        Create a temporary directory for the InpaintingRunner.
        Dump the pose temporarily to a PDB file in the temporary directory.
        Update the flags dictionary with the provided dictionary if any.
        Setup the command line arguments for the InpaintingRunner.
        """
        import os
        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore, clearPoseExtraScores

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
        new_pose = self.pose.clone()
        clearPoseExtraScores(new_pose) # highly important, otherwise pdbstrings in the scores get added to the pose lol
        pdbstring = io.to_pdbstring(new_pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        if self.save_original_pose:
            setPoseExtraScore(self.pose, "original_pose", pdbstring)
        # update the flags with the path to the tmpdir
        self.update_flags({"--out": out_path + "/inpaint"})
        if flag_update is not None:
            self.update_flags(flag_update)
        else:
            pass
        self.update_command()
        self.is_setup = True
        return

    @abstractmethod
    def configure_flags(self) -> None:
        """
        This function needs to be implemented by the child class of InpaintingRunner.
        """
        pass

    def generate_inpaints(self) -> Iterator[PackedPose]:
        """
        :param: pose: Pose object in which to inpaint a fusion.
        :return: None
        Run inpainting on the provided pose in a subprocess.
        Read the results from the temporary directory and store them in the pose.
        Remove the temporary directory.
        """
        import os
        import sys
        from glob import glob
        import numpy as np
        from pathlib import Path

        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.utils.io import cmd

        assert self.is_setup, "InpaintingRunner is not set up."

        # run the command in a subprocess
        out_err = cmd(self.command)
        print(out_err)

        inpaints = []
        trb_files = glob(os.path.join(self.tmpdir, "inpaint*.trb"))
        for trb_file in trb_files:
            raw_scores = np.load(trb_file, allow_pickle=True)
            # process scores
            inpaint_scores = {
                "inpaint_mean_lddt": np.mean(raw_scores["inpaint_lddt"]),
                "contigs": ';'.join(raw_scores["contigs"]),
                "sampled_mask": ';'.join(raw_scores["sampled_mask"]),
                "inpaint_seq_resis": ','.join(str(i) for i, mask in enumerate(raw_scores["inpaint_seq"], start=1) if not mask),
                "inpaint_str_resis": ','.join(str(i) for i, mask in enumerate(raw_scores["inpaint_str"], start=1) if not mask),
                "inpaint_length": len(raw_scores["inpaint_lddt"]),
                "total_length": len(raw_scores["inpaint_seq"]),
            }
    
            inpaint_pose = io.to_pose(io.pose_from_file(trb_file[:-3]+"pdb"))
            for k, v in self.scores.items():
                setPoseExtraScore(inpaint_pose, k, v)
            for k, v in inpaint_scores.items():
                setPoseExtraScore(inpaint_pose, k, v)
            
            inpaints.append(io.to_packed(inpaint_pose))

        self.teardown_tmpdir()

        for inpaint in inpaints:
            yield inpaint

class InpaintingFusion(InpaintingRunner):
    """
    Class for inpainting rigid fusions
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to InpaintingRunner.
        :param: kwargs: keyword arguments to pass to InpaintingRunner.
        Initialize the base class for inpainting runners with common attributes.
        """
        super().__init__(*args, **kwargs)

    def configure_flags(self) -> None:
        """
        Configure contigs and inpaint_seq flags for building junctions. If the termini are far enough apart, calculate
        the number of residues to inpaint by the length a helix between the termini would be and adding 6 residues for
        loops. Maybe build a topo_pdb file by modeling that helix and docking it into the junction? Or add a guiding 
        helical fragment from such a dock. Configure the inpaint_seq flag by an interfacebyvector selector between the
        chains to join.
        """
        return