# Python standard library
from typing import *
# 3rd party library imports
import pandas as pd
from tqdm import tqdm
# Rosetta library imports
# Custom library imports

def parse_scorefile_oneshot(scores:str) -> pd.DataFrame:
    """
    Read in a scores.json from PyRosettaCluster in a single shot.
    Memory intensive for a larger scorefile because it does a matrix transposition.
    """
    import pandas as pd


    scores = pd.read_json(scores, orient="records", typ="frame", lines=True)
    scores = scores.T
    mat = scores.values
    n = mat.shape[0]
    dicts = list(mat[range(n), range(n)])
    index = scores.index
    tabulated_scores = pd.DataFrame(dicts, index=index)
    return tabulated_scores


def parse_scorefile_linear(scores:str) -> pd.DataFrame:
    """
    Read in a scores.json from PyRosettaCluster line by line.
    Uses less memory thant the oneshot method but takes longer to run.
    """
    import pandas as pd
    from tqdm import tqdm

    dfs = []
    with open(scores, "r") as f:
        for line in tqdm(f):
            dfs.append(pd.read_json(line).T)
    tabulated_scores = pd.concat(dfs)
    return tabulated_scores

from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector
def pymol_selection(pose:Pose, selector:ResidueSelector, name:str=None):
    import pyrosetta
    pymol_metric = pyrosetta.rosetta.core.simple_metrics.metrics.SelectedResiduesPyMOLMetric(selector)
    if name is not None:
        pymol_metric.set_custom_type(name)
    return pymol_metric.calculate(pose)

def print_timestamp(print_str, start_time, end="\n", *args):
    from time import time
    time_min = (time() - start_time) / 60
    print(f"{time_min:.2f} min: {print_str}", end=end)
    for arg in args:
        print(arg, end=end)


# Much of the following is extensively borrowed from pyrosetta.distributed.cluster.io

try:
    import toolz
except ImportError:
    print(
        "Importing 'pyrosetta.distributed.cluster.io' requires the "
        + "third-party package 'toolz' as a dependency!\n"
        + "Please install this package into your python environment. "
        + "For installation instructions, visit:\n"
        + "https://pypi.org/project/toolz/\n"
    )
    raise

import bz2
import collections
import json
import os
import pyrosetta.distributed.io as io
import uuid

from datetime import datetime
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.distributed.cluster.exceptions import OutputError
from pyrosetta.distributed.packed_pose.core import PackedPose

def _get_instance_and_metadata(
    kwargs: Dict[Any, Any]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    Get the current state of the PyRosettaCluster instance, and split the
    kwargs into the PyRosettaCluster instance kwargs and ancillary metadata.
    """
    import pyrosetta

    # Deleted a bunch of instance stuff from the original function here. Could add back in if helpful later, particularly if returning this to object-oriented structure.
    instance_kwargs = {}
    # tracking with kwargs instead of class attributes
    instance_kwargs["compressed"] = kwargs.pop("compressed")
    instance_kwargs["decoy_dir_name"] = kwargs.pop("decoy_dir_name")
    instance_kwargs["environment"] = kwargs.pop("environment")
    instance_kwargs["output_path"] = kwargs.pop("output_path")
    instance_kwargs["score_dir_name"] = kwargs.pop("score_dir_name")
    instance_kwargs["simulation_name"] = kwargs.pop("simulation_name")
    instance_kwargs["simulation_records_in_scorefile"] = kwargs.pop("simulation_records_in_scorefile")

    instance_kwargs["tasks"] = kwargs.pop("task")
    for option in ["extra_options", "options"]:
        if option in instance_kwargs["tasks"]:
            instance_kwargs["tasks"][option] = pyrosetta.distributed._normflags(
                instance_kwargs["tasks"][option]
            )
    # instance_kwargs["seeds"] = kwargs.pop("seeds") # add this back in when you figure out how to get the pyrosetta seed...

    return instance_kwargs, kwargs

def _get_output_dir(decoy_dir: str) -> str:
    """Get the output directory in which to write files to disk."""
    
    zfill_value = 4
    max_dir_depth = 1000
    try:
        decoy_dir_list = os.listdir(decoy_dir)
    except FileNotFoundError:
        decoy_dir_list = []
    if not decoy_dir_list:
        new_dir = str(0).zfill(zfill_value)
        output_dir = os.path.join(decoy_dir, new_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        top_dir = list(reversed(sorted(decoy_dir_list)))[0]
        if len(os.listdir(os.path.join(decoy_dir, top_dir))) < max_dir_depth:
            output_dir = os.path.join(decoy_dir, top_dir)
        else:
            new_dir = str(int(top_dir) + 1).zfill(zfill_value)
            output_dir = os.path.join(decoy_dir, new_dir)
            os.makedirs(output_dir, exist_ok=True)

    return output_dir

def _format_result(result: Union[Pose, PackedPose]) -> Tuple[str, Dict[Any, Any]]:
    """
    Given a `Pose` or `PackedPose` object, return a tuple containing
    the pdb string and a scores dictionary.
    """

    _pdbstring = io.to_pdbstring(result)
    _scores_dict = io.to_dict(result)
    _scores_dict.pop("pickled_pose", None)

    return (_pdbstring, _scores_dict)

def _parse_results(
    results: Union[
        Iterable[Optional[Union[Pose, PackedPose]]],
        Optional[Union[Pose, PackedPose]],
    ]
) -> Union[List[Tuple[str, Dict[Any, Any]]], NoReturn]:
    """
    Format output results on distributed worker. Input argument `results` can be a
    `Pose` or `PackedPose` object, or a `list` or `tuple` of `Pose` and/or `PackedPose`
    objects, or an empty `list` or `tuple`. Returns a list of tuples, each tuple
    containing the pdb string and a scores dictionary.
    """

    if isinstance(results, (Pose, PackedPose,),):
        if not io.to_pose(results).empty():
            out = [_format_result(results)]
        else:
            out = []
    elif isinstance(results, collections.abc.Iterable):
        out = []
        for result in results:
            if isinstance(result, (Pose, PackedPose,),):
                if not io.to_pose(result).empty():
                    out.append(_format_result(result))
            else:
                raise OutputError(result)
    elif not results:
        out = []
    else:
        raise OutputError(results)

    return out

def save_results(results: Any, kwargs: Dict[Any, Any]) -> None:
    """Write results and kwargs to disk."""

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    # REMARK_FORMAT = "REMARK CrispyShifty: " # sadly, this interferes with using PRC functions which read PRC remarks...
    REMARK_FORMAT = "REMARK PyRosettaCluster: "

    # could hack PRC with the following before running path_to_pose_or_ppose, but then you lose compatibility with outputs actually generated by PRC...
    # from pyrosetta.distributed.cluster.io import IO
    # IO.REMARK_FORMAT = 'REMARK CrispyShifty: '

    compressed = kwargs['compressed']
    decoy_dir_name = kwargs['decoy_dir_name']
    environment_file = kwargs['environment']
    output_path = kwargs['output_path']
    score_dir_name = kwargs['score_dir_name']
    simulation_name = kwargs['simulation_name']
    simulation_records_in_scorefile = kwargs['simulation_records_in_scorefile']

    # Parse and save results
    for pdbstring, scores in _parse_results(results):
        # kwargs = _process_kwargs(kwargs)
        output_dir = _get_output_dir(decoy_dir=os.path.join(output_path, decoy_dir_name))
        decoy_name = "_".join([simulation_name, uuid.uuid4().hex])
        output_file = os.path.join(output_dir, decoy_name + ".pdb")
        if compressed:
            output_file += ".bz2"
        score_dir = _get_output_dir(decoy_dir=os.path.join(output_path, score_dir_name))
        score_file = os.path.join(score_dir, decoy_name + ".json")
        extra_kwargs = {
            "crispy_shifty_decoy_name": decoy_name,
            "crispy_shifty_output_file": output_file,
        }
        if os.path.exists(environment_file):
            extra_kwargs[
                "crispy_shifty_environment_file"
            ] = environment_file
        if "crispy_shifty_datetime_start" in kwargs:
            datetime_end = datetime.now().strftime(DATETIME_FORMAT)
            duration = str(
                (
                    datetime.strptime(datetime_end, DATETIME_FORMAT)
                    - datetime.strptime(
                        kwargs["crispy_shifty_datetime_start"],
                        DATETIME_FORMAT,
                    )
                ).total_seconds()
            )  # For build-in functions
            extra_kwargs.update(
                {
                    "crispy_shifty_datetime_end": datetime_end,
                    "crispy_shifty_total_seconds": duration,
                }
            )
        instance, metadata = _get_instance_and_metadata(
            toolz.dicttoolz.keymap(
                lambda k: k.split("crispy_shifty_")[-1],
                toolz.dicttoolz.merge(extra_kwargs, kwargs),
            )
        )
        pdbfile_data = json.dumps(
            {
                "instance": collections.OrderedDict(sorted(instance.items())),
                "metadata": collections.OrderedDict(sorted(metadata.items())),
                "scores": collections.OrderedDict(sorted(scores.items())),
            }
        )
        # Write full .pdb record
        pdbstring_data = pdbstring + os.linesep + REMARK_FORMAT + pdbfile_data
        if compressed:
            with open(output_file, "wb") as f:
                f.write(bz2.compress(str.encode(pdbstring_data)))
        else:
            with open(output_file, "w") as f:
                f.write(pdbstring_data)
        if simulation_records_in_scorefile:
            scorefile_data = pdbfile_data
        else:
            scorefile_data = json.dumps(
                {
                    metadata["output_file"]: collections.OrderedDict(
                        sorted(scores.items())
                    ),
                }
            )
        # Write data to new scorefile per decoy
        with open(score_file, "w") as f:
            f.write(scorefile_data)


# Old with only metadata, no separation between instance and metadata
# def save_results(results: Any, kwargs: Dict[Any, Any], decoy_path: str, environment_file: str,
#                  simulation_name: str = '', compressed: bool = True, simulation_records_in_scorefile: bool = False
# ) -> None:
#     """Write results and kwargs to disk."""

#     DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
#     REMARK_FORMAT = "REMARK Crispy Shifty: "

#     # Parse and save results
#     for pdbstring, scores in _parse_results(results):
#         # kwargs = _process_kwargs(kwargs)
#         output_dir = _get_output_dir(decoy_dir=decoy_path)
#         decoy_name = "_".join([simulation_name, uuid.uuid4().hex])
#         output_file = os.path.join(output_dir, decoy_name + ".pdb")
#         output_scorefile = os.path.join(output_dir, decoy_name + ".json")
#         if compressed:
#             output_file += ".bz2"
#         extra_kwargs = {
#             "crispy_shifty_decoy_name": decoy_name,
#             "crispy_shifty_output_file": output_file,
#         }
#         if os.path.exists(environment_file):
#             extra_kwargs[
#                 "crispy_shifty_environment_file"
#             ] = environment_file
#         if "crispy_shifty_datetime_start" in kwargs:
#             datetime_end = datetime.now().strftime(DATETIME_FORMAT)
#             duration = str(
#                 (
#                     datetime.strptime(datetime_end, DATETIME_FORMAT)
#                     - datetime.strptime(
#                         kwargs["crispy_shifty_datetime_start"],
#                         DATETIME_FORMAT,
#                     )
#                 ).total_seconds()
#             )  # For build-in functions
#             extra_kwargs.update(
#                 {
#                     "crispy_shifty_datetime_end": datetime_end,
#                     "crispy_shifty_total_seconds": duration,
#                 }
#             )
#         metadata = toolz.dicttoolz.keymap(
#                 lambda k: k.split("crispy_shifty_")[-1],
#                 toolz.dicttoolz.merge(extra_kwargs, kwargs),
#         )
#         pdbfile_data = json.dumps(
#             {
#                 "metadata": collections.OrderedDict(sorted(metadata.items())),
#                 "scores": collections.OrderedDict(sorted(scores.items())),
#             }
#         )
#         # Write full .pdb record
#         pdbstring_data = pdbstring + os.linesep + REMARK_FORMAT + pdbfile_data
#         if compressed:
#             with open(output_file, "wb") as f:
#                 f.write(bz2.compress(str.encode(pdbstring_data)))
#         else:
#             with open(output_file, "w") as f:
#                 f.write(pdbstring_data)
#         if simulation_records_in_scorefile:
#             scorefile_data = pdbfile_data
#         else:
#             scorefile_data = json.dumps(
#                 {
#                     metadata["output_file"]: collections.OrderedDict(
#                         sorted(scores.items())
#                     ),
#                 }
#             )
#         # Write data to new scorefile per decoy
#         with open(output_scorefile, "w") as f:
#             f.write(scorefile_data)


def wrapper_for_array_tasks(func, args):
    
    import argparse
    import pyrosetta
    import sys

    parser = argparse.ArgumentParser(description="Parses arguments passed to the minimal run.py")
    # required task arguments
    parser.add_argument("-pdb_path", type=str, default="", nargs='*', required=True)
    # optional task arguments
    parser.add_argument("-options", type=str, default="", nargs='*', required=False)
    parser.add_argument("-extra_options", type=str, default="", nargs='*', required=False)
    parser.add_argument("-extra_kwargs", type=str, default="", nargs='*', required=False)
    # arguments tracked by pyrosettacluster. could add some of the others below in save_kwargs
    parser.add_argument("-instance", type=str, default="", nargs='*', required=True)
    
    args = parser.parse_args(sys.argv[1:])
    print("Design will proceed with the following options:")
    print(args)

    # The options strings are passed without the leading "-" so that argparse doesn't interpret them as arguments. Read them in,
    # assuming they are a list of key-value pairs where odd-indexed elements are keys and even-indexed elements are values. Add
    # in the leading "-" and pass them to pyrosetta.
    pyro_kwargs = {"options": {'-'+args.options[i]: args.options[i+1] for i in range(0, len(args.options), 2)}, 
                "extra_options": {'-'+args.extra_options[i]: args.extra_options[i+1] for i in range(0, len(args.extra_options), 2)}
                }
    # print(pyro_kwargs)
    pyrosetta.distributed.maybe_init(**pyro_kwargs)

    # Get kwargs to pass to the function from the extra kwargs
    func_kwargs = {args.extra_kwargs[i]: args.extra_kwargs[i+1] for i in range(0, len(args.extra_kwargs), 2)}
    instance_kwargs = {args.instance[i]: args.instance[i+1] for i in range(0, len(args.instance), 2)}

    for pdb_path in args.pdb_path:
        # Add the required kwargs
        func_kwargs["pdb_path"] = pdb_path
        # print(func_kwargs)

        datetime_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # Run the function
        pposes = func(**func_kwargs)
        
        # task_kwargs is everything that would be passed in a task in pyrosetta distributed.
        # This isn't a perfect way of figuring out which are which, but it's the best I can do here easily
        # without deviating too far.
        task_kwargs = func_kwargs
        task_kwargs.update(pyro_kwargs)
        save_kwargs = {'compressed': True,
                    'decoy_dir_name': 'decoys',
                    'score_dir_name': 'scores',
                    'environment': '',
                    'task': task_kwargs,
                    'output_path': '~',
                    'simulation_name': '',
                    'simulation_records_in_scorefile': False,
                    'crispy_shifty_datetime_start': datetime_start
                    }
        save_kwargs.update(instance_kwargs)
        #print(save_kwargs)
        
        save_results(pposes, save_kwargs)
    
def gen_array_tasks(
    distribute_func: str, 
    design_list_file: str, 
    output_path: str, 
    queue: str, 
    memory: str = '4G', 
    nstruct: int = 1,
    nstruct_per_task: int = 1,
    options: str = '', # options for pyrosetta initialization
    simulation_name: str = 'crispy_shifty',
    extra_kwargs: dict = {} # kwargs to pass to crispy_shifty_func. keys and values must be strings containing no spaces
    ):
    import os, stat
    from more_itertools import ichunked

    os.makedirs(output_path, exist_ok=True)

    # def create_tasks(design_list_file, options):
    #     with open(design_list_file, "r") as f:
    #         for line in f:
    #             tasks = {"-options": "corrections::beta_nov16 true"} # no dash in from of corrections- this is not a typo
    #             tasks["-extra_options"] = options
    #             tasks["-pdb_path"] = line.rstrip()
    #             yield tasks
    
    def create_tasks(design_list_file, options, nstruct_per_task):
        with open(design_list_file, "r") as f:
            # returns an iteratable with nstruct_per_task elements: lines of design_list_file
            for lines in ichunked(f, nstruct_per_task):
                tasks = {"-options": "corrections::beta_nov16 true"} # no dash in from of corrections- this is not a typo
                tasks["-extra_options"] = options
                 # join the lines of design_list_file with spaces, removing trailing newlines
                tasks["-pdb_path"] = ' '.join(line.rstrip() for line in lines)
                yield tasks

    jid = "{SLURM_JOB_ID%;*}"
    sid = "{SLURM_ARRAY_TASK_ID}p"

    slurm_dir = os.path.join(output_path, 'slurm_logs')
    os.makedirs(slurm_dir, exist_ok=True)

    tasklist = os.path.join(output_path, 'tasks.cmds')
    # run_sh = """#!/usr/bin/env bash \n#SBATCH -J crispy_shifty \n#SBATCH -e {slurm_dir}/crispy_shifty-%J.err \n#SBATCH -o {slurm_dir}/crispy_shifty-%J.out \n#SBATCH -p {queue} \n#SBATCH --mem={memory} \n\nJOB_ID=${jid} \nCMD=$(sed -n "${sid}" {tasklist}) \necho "${c}" | bash""".format(
    #     slurm_dir=slurm_dir, queue=queue, memory=memory, jid=jid, sid=sid, tasklist=tasklist, c="{CMD}"
    # )
    run_sh = f"""#!/usr/bin/env bash \n#SBATCH -J {simulation_name} \n#SBATCH -e {slurm_dir}/{simulation_name}-%J.err \n#SBATCH -o {slurm_dir}/{simulation_name}-%J.out \n#SBATCH -p {queue} \n#SBATCH --mem={memory} \n\nJOB_ID=${jid} \nCMD=$(sed -n "${sid}" {tasklist}) \necho "${{CMD}}" | bash"""
    run_sh_file = os.path.join(output_path, 'run.sh')
    with open(run_sh_file, "w+") as f:
        print(run_sh, file=f)
    st = os.stat(run_sh_file)
    os.chmod(run_sh_file, st.st_mode | stat.S_IEXEC)

    func_split = distribute_func.split('.')
    func_name = func_split[-1]
    run_py = f"""#!/usr/bin/env python\nimport sys\nsys.path.insert(0, "/home/broerman/projects/crispy_shifty")\nfrom crispy_shifty.utils.io import wrapper_for_array_tasks\nfrom {'.'.join(func_split[:-1])} import {func_name}\nwrapper_for_array_tasks({func_name}, sys.argv)"""
    run_py_file = os.path.join(output_path, 'run.py')
    with open(run_py_file, "w+") as f:
        print(run_py, file=f)
    st = os.stat(run_py_file)
    os.chmod(run_py_file, st.st_mode | stat.S_IEXEC)
    
    instance_dict = {'output_path': output_path,
                     'simulation_name': simulation_name
                     }

    instance_str = "-instance " + " ".join([" ".join([k, str(v)]) for k, v in instance_dict.items()])
    extra_kwargs_str = "-extra_kwargs " + " ".join([" ".join([k, str(v)]) for k, v in extra_kwargs.items()])

    with open(tasklist, "w+") as f:
        for i in range(0, nstruct):
            for tasks in create_tasks(design_list_file, options, nstruct_per_task):
                task_str = " ".join([" ".join([k, str(v)]) for k, v in tasks.items()])
                cmd = f"{run_py_file} {task_str} {extra_kwargs_str} {instance_str}"
                print(cmd, file=f)

    # Let's go
    print("Run the following command with your desired environment active:")
    print(f"sbatch -a 1-$(cat {tasklist} | wc -l) {run_sh_file}")

def collect_score_file(output_path: str, score_dir_name: str = 'scores'):
    import os
    from glob import iglob
    score_dir = os.path.join(output_path, score_dir_name)
    with open(os.path.join(output_path, 'scores.json'), 'w') as scores_file:
        for score_file in iglob(os.path.join(score_dir, '*', '*.json')):
            with open(score_file, 'r') as f:
                scores_file.write(f.read() + '\n')
        

from pyrosetta.distributed import requires_init
@requires_init
def test_func(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Generator[PackedPose, PackedPose, None]:

    # test env
    import pycorn

    import sys
    import pyrosetta.distributed.io as io

    sys.path.insert(0, "/mnt/home/broerman/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

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
        print('hi!')

        ppose = io.to_packed(pose)
        yield ppose
