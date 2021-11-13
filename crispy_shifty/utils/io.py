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
import copy
import json
import os
import pyrosetta.distributed.io as io
import uuid

from datetime import datetime
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.distributed.cluster.exceptions import OutputError
from pyrosetta.distributed.packed_pose.core import PackedPose

"""Input/Output methods for PyRosettaCluster."""

DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"
REMARK_FORMAT: str = "REMARK PyRosettaCluster: "

# TODO finish transitioning this function
# this will need to be done in conjunction with your wrapper functions for arrayjobs, since
# then you'll decide what kwargs to keep track of
def _get_instance_and_metadata(
    self, kwargs: Dict[Any, Any]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    Get the current state of the PyRosettaCluster instance, and split the
    kwargs into the PyRosettaCluster instance kwargs and ancillary metadata.
    """

    instance_state = dict(zip(self.__slots__, self.__getstate__()))
    instance_state.pop("client", None)
    instance_kwargs = copy.deepcopy(instance_state)
    for i in self.__attrs_attrs__:
        if not i.init:
            instance_kwargs.pop(i.name)
        if i.name == "input_packed_pose":
            instance_kwargs.pop(i.name, None)
    instance_kwargs["tasks"] = kwargs.pop("task")
    for option in ["extra_options", "options"]:
        if option in instance_kwargs["tasks"]:
            instance_kwargs["tasks"][option] = pyrosetta.distributed._normflags(
                instance_kwargs["tasks"][option]
            )
    instance_kwargs["seeds"] = kwargs.pop("seeds")
    instance_kwargs["decoy_ids"] = kwargs.pop("decoy_ids")

    return instance_kwargs, kwargs

def _get_output_dir(decoy_dir: str) -> str:
    """Get the output directory in which to write files to disk."""
    
    zfill_value = 4
    max_dir_depth = 1000
    decoy_dir_list = os.listdir(decoy_dir)
    if not decoy_dir_list:
        new_dir = str(0).zfill(zfill_value)
        output_dir = os.path.join(decoy_dir, new_dir)
        os.mkdir(output_dir)
    else:
        top_dir = list(reversed(sorted(decoy_dir_list)))[0]
        if len(os.listdir(os.path.join(decoy_dir, top_dir))) < max_dir_depth:
            output_dir = os.path.join(decoy_dir, top_dir)
        else:
            new_dir = str(int(top_dir) + 1).zfill(zfill_value)
            output_dir = os.path.join(decoy_dir, new_dir)
            os.mkdir(output_dir)

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

def save_results(results: Any, kwargs: Dict[Any, Any], decoy_path: str, environment_file: str,
                 simulation_name: str = '', compressed: bool = True, simulation_records_in_scorefile: bool = False
) -> None:
    """Write results and kwargs to disk."""

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    REMARK_FORMAT = "REMARK Crispy Shifty: "

    # Parse and save results
    for pdbstring, scores in _parse_results(results):
        # kwargs = _process_kwargs(kwargs)
        output_dir = _get_output_dir(decoy_dir=decoy_path)
        decoy_name = "_".join([simulation_name, uuid.uuid4().hex])
        output_file = os.path.join(output_dir, decoy_name + ".pdb")
        output_scorefile = os.path.join(output_dir, decoy_name + ".json")
        if compressed:
            output_file += ".bz2"
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
                lambda k: k.split("PyRosettaCluster_")[-1],
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
        with open(output_scorefile, "w") as f:
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
