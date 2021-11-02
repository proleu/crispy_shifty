# Python standard library
from glob import glob
import os
import socket
import sys

# 3rd party library imports
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# Rosetta library imports
from pyrosetta.distributed.cluster.core import PyRosettaCluster

# Custom library imports
os.chdir("/home/broerman/projects/crispy_shifty")
sys.path.insert(0, os.getcwd())
from crispy_shifty.protocols.design import one_state_design_unlooped_dimer  # the functions we will distribute

print(f"running in directory: {os.getcwd()}")  # where are we?
print(f"running on node: {socket.gethostname()}")  # what node are we on?
print(f"view dashboard at http://{socket.gethostname()}:8787")

def create_tasks(selected, options):
    with open(selected, "r") as f:
        for line in f:
            tasks = {"options": "-corrections::beta_nov16 true"}
            tasks["extra_options"] = options
            tasks["pdb_path"] = line.rstrip()
            yield tasks

selected = os.path.join(os.getcwd(), "projects/crispy_shifty_dimers/01_make_states/states.list")

options = {
    "-out:level": "100",
    "-holes:dalphaball": "/home/bcov/ppi/tutorial_build/main/source/external/DAlpahBall/DAlphaBall.gcc",
    "-indexed_structure_store:fragment_store": "/net/databases/VALL_clustered/connect_chains/ss_grouped_vall_helix_shortLoop.h5",
    "-precompute_ig": "true"
}

output_path = os.path.join(os.getcwd(), "projects/crispy_shifty_dimers/02_design_filter")
os.makedirs(output_path, exist_ok=True)

if __name__ == "__main__":
    # configure SLURM cluster as a context manager
    with SLURMCluster(
        cores=1,
        processes=1,
        job_cpu=1,
        memory="10GB",
        queue="backfill",
        walltime="11:30:00",
        death_timeout=120,
        local_directory="$TMPDIR",  # spill worker litter on local node temp storage
        log_directory=os.path.join(output_path, "slurm_logs"),
        extra=["--lifetime", "11h", "--lifetime-stagger", "5m"],
    ) as cluster:
        print(cluster.job_script())
        # scale between 1-300 workers,
        cluster.adapt(
            minimum=1,
            maximum=300,
            wait_count=999,  # Number of consecutive times that a worker should be suggested for removal it is removed
            interval="5s",  # Time between checks
            target_duration="60s",
        )
        # setup a client to interact with the cluster as a context manager
        with Client(cluster) as client:
            print(client)
            client.upload_file(
                os.path.join(os.getcwd(), "crispy_shifty/protocols/design.py")
            )  # upload the script that contains the functions to distribute
            PyRosettaCluster(
                client=client,
                logging_level="WARNING",
                output_path=output_path,
                project_name="crispy_shifty_dimers",
                scratch_dir=output_path,
                simulation_name="02_design_filter",
                tasks=create_tasks(selected, options),
                nstruct=10,
                ignore_errors=True # for large runs so that the head process doesn't die due to a rare segfault in one of its children
            ).distribute(protocols=[one_state_design_unlooped_dimer])
            client.close()
        cluster.scale(0)
        cluster.close()
    print("distributed run complete")