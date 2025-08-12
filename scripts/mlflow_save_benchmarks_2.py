import uuid
import mlflow
import subprocess
import json
import pandas as pd
import os
from loguru import logger


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        return "unknown"


REMOTE_HOST = ""


def run_benchmarks_and_save(params: dict, run_description: str):
    local = not REMOTE_HOST
    logger.info("Setting up MlFlow")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("2025-xbengine-benchmarks")

    job_id = str(uuid.uuid4())
    logger.info(f"job_id: {job_id}")

    LOCAL_OUTPUT_DIR = job_id
    subprocess.run(["mkdir", job_id])

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    SLURM_FILE_NAME = "submit_job.sh"
    ncu_dataset = "~/projects/output_10.pcap"
    benchmark_file_name = f"benchmarks_{job_id}.json"

    params["GENERATED_INPUT_FILE_NAME"] = f"input_{job_id}.pcap"
    params["REMOTE_PATH"] = "/home/jsmallwo/projects/cuda-spatial-filtering/build"
    params["REMOTE_EXEC"] = (
        f"./beamform_spatial {params['REMOTE_PATH']}/{job_id}/{params['GENERATED_INPUT_FILE_NAME']} {benchmark_file_name}"
    )
    params["REMOTE_EXEC_NCU"] = (
        f"./beamform_spatial {ncu_dataset} {benchmark_file_name}.ncu"
    )
    params["PROFILE_OUTPUT"] = f"profile_output_{job_id}"
    params["NSYS_PROFILE_OUTPUT"] = f"nsys_profile_output_{job_id}"
    params["JOB_OUTPUT_FILE_NAME"] = f"output_{job_id}.txt"

    logger.info(f"PARAMS:\n")
    logger.info(params)

    logger.info("Creating input dataset....")
    cmd = (
        "source .venv/bin/activate && "
        f"python create_pcap.py --output {job_id}/{params['GENERATED_INPUT_FILE_NAME']} "
        f"--number_receivers {params['NR_RECEIVERS']} "
        f"--number_packets {params['NR_PACKETS_TOTAL']} "
        f"--number_channels {params['NR_CHANNELS']}"
    )
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

    if local:
        subprocess.run(["mkdir", f"{params['REMOTE_PATH']}/{job_id}"])
        subprocess.run(
            [
                "cp",
                os.path.join(job_id, params["GENERATED_INPUT_FILE_NAME"]),
                f"{params['REMOTE_PATH']}/{job_id}/{params['GENERATED_INPUT_FILE_NAME']}",
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "rsync",
                "-avz",
                os.path.join(job_id, params["GENERATED_INPUT_FILE_NAME"]),
                f"{REMOTE_HOST}:{params['REMOTE_PATH']}/{job_id}/",
            ],
            check=True,
        )

    logger.info("Creating slurm script")
    slurm_script = f"""#!/bin/bash
#
#SBATCH --job-name=profile
#SBATCH --output={params["JOB_OUTPUT_FILE_NAME"]}
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=4g
#SBATCH --gres=gpu:1

srun apptainer exec --nv /fred/oz002/jsmallwo/apptainer.sif /bin/bash -c "cd {params["REMOTE_PATH"]} && \\
    cmake -DBUILD_TESTING=OFF -DBENCHMARKING=1 -DBUILD_TARGET=LAMBDA -DNR_CHANNELS={params["NR_CHANNELS"]} -DNR_RECEIVERS={params["NR_RECEIVERS"]} .. && cmake --build . && \\
    cd apps && {params["REMOTE_EXEC"]}  &&
    ncu -f --set full --target-processes all --export {params["PROFILE_OUTPUT"]} {params["REMOTE_EXEC_NCU"]} && \\
    ncu --import {params["PROFILE_OUTPUT"]}.ncu-rep --csv --page details > {params["PROFILE_OUTPUT"]}.csv && \\
    nsys profile -t cuda,nvtx -o {params["NSYS_PROFILE_OUTPUT"]} --stats=true --force-overwrite true {params["REMOTE_EXEC"]}"
    """
    logger.info("Writing to file")
    # Write to file
    with open("submit_job.sh", "w") as f:
        f.write(slurm_script)

    logger.info("Slurm script written to 'submit_job.sh'")

    # === Step 6: Sync slurm script to server ===
    if not local:
        logger.info("Syncing slurm script to remote...")
        subprocess.run(
            ["rsync", "-avz", SLURM_FILE_NAME, f"{REMOTE_HOST}:{params['REMOTE_PATH']}"]
        )
    else:
        subprocess.run(
            [
                "cp",
                SLURM_FILE_NAME,
                os.path.join(params["REMOTE_PATH"], SLURM_FILE_NAME),
            ],
            check=True,
        )

    # === Step 7: Compile and profile remotely ===
    logger.info("Submitting slurm job...")
    if local:
        subprocess.run(
            f"cd {params['REMOTE_PATH']} && sbatch -W submit_job.sh",
            shell=True,
            check=True,
        )
    else:
        subprocess.run(
            f'ssh {REMOTE_HOST} -t "cd {params["REMOTE_PATH"]} && sbatch -W submit_job.sh"',
            shell=True,
        )

    logger.info("Pulling back results...")
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    if local:
        subprocess.run(
            [
                "cp",
                f"{params['REMOTE_PATH']}/apps/{params['PROFILE_OUTPUT']}.csv",
                os.path.join(LOCAL_OUTPUT_DIR, f"{params['PROFILE_OUTPUT']}.csv"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "cp",
                f"{params['REMOTE_PATH']}/apps/{params['PROFILE_OUTPUT']}.ncu-rep",
                os.path.join(LOCAL_OUTPUT_DIR, f"{params['PROFILE_OUTPUT']}.ncu-rep"),
            ],
            check=True,
        )

        subprocess.run(
            [
                "cp",
                f"{params['REMOTE_PATH']}/apps/{params['NSYS_PROFILE_OUTPUT']}.nsys-rep",
                os.path.join(
                    LOCAL_OUTPUT_DIR, f"{params['NSYS_PROFILE_OUTPUT']}.nsys-rep"
                ),
            ],
            check=True,
        )

        subprocess.run(
            [
                "cp",
                f"{params['REMOTE_PATH']}/{params['JOB_OUTPUT_FILE_NAME']}",
                os.path.join(LOCAL_OUTPUT_DIR, f"{params['JOB_OUTPUT_FILE_NAME']}"),
            ],
            check=True,
        )

        subprocess.run(
            [
                "cp",
                f"{params['REMOTE_PATH']}/apps/{benchmark_file_name}",
                os.path.join(LOCAL_OUTPUT_DIR, benchmark_file_name),
            ],
            check=True,
        )

    else:
        subprocess.run(
            [
                "rsync",
                "-avz",
                f"{REMOTE_HOST}:{params['REMOTE_PATH']}/apps/{params['PROFILE_OUTPUT']}.csv",
                LOCAL_OUTPUT_DIR,
            ],
            check=True,
        )

        subprocess.run(
            [
                "rsync",
                "-avz",
                f"{REMOTE_HOST}:{params['REMOTE_PATH']}/apps/{params['PROFILE_OUTPUT']}.ncu-rep",
                LOCAL_OUTPUT_DIR,
            ],
            check=True,
        )

        subprocess.run(
            [
                "rsync",
                "-avz",
                f"{REMOTE_HOST}:{params['REMOTE_PATH']}/apps/{params['NSYS_PROFILE_OUTPUT']}.nsys-rep",
                LOCAL_OUTPUT_DIR,
            ],
            check=True,
        )

        subprocess.run(
            [
                "rsync",
                "-avz",
                f"{REMOTE_HOST}:{params['REMOTE_PATH']}/{params['JOB_OUTPUT_FILE_NAME']}",
                LOCAL_OUTPUT_DIR,
            ],
            check=True,
        )

        subprocess.run(
            [
                "rsync",
                "-avz",
                f"{REMOTE_HOST}:{params['REMOTE_PATH']}/apps/{benchmark_file_name}",
                LOCAL_OUTPUT_DIR,
            ],
            check=True,
        )

    local_rep_path = os.path.join(LOCAL_OUTPUT_DIR, f"{params['PROFILE_OUTPUT']}.csv")

    logger.info("Starting MLFlow run...")
    description = f"""
    **Description**
    {run_description}
    
    """

    with mlflow.start_run(description=description) as run:
        profile_path = f"{os.path.join(LOCAL_OUTPUT_DIR, params['PROFILE_OUTPUT'])}.csv"
        data = pd.read_csv(profile_path)
        # model_params = extract_parameters_from_csv(data)
        logger.info("Logging parameters...")
        # mlflow.log_params(model_params)
        mlflow.log_param("git_commit_hash", get_git_commit())
        mlflow.log_params(params)
        # mlflow.log_param("benchmark_dataset", benchmark_dataset)

        logger.info("Logging metrics...")

        with open(os.path.join(LOCAL_OUTPUT_DIR, benchmark_file_name), "r") as f:
            timings = json.load(f)
            timings["beamforming_duration_us"] = (
                timings["checkpoint_end_beamforming"]
                - timings["checkpoint_begin_beamforming"]
            )
            mlflow.log_metrics(timings)

        mlflow.log_artifact(profile_path)
        # log the original ncu-rep file as well.
        mlflow.log_artifact(profile_path.replace(".csv", ".ncu-rep"))
        mlflow.log_artifact(
            os.path.join(LOCAL_OUTPUT_DIR, f"{params['NSYS_PROFILE_OUTPUT']}.nsys-rep")
        )
        mlflow.log_artifact(os.path.join(LOCAL_OUTPUT_DIR, benchmark_file_name))
        mlflow.log_artifact(
            os.path.join(LOCAL_OUTPUT_DIR, params["JOB_OUTPUT_FILE_NAME"])
        )

        logger.info(f"✅ MLflow run completed: {run.info.run_id}")

    logger.info("Cleaning up....")
    if not local:
        subprocess.run(
            f"ssh {REMOTE_HOST} rm -r {params['REMOTE_PATH']}/{job_id}/", shell=True
        )
    else:
        subprocess.run(["rm", "-r", f"{params['REMOTE_PATH']}/{job_id}"], check=True)
    subprocess.run(f"rm -r {job_id}", shell=True)


if __name__ == "__main__":
    run_description = input()

    for channels in [500, 5000]:
        for packets in [500]:
            params = {}

            params["NR_CHANNELS"] = channels
            params["NR_RECEIVERS"] = 32
            params["NR_BITS"] = 8
            params["NR_POLARIZATIONS"] = 2
            params["NR_PACKETS_FOR_CORRELATION"] = 16
            params["NR_TIME_STEPS_PER_PACKET"] = 64
            params["NR_RECEIVERS_PER_BLOCK"] = 32
            params["NR_BUFFERS"] = 2
            params["NR_CORRELATIONS_BLOCKS_TO_INTEGRATE"] = 10
            params["NR_PACKETS_TOTAL"] = packets

            logger.info(f"PARAMS:\n")
            logger.info(params)

            run_benchmarks_and_save(params, run_description)
