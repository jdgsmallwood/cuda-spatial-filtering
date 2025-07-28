import mlflow
import argparse
import os
from datetime import datetime
from loguru import logger


def parse_ncu_file(filepath):
    # Stub for Nsight Compute parsing logic
    # Implement actual parsing as needed
    return {"ncu_metric_example": 42}


def parse_nsys_file(filepath):
    # Stub for Nsight Systems parsing logic
    # Implement actual parsing as needed
    return {"nsys_metric_example": 99}


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("2025-cuda-xb-benchmarks")

    parser = argparse.ArgumentParser(
        description="Parse Nsight Compute/System report and log to MLflow."
    )
    parser.add_argument(
        "run_description", type=str, help="Description of the profiling run"
    )
    parser.add_argument(
        "filepath", type=str, help="Filepath to .ncu-rep or .nsys-rep file"
    )
    parser.add_argument(
        "--NR_CHANNELS", type=int, required=True, help="Number of channels"
    )
    parser.add_argument(
        "--NR_RECEIVERS", type=int, required=True, help="Number of receivers"
    )
    parser.add_argument(
        "--NR_PACKETS_FOR_CORRELATION",
        type=int,
        required=True,
        help="Packets for correlation",
    )
    parser.add_argument("--TARGET", type=str, required=True, help="Target name")
    parser.add_argument(
        "--NR_CORRELATION_PACKETS_TO_INTEGRATE",
        type=int,
        required=True,
        help="Correlation packets to integrate",
    )
    parser.add_argument("--NR_BITS", type=int, required=True, help="Number of bits")

    args = parser.parse_args()

    description = f"""
    **Description**
    {args.run_description}
    """

    # Decide parser based on file extension
    file_ext = os.path.splitext(args.filepath)[1]
    if file_ext == ".ncu-rep":
        metrics = parse_ncu_file(args.filepath)
    elif file_ext == ".nsys-rep":
        metrics = parse_nsys_file(args.filepath)
    else:
        logger.error("Unsupported file type. Supported: .ncu-rep, .nsys-rep")
        return

    with mlflow.start_run(description=description) as run:
        logger.info("Starting run and logging parameters.")

        # Log experimental parameters
        mlflow.log_param("NR_CHANNELS", args.NR_CHANNELS)
        mlflow.log_param("NR_RECEIVERS", args.NR_RECEIVERS)
        mlflow.log_param("NR_PACKETS_FOR_CORRELATION", args.NR_PACKETS_FOR_CORRELATION)
        mlflow.log_param("TARGET", args.TARGET)
        mlflow.log_param(
            "NR_CORRELATION_PACKETS_TO_INTEGRATE",
            args.NR_CORRELATION_PACKETS_TO_INTEGRATE,
        )
        mlflow.log_param("NR_BITS", args.NR_BITS)

        # Log parsed metrics (stub metrics here)
        mlflow.log_metrics(metrics)

        # Log the report file as an artifact
        mlflow.log_artifact(args.filepath)

        logger.info(f"✅ MLflow run completed: {run.info.run_id}")


if __name__ == "__main__":
    main()
