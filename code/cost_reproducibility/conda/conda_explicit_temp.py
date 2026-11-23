#! /usr/bin/env python
import argparse
import tempfile
import subprocess
import os

def temp_conda_explicit_run(python_file, env_file, requirements_file,  deterministic):
    """ Create a temporary conda environment, run a command inside it, and clean up. """
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = os.path.join(tmpdir, "env")
        subprocess.run(["conda", "create", "--quiet", "--prefix", env_path, "--file", env_file], check=True)
        subprocess.run(["conda", "run", "--prefix", env_path, "python3", "-m", "pip", "install", "-r", requirements_file], check=True)
        cmd = ["conda", "run", "--prefix", env_path, "python3", python_file, "conda"]
        if deterministic:
            cmd += ["--deterministic"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run command in a temporary conda environment.")
    parser.add_argument("file", help="Python file to run inside the environment.")
    parser.add_argument("--env-file", help="Path to environment.yml")
    parser.add_argument("--requirements-file", help="Path to requirements.txt")
    parser.add_argument("--deterministic", help="Whether to use deterministic torch or not.", action='store_true')


    args = parser.parse_args()

    temp_conda_explicit_run(args.file, args.env_file, args.requirements_file, args.deterministic)
