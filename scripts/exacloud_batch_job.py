import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import logging

log = logging.getLogger(__name__)


if __name__ == '__main__':
    """Creates a slurm batch file to run. 
    
    Batch files are saved in the user job_history directory in /lustre1/LBHB.
    """
    # parse arguments in order to collect all args into list
    parser = argparse.ArgumentParser(description='Run jobs on exacloud!')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    job_dir = Path.home() / 'job_history'
    # create job dir if doesn't exist
    job_dir.mkdir(exist_ok=True, parents=True)

    job_file_name = datetime.now().strftime('%Y-%m-%d-T%H%M%S') + '.sh'
    job_file_loc = job_dir / job_file_name

    with open(job_file_loc, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --account=lbhb\n')
        # f.write('#SBATCH --time=2:00:00\n')
        f.write('#SBATCH --partition=gpu\n')
        # f.write('#SBATCH --mem=4G\n')
        # f.write('#SBATCH --gres=disk:5G\n')
        # f.write('#SBATCH --job-name=nems\n')
        f.write(' '.join(['srun'] + args.arguments))
        # f.write('\n')

    subprocess.run(['sbatch', str(job_file_loc)])
