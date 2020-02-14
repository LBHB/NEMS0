import argparse
import subprocess
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    """Creates a slurm batch file to run. Batch files are saved in the user job_history directory in /lustre1/LBHB.
    
    Expects several arguments:
      - "exec_path": The executable path (i.e. Python install to call).
      - "exec_script": The script invoked by the executable (i.e. 'fit_single.py' or other).
      
    Any arguments included after are appended to the call. 
    """
    # parse arguments using argparse in order to have defaults and help text
    parser = argparse.ArgumentParser(description='Run jobs on exacloud!')

    # execution arguments
    exec_group = parser.add_argument_group('Script execution')

    exec_default = Path(r'/home/exacloud/lustre1/LBHB/code/python-envs/nems-gpu/bin/python')
    exec_group.add_argument('--exec_path', type=Path, help='Python executable location, defaults to standard nems.',
                            default=exec_default)
    script_default = Path(r'/home/exacloud/lustre1/LBHB/code/NEMS/scripts/fit_single.py')
    exec_group.add_argument('--script_path', type=Path, help='Python script to call, defaults to "fit_single.py".',
                            default=script_default)

    # script arguments
    script_group = parser.add_argument_group('Script arguments')
    script_group.add_argument('arguments', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    job_dir = Path.home() / 'job_history'
    # create job dir if doesn't exist
    job_dir.mkdir(exist_ok=True, parents=True)

    job_file_name = datetime.now().isoformat(timespec='seconds') + \
                    '_'.join(args.arguments) + '.sh'
    job_file_loc = job_dir / job_file_name

    with open(job_file_loc, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --account=lbhb\n')
        f.write('#SBATCH --time=2:00:00\n')
        f.write('#SBATCH --partition=gpu\n')
        f.write('#SBATCH --cpus-per-task=1\n')
        f.write('#SBATCH --mem=4G\n')
        f.write('#SBATCH --gres=disk:5G\n')
        f.write('#SBATCH --job-name=nems\n')
        f.writelines(f'srun {args.exec_path} {args.script_path}')
        f.write(' '.join(['srun', args.exec_path, args.script_path] + args.arguments))

    subprocess.run(f'sbatch {str(job_file_loc)}')
