import argparse
from pathlib import Path

from paramiko import SSHClient, AutoAddPolicy


class ExacloudConnection:
    """Resource manager for exacloud commands.

    Helpful exacloud commands:

    - View jobs in queue by user:
        $ squeue -u <user>
    - View all jobs by user (default is jobs since midnight, add in date to see further back):
        $ sacct -X -u <user> [--startime YYYY-MM-DD]
    - Above but with better formatting (assuming full width terminal window):
        $ sacct -X -u <user> --format "JobID,Elapsed,Timelimit,ReqMem,MaxVMSize,State,NodeList,Partition,JobName%25,Comment%125"
    - Cancel a job
        $ scancel <jobid>
    - Cancel all jobs by user:
        $ scancel -u <user>
    """
    def __init__(self, user, ssh_key=None, password=None):
        """Prepares the SSH connection paramters.

        :param user: Exacloud username.
        :param command: Command to be sent to exacloud.
        :param ssh_key: Path to key. Required if password is None.
        :param password: Exacloud password. Required if ssh_key is None.
        """
        self.host = 'exahead1.ohsu.edu'
        self.user = user

        # setup credentials
        if password is None and ssh_key is None:
            raise ValueError('One of password or ssh_key should be passed.')
        # if both included, default to ssh key
        elif ssh_key is not None:
            assert Path(ssh_key).exists(), f'Could not find ssh key "{ssh_key}". Make sure it exists and '\
                                           f'is located adjacent to the public key.'
            self.creds = {'key_filename': ssh_key}
        else:
            self.creds = {'password': password}

    def __enter__(self):
        """Makes the SSH connection."""
        self.client = SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(AutoAddPolicy())

        self.client.connect(self.host, username=self.user, **self.creds)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the SSH connection."""
        self.client.close()

    def send_command(self, command):
        """Sends the command.

        :param command: The command to send.
        """
        stdin, stdout, stderr = self.client.exec_command(command)
        return stdin, stdout, stderr

    def cancel_all_user_jobs(self):
        """Cancels all jobs by the user."""
        self.send_command(f'scancel -u {self.user}')

    def cancel_pending_user_jobs(self):
        """Cancels all jobs pending by the user."""
        self.send_command(f'scancel -u {self.user} --state=pending')


def enqueue_exacloud_job(*, user, cellid='', batch='', modelname='', exec_path=None, script_path=None, password=None,
                         ssh_key=None):
    """Queues a job on the exacloud cluster. All arguments must be named.

    One of either password or ssh_key is required. Defaults to ssh_key if both passed.

    :param user: OHSU username with access to exacloud.
    :param password: OHSU password for logging into exacloud.
    :param ssh_key: Location of ssh key associated with exacloud.
    :param exec_path: Python executable location.
    :param script_path: Python script to call.
    :param cellid: NEMS cellid, must be found on Baphy.
    :param batch: NEMS batch number.
    :param modelname: NEMS modelname.
    """
    if password is None and ssh_key is None:
        raise ValueError('One of either password or ssh_key must be passed.')
    elif ssh_key is not None:
        ssh_key = Path(ssh_key)
        if not ssh_key.exists():
            raise FileNotFoundError(f'Could not find ssh key "{str(args.ssh_key)}". Make sure it exists and ' \
                                    f'is located adjacent to the public key.')
        creds = {'ssh_key': str(ssh_key)}
    else:
        creds = {'password': password}

    if exec_path is None:
        exec_path = Path(r'/home/exacloud/lustre1/LBHB/code/python-envs/nems-gpu/bin/python')
    if script_path is None:
        script_path = Path(r'/home/exacloud/lustre1/LBHB/code/NEMS/scripts/fit_single.py')

    # default srun params for now
    batch_maker = Path(r'/home/exacloud/lustre1/LBHB/code/NEMS/scripts/exacloud_batch_job.py')

    command = ' '.join([str(exec_path),
                        str(batch_maker),
                        str(exec_path),
                        str(script_path),
                        cellid,
                        str(batch),
                        modelname])

    with ExacloudConnection(user, **creds) as exa:
        stdin, stdout, stderr = exa.send_command(command)

    return stdin, stdout, stderr


if __name__ == '__main__':
    # parse arguments using argparse in order to have defaults and help text
    parser = argparse.ArgumentParser(description='Run jobs on exacloud!')

    # ssh stuff
    ssh_group = parser.add_argument_group('SSH Arguments')

    ssh_group.add_argument('--user', required=True, type=str, help='Your OHSU username with access to exacloud.')
    # need either password or ssh key location
    cred_group = ssh_group.add_mutually_exclusive_group(required=True)
    cred_group.add_argument('--password', type=str, help='OHSU password for logging into exacloud.')
    cred_group.add_argument('--ssh_key', type=Path, help='Location of your ssh key associated with exacloud.')

    # execution arguments
    exec_group = parser.add_argument_group('Script execution')

    exec_default = Path(r'/home/exacloud/lustre1/LBHB/code/python-envs/nems-gpu/bin/python')
    exec_group.add_argument('--exec_path', type=Path, help='Python executable location, defaults to standard nems.',
                            default=exec_default)
    script_default = Path(r'/home/exacloud/lustre1/LBHB/code/NEMS/scripts/fit_single.py')
    exec_group.add_argument('--script_path', type=Path, help='Python script to call, defaults to "fit_single.py".',
                            default=script_default)

    # nems info
    nems_group = parser.add_argument_group('NEMS arguments')

    nems_group.add_argument('--cellid', required=True, type=str, help='NEMS cellid, must be found on Baphy.')
    nems_group.add_argument('--batch', required=True, type=str, help='NEMS batch number.')
    nems_group.add_argument('--modelname', required=True, type=str, help='NEMS modelname.')

    args = parser.parse_args()

    if args.ssh_key is not None:
        if not args.ssh_key.exists():
            raise FileNotFoundError(f'Could not find ssh key "{str(args.ssh_key)}". Make sure it exists and ' \
                                    f'is located adjacent to the public key.')

    enqueue_exacloud_job(user=args.user, ssh_key=args.ssh_key, password=args.password, exec_path=args.exec_path,
                         script_path=args.script_path, cellid=args.cellid, batch=args.batch, modelname=args.modelname)
