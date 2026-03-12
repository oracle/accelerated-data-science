# SLURM SSH Magics (ads.slurm.magics)

Interact with a remote SLURM cluster from a notebook using simple IPython magics. Commands are executed over SSH to your SLURM login node, so you do not need SLURM or MUNGE installed locally.

## Motivation
- Run common SLURM commands and submit jobs directly from notebooks.
- Keep your workflow in one place while leveraging your existing SLURM cluster.
- Avoid local SLURM/MUNGE setup; everything is executed remotely via SSH.

## Prerequisites
- Network and SSH access to a SLURM login node.
- A private key that can authenticate to the login node (and optionally a bastion/jump host).
- SLURM installed and available on the remote login node (squeue, sacct, sinfo, scancel, sbatch).
- The remote SLURM configuration file path (commonly `/etc/slurm/slurm.conf`).

## Configuration (environment variables and %slurm_config)
You can configure the extension either by setting environment variables or at runtime with `%slurm_config`.

Environment variables (with defaults used by the extension):
- `SLURM_SSH_HOST` (no default): SSH target, e.g. `user@login-node`
- `SLURM_SSH_KEY` (default `~/.ssh/id_rsa`): Path to private key for SSH
- `SLURM_CONF_REMOTE` (default `/etc/slurm/slurm.conf`): Remote SLURM conf path
- `SLURM_SSH_JUMP` (optional): Bastion/jump host, e.g. `user@bastion`
- `SLURM_LUSTRE_BASE` (optional): Lustre base directory, e.g. `/lustre/projectX/$USER`
- `SLURM_LUSTRE_WORK` (optional): Overrides work dir; defaults to `$SLURM_LUSTRE_BASE/work` if not set
- `SLURM_LUSTRE_LOGS` (optional): Overrides logs dir; defaults to `$SLURM_LUSTRE_BASE/logs` if not set

Example (Python) to set env vars in a notebook:
```python
import os
os.environ["SLURM_SSH_HOST"] = "user@login-node"
os.environ["SLURM_SSH_KEY"] = os.path.expanduser("~/.ssh/id_rsa")
os.environ["SLURM_CONF_REMOTE"] = "/etc/slurm/slurm.conf"
# Optional:
# os.environ["SLURM_SSH_JUMP"] = "user@bastion"
```

Runtime configuration:
```
%slurm_config --host user@login-node --key ~/.ssh/id_rsa --conf /etc/slurm/slurm.conf --jump user@bastion
%slurm_config --lustre-base /lustre/projectX/$USER --lustre-work /lustre/projectX/$USER/work --lustre-logs /lustre/projectX/$USER/logs
```

The extension prints the effective configuration after running `%slurm_config`.

## Commands (magics)
Load the extension:
```
%load_ext ads.slurm.magics
```
If your environment exposes the module under a different name, you may also try:
```
%load_ext slurm.magics
```

Available magics:
- `%slurm_help`  
  Prints built-in help, environment variables, and examples.
- `%squeue [args]`  
  Runs `squeue` remotely with provided arguments.
- `%sacct [args]`  
  Runs `sacct` remotely with provided arguments.
- `%sinfo [args]`  
  Runs `sinfo` remotely with provided arguments.
- `%scancel <JOBID>`  
  Cancels the given job ID.
- `%%sbatch [sbatch args]` (cell magic)  
  Submits the cell content as a job script. Shebang and `#SBATCH` lines must start at column 0 (no indentation).  
  If no extra args are provided to `%%sbatch`, defaults are applied. If Lustre variables are configured, defaults are `--chdir=$SLURM_LUSTRE_WORK --output=$SLURM_LUSTRE_LOGS/%x-%j.out --error=$SLURM_LUSTRE_LOGS/%x-%j.err`; otherwise `--chdir=$HOME --output=slurm-%j.out`.  
  The extension parses the job ID from `sbatch --parsable` output and prints `Submitted JOBID=<id>` on success.

Notes:
- All actions are executed over SSH on the remote login node.
- The extension sets `SLURM_CONF` in the remote command context based on `SLURM_CONF_REMOTE`.
- Temporary scripts for `%%sbatch` are uploaded via `scp` to the login node and cleaned up after submission. If Lustre paths are configured, the extension will create the work/log directories on the remote (`mkdir -p`) before submission.

## Examples
Load and get help:
```
%load_ext ads.slurm.magics
%slurm_help
```

Configure via env:
```python
import os
os.environ["SLURM_SSH_HOST"] = "user@login-node"
os.environ["SLURM_SSH_KEY"] = os.path.expanduser("~/.ssh/id_rsa")
os.environ["SLURM_CONF_REMOTE"] = "/etc/slurm/slurm.conf"
# os.environ["SLURM_SSH_JUMP"] = "user@bastion"
```

Or configure at runtime:
```
%slurm_config --host user@login-node --key ~/.ssh/id_rsa --conf /etc/slurm/slurm.conf --jump user@bastion
```

Inspect cluster state:
```
%sinfo
%squeue -u $USER
```

Submit a minimal job:
```
%%sbatch
#!/usr/bin/env bash
#SBATCH --time=00:01:00
echo "Hello from $(hostname) at $(date)"
sleep 5
echo "Done."
```
Expected output includes a parsed job ID, e.g.:
```
Submitted JOBID=12345
```

Cancel a job (optional):
```
%scancel 12345
```

Check output:
- By default (if no extra args to `%%sbatch`), SLURM writes `slurm-%j.out` in `$HOME` on the remote.
- You can inspect it via SSH:
  ```
  !ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa user@login-node 'ls -ltr slurm-*.out | tail -n 1; tail -n +1 slurm-*.out | sed -n "1,80p"'
  ```
- Or download with `scp` for local viewing.

## Troubleshooting
- Error: `slurm.magics: SLURM_SSH_HOST not set. Use %slurm_config or env vars. See %slurm_help.`  
  Set the environment variables above or run `%slurm_config ...`.
- SSH/SCP failures:  
  Check key path and permissions, passphrase requirements, network/firewall rules, and whether a bastion (`SLURM_SSH_JUMP`) is required.
- `squeue/sacct/sinfo/scancel/sbatch: command not found` on remote:  
  Ensure SLURM is installed and available on the login node `PATH`.
- `sbatch` warnings on stderr but a job ID is printed:  
  Submission likely succeeded. Confirm with `%squeue` or check the output file.
- Output file not found:  
  Verify working directory and defaults; pass explicit `--chdir` and `--output` to `%%sbatch` if needed.
- Wrong or missing SLURM conf:  
  Verify `SLURM_CONF_REMOTE` points to the correct `slurm.conf` on the login node.

## Security notes
- The extension disables strict host key checking (`StrictHostKeyChecking=no`) for SSH and SCP. This avoids interactive prompts but increases MITM risk. Ensure you trust your network and hosts.
- Protect your private keys. Prefer passphrase-protected keys and secure storage. Do not commit credentials to version control.
- Use least-privilege accounts and restrict bastion/login node access.
- Avoid embedding secrets directly in notebooks; prefer environment variables or secret managers.

## Limitations
- All operations occur on the remote login node; no local SLURM execution.
- Requires network connectivity and SSH access from the notebook environment.
- Job submission parsing relies on `sbatch --parsable` output; unusual scheduler configurations may affect parsing.
