from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
import os, re, sys, subprocess, tempfile, uuid, shlex, textwrap


def _cfg():
    return {
        "host": os.environ.get("SLURM_SSH_HOST", ""),
        "key": os.path.expanduser(os.environ.get("SLURM_SSH_KEY", "~/.ssh/id_rsa")),
        "conf": os.environ.get("SLURM_CONF_REMOTE", "/etc/slurm/slurm.conf"),
        "jump": os.environ.get("SLURM_SSH_JUMP"),
    }


def _ssh_base(cfg):
    base = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", cfg["key"]]
    if cfg["jump"]:
        base += ["-J", cfg["jump"]]
    base += [cfg["host"]]
    return base


def _scp_base(cfg):
    base = ["scp", "-q", "-o", "StrictHostKeyChecking=no", "-i", cfg["key"]]
    if cfg["jump"]:
        base += ["-J", cfg["jump"]]
    return base


def _run_ssh(cfg, remote_cmd, input_bytes=None):
    if not cfg["host"]:
        return 255, "", "slurm.magics: SLURM_SSH_HOST not set. Use %slurm_config or env vars. See %slurm_help."
    try:
        p = subprocess.run(
            _ssh_base(cfg) + [remote_cmd],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p.returncode, p.stdout.decode(), p.stderr.decode()
    except Exception as e:
        return 255, "", f"ssh error: {e}"


def _upload(cfg, local_path, remote_path):
    if not cfg["host"]:
        return 255, "", "slurm.magics: SLURM_SSH_HOST not set. Use %slurm_config or env vars. See %slurm_help."
    try:
        p = subprocess.run(
            _scp_base(cfg) + [local_path, f'{cfg["host"]}:{remote_path}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p.returncode, p.stdout.decode(), p.stderr.decode()
    except Exception as e:
        return 255, "", f"scp error: {e}"


@magics_class
class SlurmMagics(Magics):
    @line_magic
    def slurm_help(self, line=""):
        msg = """
        Slurm SSH Magics (ads.slurm.magics)

        Configure via environment variables:
          SLURM_SSH_HOST=user@login-node
          SLURM_SSH_KEY=~/.ssh/id_rsa
          SLURM_CONF_REMOTE=/etc/slurm/slurm.conf
          SLURM_SSH_JUMP=user@bastion   (optional ProxyJump)

        Or at runtime:
          %slurm_config --host user@host --key ~/.ssh/key --conf /etc/slurm/slurm.conf [--jump user@bastion]

        Commands:
          %squeue [args]        %sacct [args]        %sinfo [args]        %scancel <JOBID>
          %%sbatch [sbatch args]    (cell contains job script; shebang/#SBATCH flush-left)

        Examples:
          %squeue -u USER
          %%sbatch
          #!/usr/bin/env bash
          #SBATCH --time=00:01:00
          echo "Hello from $(hostname) at $(date)"

        Notes:

          - All actions run over SSH on the login node; no Slurm/MUNGE installed locally.
          - Defaults (if none provided): --chdir=$HOME --output=slurm-%j.out for predictable logs.
          - JOBID is parsed from stdout only; stderr warnings are shown but ignored for parsing.
        """
        print(textwrap.dedent(msg).strip())

    @line_magic
    def slurm_config(self, line=""):
        args = shlex.split(line)
        # simple --key value parsing
        opts = {}
        i = 0
        while i < len(args):
            if args[i].startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
                opts[args[i]] = args[i + 1]
                i += 2
            else:
                i += 1

        if "--host" in opts:
            os.environ["SLURM_SSH_HOST"] = opts["--host"]
        if "--key" in opts:
            os.environ["SLURM_SSH_KEY"] = opts["--key"]
        if "--conf" in opts:
            os.environ["SLURM_CONF_REMOTE"] = opts["--conf"]
        if "--jump" in opts:
            os.environ["SLURM_SSH_JUMP"] = opts["--jump"]

        c = _cfg()
        print(f"Configured host={c['host']}, key={c['key']}, conf={c['conf']}, jump={c['jump']}")

    @line_magic
    def squeue(self, line=""):
        c = _cfg()
        rc, out, err = _run_ssh(c, f"export SLURM_CONF={shlex.quote(c['conf'])}; squeue {line}")
        if err.strip():
            print(err, file=sys.stderr)
        print(out, end="")

    @line_magic
    def sacct(self, line=""):
        c = _cfg()
        rc, out, err = _run_ssh(c, f"export SLURM_CONF={shlex.quote(c['conf'])}; sacct {line}")
        if err.strip():
            print(err, file=sys.stderr)
        print(out, end="")

    @line_magic
    def sinfo(self, line=""):
        c = _cfg()
        rc, out, err = _run_ssh(c, f"export SLURM_CONF={shlex.quote(c['conf'])}; sinfo {line}")
        if err.strip():
            print(err, file=sys.stderr)
        print(out, end="")

    @line_magic
    def scancel(self, line=""):
        job = line.strip()
        if not job:
            print("Usage: %scancel <JOBID>", file=sys.stderr)
            return
        c = _cfg()
        rc, out, err = _run_ssh(c, f"export SLURM_CONF={shlex.quote(c['conf'])}; scancel {shlex.quote(job)}")
        if err.strip():
            print(err, file=sys.stderr)
        print(out, end="")

    @cell_magic
    def sbatch(self, line, cell):
        c = _cfg()
        script = cell.lstrip()
        local_tmp = os.path.join(tempfile.gettempdir(), f"nb{uuid.uuid4().hex}.sh")
        remote_tmp = f"/tmp/nb{uuid.uuid4().hex}.sh"
        # Write local script
        try:
            with open(local_tmp, "w") as f:
                f.write(script)
        except Exception as e:
            print(f"Failed to write temp file: {e}", file=sys.stderr)
            return
        try:
            rc, out, err = _upload(c, local_tmp, remote_tmp)
            if rc != 0:
                print(f"Upload failed: {err or out}", file=sys.stderr)
                return
            defaults = "--chdir=$HOME --output=slurm-%j.out"
            extra = line.strip() if line.strip() else defaults
            rc, out, err = _run_ssh(
                c,
                f"export SLURM_CONF={shlex.quote(c['conf'])}; chmod +x {shlex.quote(remote_tmp)} && sbatch --parsable {extra} {shlex.quote(remote_tmp)}"
            )
            jobid = out.strip().split(";")[0].strip() if out is not None else ""
            if re.fullmatch(r"[0-9]+", jobid):
                print(f"Submitted JOBID={jobid}")
            else:
                print("Submission failed. stdout/stderr below:", file=sys.stderr)
                if out and out.strip():
                    print(out, file=sys.stderr)
                if err and err.strip():
                    print(err, file=sys.stderr)
        finally:
            try:
                os.remove(local_tmp)
            except Exception:
                pass
            # best-effort remote cleanup
            _run_ssh(c, f"rm -f {shlex.quote(remote_tmp)}")


def load_ipython_extension(ip):
    ip.register_magics(SlurmMagics)


def unload_ipython_extension(ip):
    pass
