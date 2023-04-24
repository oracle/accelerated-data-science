import json
from ads.jobs.builders.runtimes.artifact import PythonArtifact, GitPythonArtifact
from ads.jobs.builders.runtimes.python_runtime import (
    PythonRuntime,
    GitPythonRuntime,
)


class PyTorchDistributedRuntime(PythonRuntime):
    CONST_GIT = "git"
    CONST_REPLICA = "replicas"
    CONST_INPUT = "inputs"

    def with_git(
        self, url: str, branch: str = None, commit: str = None, secret_ocid: str = None
    ):
        """Specifies the Git repository and branch/commit for the job source code.

        Parameters
        ----------
        url : str
            URL of the Git repository.
        branch : str, optional
            Git branch name, by default None, the default branch will be used.
        commit : str, optional
            Git commit ID (SHA1 hash), by default None, the most recent commit will be used.
        secret_ocid : str
            The secret OCID storing the SSH key content for checking out the Git repository.

        Returns
        -------
        self
            The runtime instance.
        """
        git_spec = {GitPythonRuntime.CONST_GIT_URL: url}
        if branch:
            git_spec[GitPythonRuntime.CONST_BRANCH] = branch
        if commit:
            git_spec[GitPythonRuntime.CONST_COMMIT] = commit
        if secret_ocid:
            git_spec[GitPythonRuntime.CONST_GIT_SSH_SECRET_ID] = secret_ocid
        return self.set_spec(self.CONST_GIT, git_spec)

    @property
    def git(self) -> str:
        return self.get_spec(self.CONST_GIT)

    def with_inputs(self, mappings: dict):
        return self.set_spec(self.CONST_INPUT, mappings)

    @property
    def inputs(self) -> dict:
        return self.get_spec(self.CONST_INPUT)

    def with_replica(self, count: int):
        return self.set_spec(self.CONST_REPLICA, count)

    @property
    def replica(self) -> int:
        return self.get_spec(self.CONST_REPLICA)

    def run(self, dsc_job, **kwargs):
        replicas = self.replica if self.replica else 1
        main_run = None
        for i in range(replicas):
            replica_kwargs = kwargs.copy()
            envs = replica_kwargs.get("environment_variables")
            if not envs:
                envs = {}
            if main_run:
                envs["MAIN_JOB_RUN_OCID"] = main_run.id
            name = replica_kwargs.get("display_name")
            if not name:
                name = dsc_job.display_name

            replica_kwargs["display_name"] = f"{name}-{str(i)}"
            replica_kwargs["environment_variables"] = envs
            run = dsc_job.run(**replica_kwargs)
            if i == 0:
                main_run = run
        return main_run


class PyTorchDistributedArtifact(PythonArtifact):
    CONST_DRIVER_SCRIPT = "driver_pytorch.py"
    CONST_LIB_HOSTNAME = "hostname_from_env.c"
    CONST_OCI_METRICS = "oci_metrics.py"

    def __init__(self, source, runtime=None) -> None:
        if not source:
            source = ""
        super().__init__(source, runtime)

    def build(self):
        """Prepares job artifact."""
        self._copy_artifacts(
            drivers=[
                self.CONST_DRIVER_UTILS,
                self.CONST_DRIVER_SCRIPT,
                self.CONST_LIB_HOSTNAME,
                self.CONST_OCI_METRICS,
                GitPythonArtifact.CONST_DRIVER_SCRIPT,
            ]
        )

        # Zip the job artifact
        self.path = self._zip_artifacts()
