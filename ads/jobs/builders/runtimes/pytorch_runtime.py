import json
from ads.jobs.builders.runtimes.artifact import PythonArtifact, GitPythonArtifact
from ads.jobs.builders.runtimes.python_runtime import (
    PythonRuntime,
    GitPythonRuntime,
)
from ads.jobs.builders.infrastructure.dsc_job_runtime import (
    IncompatibleRuntime,
    PythonRuntimeHandler,
    GitPythonRuntimeHandler,
)
from ads.opctl.distributed.common import cluster_config_helper


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
                GitPythonArtifact.CONST_DRIVER_SCRIPT,
            ]
        )

        # Zip the job artifact
        self.path = self._zip_artifacts()


class PyTorchDistributedRuntimeHandler(PythonRuntimeHandler):
    RUNTIME_CLASS = PyTorchDistributedRuntime
    CONST_WORKER_COUNT = "OCI__WORKER_COUNT"
    CONST_INPUT_MAPPINGS = "OCI__INPUT_MAPPINGS"

    GIT_SPEC_MAPPINGS = {
        cluster_config_helper.OCI__RUNTIME_URI: GitPythonRuntime.CONST_GIT_URL,
        cluster_config_helper.OCI__RUNTIME_GIT_BRANCH: GitPythonRuntime.CONST_BRANCH,
        cluster_config_helper.OCI__RUNTIME_GIT_COMMIT: GitPythonRuntime.CONST_COMMIT,
        cluster_config_helper.OCI__RUNTIME_GIT_SECRET_ID: GitPythonRuntime.CONST_GIT_SSH_SECRET_ID,
    }

    def _translate_artifact(self, runtime: PyTorchDistributedRuntime):
        return PyTorchDistributedArtifact(runtime.source_uri, runtime)

    def _translate_env(self, runtime: PyTorchDistributedRuntime) -> dict:
        envs = super()._translate_env(runtime)
        replica = runtime.replica if runtime.replica else 1
        envs[self.CONST_WORKER_COUNT] = str(replica - 1)
        envs[self.CONST_JOB_ENTRYPOINT] = PyTorchDistributedArtifact.CONST_DRIVER_SCRIPT
        if runtime.inputs:
            envs[self.CONST_INPUT_MAPPINGS] = json.dumps(runtime.inputs)
        if runtime.git:
            envs[GitPythonRuntimeHandler.CONST_ENTRYPOINT] = envs.pop(
                PythonRuntimeHandler.CONST_CODE_ENTRYPOINT
            )
            for env_key, spec_key in self.GIT_SPEC_MAPPINGS.items():
                if not runtime.git.get(spec_key):
                    continue
                envs[env_key] = runtime.git[spec_key]
        return envs

    def _extract_envs(self, dsc_job) -> dict:
        spec = super()._extract_envs(dsc_job)
        envs = spec.pop(PythonRuntime.CONST_ENV_VAR, {})
        if self.CONST_WORKER_COUNT not in envs:
            raise IncompatibleRuntime()
        spec[PyTorchDistributedRuntime.CONST_REPLICA] = envs.pop(
            self.CONST_WORKER_COUNT
        )
        input_mappings = envs.pop(self.CONST_INPUT_MAPPINGS, None)
        if input_mappings:
            spec[PyTorchDistributedRuntime.CONST_INPUT] = input_mappings
        if envs:
            spec[PythonRuntime.CONST_ENV_VAR] = envs
        return spec
