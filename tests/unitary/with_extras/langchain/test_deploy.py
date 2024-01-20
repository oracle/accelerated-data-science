import os
import tempfile
from unittest.mock import MagicMock, patch

from ads.llm.deploy import ChainDeployment

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tests.unitary.with_extras.langchain.test_guardrails import FakeLLM


class TestLangChainDeploy:
    def generate_chain_application(self):
        prompt = PromptTemplate.from_template("Tell me a joke about {subject}")
        llm_chain = LLMChain(prompt=prompt, llm=FakeLLM(), verbose=True)
        return llm_chain

    @patch("ads.model.datascience_model.DataScienceModel._to_oci_dsc_model")
    def test_initialize(self, mock_to_oci_dsc_model):
        chain_application = self.generate_chain_application()
        chain_deployment = ChainDeployment(chain_application, auth=MagicMock())
        mock_to_oci_dsc_model.assert_called()
        assert isinstance(chain_deployment.chain, LLMChain)

    @patch("ads.model.runtime.env_info.get_service_packs")
    @patch("ads.common.auth.default_signer")
    @patch("ads.model.datascience_model.DataScienceModel._to_oci_dsc_model")
    def test_prepare(
        self, mock_to_oci_dsc_model, mock_default_signer, mock_get_service_packs
    ):
        mock_default_signer.return_value = MagicMock()
        inference_conda_env = "oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1"
        inference_python_version = "3.7"
        mock_get_service_packs.return_value = (
            {
                inference_conda_env: ("mlcpuv1", inference_python_version),
            },
            {
                "mlcpuv1": (inference_conda_env, inference_python_version),
            },
        )
        artifact_dir = tempfile.mkdtemp()
        chain_application = self.generate_chain_application()
        chain_deployment = ChainDeployment(
            chain_application, artifact_dir=artifact_dir, auth=MagicMock()
        )

        mock_to_oci_dsc_model.assert_called()

        chain_deployment.prepare(
            inference_conda_env="oci://service-conda-packs@ociodscdev/service_pack/cpu/General_Machine_Learning_for_CPUs/1.0/mlcpuv1",
            inference_python_version="3.7",
        )

        mock_get_service_packs.assert_called()

        score_py_file_location = os.path.join(chain_deployment.artifact_dir, "score.py")
        chain_yaml_file_location = os.path.join(
            chain_deployment.artifact_dir, "chain.yaml"
        )
        runtime_yaml_file_location = os.path.join(
            chain_deployment.artifact_dir, "runtime.yaml"
        )
        assert (
            os.path.isfile(score_py_file_location)
            and os.path.getsize(score_py_file_location) > 0
        )
        assert (
            os.path.isfile(chain_yaml_file_location)
            and os.path.getsize(chain_yaml_file_location) > 0
        )
        assert (
            os.path.isfile(runtime_yaml_file_location)
            and os.path.getsize(runtime_yaml_file_location) > 0
        )
