from subprocess import CompletedProcess
from unittest.mock import patch, Mock, call

from oci.marketplace.models import AcceptedAgreementSummary, AgreementSummary

from ads.opctl.backend.marketplace.models.marketplace_type import HelmMarketplaceListingDetails, \
    MarketplaceListingDetails
from ads.opctl.backend.marketplace.prerequisite_checker import check_prerequisites, _prompt_kubernetes_confirmation_, \
    _check_binaries_, _check_license_for_listing_


@patch(
    "ads.opctl.backend.marketplace.prerequisite_checker._prompt_kubernetes_confirmation_")
@patch("ads.opctl.backend.marketplace.prerequisite_checker._check_kubernetes_secret_")
@patch("ads.opctl.backend.marketplace.prerequisite_checker._check_binaries_")
@patch("ads.opctl.backend.marketplace.prerequisite_checker._check_license_for_listing_")
def test_check_prerequisites(license_api: Mock, check_binary_api: Mock, prompt_kubernetes_confirmation_api: Mock,
                             check_kubernetes_secret_api: Mock):
    listing_detail = Mock(spec=HelmMarketplaceListingDetails)
    check_prerequisites(listing_detail)
    license_api.assert_called_once()
    check_binary_api.assert_called_once()
    prompt_kubernetes_confirmation_api.assert_called_once()
    check_kubernetes_secret_api.assert_called_once()


@patch("ads.opctl.backend.marketplace.prerequisite_checker.click")
def test_prompt_kubernetes_confirmation(click: Mock):
    _prompt_kubernetes_confirmation_()
    click.confirm.assert_called_once_with(text='Is it safe to proceed?', default=False, abort=True)


@patch("ads.opctl.backend.marketplace.prerequisite_checker.subprocess.run")
def test_check_binaries_success(subprocess_runner: Mock):
    binaries = ["bin_a", "bin_b"]
    subprocess_runner.return_value = CompletedProcess(args="", returncode=0)
    _check_binaries_(binaries)
    subprocess_runner.assert_has_calls([
        call(['which', 'bin_a'], capture_output=True),
        call(['which', 'bin_b'], capture_output=True)
    ])


@patch("ads.opctl.backend.marketplace.prerequisite_checker.subprocess.run")
def test_check_binaries_failure(subprocess_runner: Mock):
    binaries = ["bin_a", "bin_b"]
    subprocess_runner.return_value = CompletedProcess(args="", returncode=-1)
    _check_binaries_(binaries)
    subprocess_runner.assert_has_calls([
        call(['which', 'bin_a'], capture_output=True),
        call(['which', 'bin_b'], capture_output=True)
    ])


@patch("ads.opctl.backend.marketplace.prerequisite_checker.get_marketplace_client")
def test_check_license_for_listing_when_already_accepted(marketplace_client: Mock):
    listing_detail = MarketplaceListingDetails(listing_id="lid", compartment_id="c_id", version="1.0")

    accepted_agreement = AcceptedAgreementSummary(agreement_id="accepted_agreement_ocid")
    marketplace_client.return_value.list_accepted_agreements.return_value.data = [accepted_agreement]
    marketplace_client.return_value.list_agreements.return_value.data = [AgreementSummary(id="accepted_agreement_ocid")]

    _check_license_for_listing_(listing_detail)
    marketplace_client.assert_called_once()
    marketplace_client.return_value.list_accepted_agreements.assert_called_once()
    marketplace_client.return_value.list_agreements.assert_called_once()

@patch("ads.opctl.backend.marketplace.prerequisite_checker.click")
@patch("ads.opctl.backend.marketplace.prerequisite_checker.time")
@patch("ads.opctl.backend.marketplace.prerequisite_checker.webbrowser.open")
@patch("ads.opctl.backend.marketplace.prerequisite_checker.get_marketplace_client")
def test_check_license_for_listing_pending_and_accepted(marketplace_client: Mock, web_client:Mock, timer:Mock, click:Mock):
    listing_detail = MarketplaceListingDetails(listing_id="lid", compartment_id="c_id", version="1.0")
    AGREEMENT_CONTENT_URL = "https://test.content.url"
    AGREEMENT_PROMPT = "test-prompt"

    click.confirm.return_value = True
    marketplace_client.return_value.list_accepted_agreements.return_value.data = []
    marketplace_client.return_value.list_agreements.return_value.data = [AgreementSummary(id="accepted_agreement_ocid")]
    marketplace_client.return_value.get_agreement.return_value.data.content_url = AGREEMENT_CONTENT_URL
    marketplace_client.return_value.get_agreement.return_value.data.prompt = AGREEMENT_PROMPT

    _check_license_for_listing_(listing_detail)

    marketplace_client.assert_called_once()
    marketplace_client.return_value.list_accepted_agreements.assert_called_once()
    marketplace_client.return_value.list_agreements.assert_called_once()
    web_client.assert_called_once_with(AGREEMENT_CONTENT_URL)
    click.confirm.assert_called_once_with(AGREEMENT_PROMPT, default=False)
    marketplace_client.return_value.create_accepted_agreement.assert_called_once()