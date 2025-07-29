from unittest.mock import MagicMock, patch

import pytest
from tornado.web import HTTPError

from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.extension.recommend_handler import AquaRecommendHandler


@pytest.fixture
def handler():
    # Patch AquaAPIhandler.__init__ for unit test stubbing
    AquaAPIhandler.__init__ = lambda self, *args, **kwargs: None
    h = AquaRecommendHandler(MagicMock(), MagicMock())
    h.finish = MagicMock()
    h.request = MagicMock()
    # Set required Tornado internal fields
    h._headers = {}
    h._write_buffer = []
    return h


def test_post_valid_input(monkeypatch, handler):
    input_data = {"model_ocid": "ocid1.datasciencemodel.oc1.XYZ"}
    expected = {"recommendations": ["VM.GPU.A10.1"], "troubleshoot": ""}

    # Patch class on correct import path, so handler sees our fake implementation
    class FakeAquaRecommendApp:
        def which_gpu(self, **kwargs):
            return expected

    monkeypatch.setattr(
        "ads.aqua.extension.recommend_handler.AquaRecommendApp", FakeAquaRecommendApp
    )

    handler.get_json_body = MagicMock(return_value=input_data)
    handler.post()
    handler.finish.assert_called_once_with(expected)


def test_post_no_input(handler):
    handler.get_json_body = MagicMock(return_value=None)
    handler._headers = {}
    handler._write_buffer = []
    handler.write_error = MagicMock()
    handler.post()
    handler.write_error.assert_called_once()
    exc_info = handler.write_error.call_args.kwargs.get("exc_info")
    assert exc_info is not None
    exc_type, exc_value, _ = exc_info
    assert exc_type is HTTPError
    assert exc_value.status_code == 400
    assert exc_value.log_message == Errors.NO_INPUT_DATA


def test_post_invalid_input(handler):
    handler.get_json_body = MagicMock(side_effect=Exception("bad input"))
    handler._headers = {}
    handler._write_buffer = []
    handler.write_error = MagicMock()
    handler.post()
    handler.write_error.assert_called_once()
    exc_info = handler.write_error.call_args.kwargs.get("exc_info")
    assert exc_info is not None
    exc_type, exc_value, _ = exc_info
    assert exc_type is HTTPError
    assert exc_value.status_code == 400
    assert exc_value.log_message == Errors.INVALID_INPUT_DATA_FORMAT
