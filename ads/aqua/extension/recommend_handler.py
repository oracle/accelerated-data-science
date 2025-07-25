
from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.shaperecommend.recommend import AquaRecommendApp
from ads.config import COMPARTMENT_OCID


class AquaRecommendHandler(AquaAPIhandler):
    """
    Handler for Aqua GPU Recommendation REST APIs.

    Methods
    -------
    get(self, id: Union[str, List[str]])
        Retrieves a list of AQUA deployments or model info or logs by ID.
    post(self, *args, **kwargs)
        Obtains the eligible compute shapes that would fit the specifed model, context length, model weights, and quantization level.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """
        Lists the eligible GPU compute shapes for the specifed model.

        Returns
        -------
        List[ComputeShapeSummary]:
            The list of the model deployment shapes.
        """
        try:
            input_data = self.get_json_body()
            # input_data["compartment_id"] = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        self.finish(AquaRecommendApp().which_gpu(**input_data))

__handlers__ = [
    ("recommendation/?([^/]*)", AquaRecommendHandler),
]
