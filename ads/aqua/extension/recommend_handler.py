from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.shaperecommend.recommend import AquaRecommendApp


class AquaRecommendHandler(AquaAPIhandler):
    """
    Handler for Aqua GPU Recommendation REST APIs.

    Methods
    -------
    post(self, *args, **kwargs)
        Obtains the eligible compute shapes that would fit the specifed model, context length, model weights, and quantization level.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """
        Obtains the eligible compute shapes that would fit the specifed model, context length, model weights, and quantization level.

        Returns
        -------
        ShapeRecommendationReport
            Report containing shape recommendations and troubleshooting advice, if any.
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        self.finish(AquaRecommendApp().which_gpu(**input_data))


__handlers__ = [
    ("recommendation/?([^/]*)", AquaRecommendHandler),
]
