
import logging
import sys
import time
from typing import Callable

import oci
from oci import Signer
from tqdm.auto import tqdm
from ads.common.oci_datascience import OCIDataScienceMixin

logger = logging.getLogger(__name__)

WORK_REQUEST_STOP_STATE = ("SUCCEEDED", "FAILED", "CANCELED")
DEFAULT_WAIT_TIME = 1200
DEFAULT_POLL_INTERVAL = 10
WORK_REQUEST_PERCENTAGE = 100
# default tqdm progress bar format: 
# {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]
# customize the bar format to remove the {n_fmt}/{total_fmt} from the right side
DEFAULT_BAR_FORMAT = '{l_bar}{bar}| [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'


class ADSWorkRequest(OCIDataScienceMixin):

    def __init__(
        self, 
        id: str, 
        description: str = "Processing",
        config: dict = None, 
        signer: Signer = None, 
        client_kwargs: dict = None, 
        **kwargs
    ) -> None:
        self.id = id
        self._description = description
        self._percentage = 0
        self._status = None
        super().__init__(config, signer, client_kwargs, **kwargs)
        

    def _sync(self):
        work_request = self.client.get_work_request(self.id).data
        work_request_logs = self.client.list_work_request_logs(
            self.id
        ).data

        self._percentage= work_request.percent_complete
        self._status = work_request.status
        self._description = work_request_logs[-1].message if work_request_logs else "Processing"

    def watch(
        self, 
        progress_callback: Callable,
        max_wait_time: int,
        poll_interval: int,
    ):
        previous_percent_complete = 0

        start_time = time.time()
        while self._percentage < 100:

            seconds_since = time.time() - start_time
            if max_wait_time > 0 and seconds_since >= max_wait_time:
                logger.error(f"Max wait time ({max_wait_time} seconds) exceeded.")
                return

            time.sleep(poll_interval)

            try:
                self._sync()
            except Exception as ex:
                logger.warn(ex)
                continue

            percent_change = self._percentage - previous_percent_complete
            previous_percent_complete = self._percentage
            progress_callback(percent_change, self._description)

            if self._status in WORK_REQUEST_STOP_STATE:
                if self._status != oci.work_requests.models.WorkRequest.STATUS_SUCCEEDED:
                    if self._description:
                        raise Exception(self._description)
                    else:
                        raise Exception(
                            "Error occurred in attempt to perform the operation. "
                            "Check the service logs to get more details. "
                        )
                else:
                    break

        progress_callback(0, "Done")


def wait_work_request(
    id: str, 
    progress_bar_description: str, 
    max_wait_time: int=DEFAULT_WAIT_TIME,
    poll_interval: int=DEFAULT_POLL_INTERVAL
):
    ads_work_request = ADSWorkRequest(id)

    with tqdm(
        total=WORK_REQUEST_PERCENTAGE,
        leave=False,
        mininterval=0,
        file=sys.stdout,
        desc=progress_bar_description,
        bar_format=DEFAULT_BAR_FORMAT
    ) as pbar:

        def progress_callback(percent_change, description):
            if percent_change != 0:
                pbar.update(percent_change)
            if description:
                pbar.set_description(description)

        ads_work_request.watch(
            progress_callback,
            max_wait_time,
            poll_interval
        )
 