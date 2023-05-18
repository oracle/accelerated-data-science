import datetime
import oci
import os
import subprocess
import time


JOB_RUN_COMPARTMENT_OCID_ENV = "JOB_RUN_COMPARTMENT_OCID"
METRIC_NAMESPACE = os.environ.get("OCI__METRICS_NAMESPACE")
METRIC_COMPARTMENT = os.environ.get(JOB_RUN_COMPARTMENT_OCID_ENV)

# When querying metrics, the smallest aggregation interval allowed is 1 minute.
# See https://docs.oracle.com/iaas/Content/Monitoring/Reference/mql.htm#Interval
METRIC_SUBMISSION_INTERVAL_SECONDS = 60


class Metric:
    """
    Class containing metric details
    """

    def __init__(self, name: str, value: float, dimensions: dict = None):
        """
        Initializes a Metrics object

        Parameters
        ----------
        name: str
            The metric name
        value: float
            The metric value
        dimensions: dict
            Dictionary of dimensions to include on the metric.
        """
        self.name = name
        self.value = value
        self.dimensions = dimensions if dimensions else {}


class GpuMetricsProvider:
    """
    Custom GPU utilization metrics provider
    """

    def get_metrics(self) -> list:
        """
        Get custom GPU metrics

        Returns
        -------
        list
            List of Metric objects.
        """
        gpu_metrics = []
        try:
            # Example output:
            # 00000000:00:04.0, 42.27, 40, 20, 16384, 15287
            # 00000000:00:05.0, 41.30, 42, 0, 16384, 479
            nvidia_smi_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=pci.bus_id,power.draw,temperature.gpu,utilization.gpu,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            ).decode("utf-8")

            for line in nvidia_smi_output.split("\n"):
                if len(line) > 0:
                    values = line.split(", ")
                    if len(values) >= 6:
                        dimensions = {"pci_bus": values[0]}
                        gpu_metrics.append(
                            Metric("gpu.power_draw", float(values[1]), dimensions)
                        )
                        gpu_metrics.append(
                            Metric("gpu.temperature", float(values[2]), dimensions)
                        )
                        gpu_metrics.append(
                            Metric("gpu.gpu_utilization", float(values[3]), dimensions)
                        )
                        gpu_metrics.append(
                            Metric(
                                "gpu.memory_usage",
                                round(float(values[5]) / float(values[4]) * 100, 2),
                                dimensions,
                            )
                        )
                    else:
                        print(
                            f"Unexpected nvidia-smi output format. Composing no metrics for output line: {line}"
                        )
        except Exception as e:
            print(f"Unexpected error encountered querying GPU: {e}")

        return gpu_metrics


metric_providers = [
    GpuMetricsProvider(),
]


def convert_to_metric_data_details(
    metric: Metric, timestamp: datetime.datetime
) -> oci.monitoring.models.MetricDataDetails:
    """
    Converts a Metric object into an oci.monitoring.models.MetricDataDetails object for submission to the Monitoring
    Service. In addition to the dimensions defined on the input Metric, the job ocid and job run ocid are also added
    as dimensions.

    Parameters
    ----------
    metric: Metric
        The Metric object to convert
    timestamp: datetime.datetime
        The timestamp to include on the metric datapoint

    Returns
    -------
    oci.monitoring.models.MetricDataDetails
        The oci.monitoring.models.MetricDataDetails object containing the metric details
    """
    dimensions = metric.dimensions
    dimensions["job_run_ocid"] = os.environ.get("JOB_RUN_OCID")
    dimensions["job_ocid"] = os.environ.get("JOB_OCID")
    return oci.monitoring.models.MetricDataDetails(
        namespace=METRIC_NAMESPACE,
        compartment_id=METRIC_COMPARTMENT,
        name=metric.name,
        dimensions=dimensions,
        datapoints=[
            oci.monitoring.models.Datapoint(
                timestamp=timestamp, value=metric.value, count=1
            )
        ],
    )


def submit_metrics(client: oci.monitoring.MonitoringClient) -> None:
    """
    Submit metrics to the Monitoring Service

    Parameters
    ----------
    client: oci.monitoring.MonitoringClient
        The OCI Monitoring Service client
    """
    metric_data_details = []
    timestamp = datetime.datetime.now()
    for provider in metric_providers:
        for metric in provider.get_metrics():
            metric_data_details.append(
                convert_to_metric_data_details(metric, timestamp)
            )
    post_metric_details = oci.monitoring.models.PostMetricDataDetails(
        metric_data=metric_data_details
    )
    client.post_metric_data(post_metric_data_details=post_metric_details)


def collect_metrics():
    if JOB_RUN_COMPARTMENT_OCID_ENV not in os.environ:
        return
    try:
        signer = oci.auth.signers.get_resource_principals_signer()
    except EnvironmentError:
        return
    client = oci.monitoring.MonitoringClient(
        config={},
        signer=signer,
        # Metrics should be submitted with the "telemetry-ingestion" endpoint instead.
        # See note here: https://docs.oracle.com/iaas/api/#/en/monitoring/20180401/MetricData/PostMetricData
        service_endpoint=f"https://telemetry-ingestion.{signer.region}.oraclecloud.com",
    )

    while True:
        try:
            submit_metrics(client)
        except:
            pass
        time.sleep(METRIC_SUBMISSION_INTERVAL_SECONDS)
