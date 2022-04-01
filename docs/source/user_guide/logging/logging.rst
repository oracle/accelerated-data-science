.. _logging:

Logging
-------

The Oracle Cloud Infrastructure (OCI) `Logging service <https://docs.oracle.com/en-us/iaas/Content/Logging/Concepts/loggingoverview.htm>`__
is a highly scalable and fully managed single pane of glass for all the logs in your tenancy. 
Logging provides access to logs from OCI resources, such as `jobs <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/jobs/index.html>`__ 
and `model deployments <https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/model_deployment/model_deployment.html>`__

ADS provides the APIs to simplify the creation, retrieval, and deletion of log groups and custom log resources.

Creating a log group requires a display name and compartment OCID. The compartment OCID is not needed if you are running 
the code in a Data Science notebook session.

.. code:: ipython3

    from ads.common.oci_logging import OCILogGroup


    # Create a new log group
    # compartment_id is optional if running in a Data Science notebook session.
    log_group = OCILogGroup(
        display_name="<your_log_group_name>",
        compartment_id="<your_compartment_ocid>"
    ).create()

    # Get the log group OCID
    log_group_ocid = log_group.id

    # Create a custom log in the log group
    log = log_group.create_log(display_name="<your_log_name>")

    # Get the log OCID
    log_ocid = log.id

    # Delete a single log resource
    log.delete()

    # Delete the log group and the log resource in the log group
    log_group.delete()

    # Get a existing log group by OCID
    log_group = OCILogGroup.from_ocid("<log_group_ocid>")

    # Get a list of existing log resources in a log group
    # A list of ads.common.oci_logging.OCILog objects will be returned
    log_group.list_logs()

    # Get the last 50 log messages as a list
    log.tail(limit=50)

    # Stream the log messages to terminal or screen
    # This block sthe main process until user interruption.
    log.stream()
