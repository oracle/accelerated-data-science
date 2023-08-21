To create and start running the job:

.. tabs::

  .. code-tab:: python
    :caption: Python

    # Create the job on OCI Data Science
    job.create()
    # Start a job run
    run = job.run()
    # Stream the job run outputs (from the first node)
    run.watch()

  .. code-tab:: bash
    :caption: YAML

    ads opctl run -f your_job.yaml
