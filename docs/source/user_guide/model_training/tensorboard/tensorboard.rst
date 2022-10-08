===========
TensorBoard
===========

TensorBoard helps visualizing your experiments. You bring up a ``TensorBoard`` session on your workstation and point to the directory that contains the TensorBoard logs.

**Prerequisite**

1. Object storage bucket
2. Access to Object Storage bucket from your workstation
3. ``ocifs`` version 1.1.0 and above

Setting up local environment
----------------------------

It is required that ``tensorboard`` is installed in a dedicated conda environment or virtual environment. Prepare an environment yaml file for creating conda environment with following command -

.. code-block:: shell

    cat <<EOF > tensorboard-dep.yaml
    dependencies:
    - python=3.8
    - pip
    - pip:
        - ocifs
        - tensorboard
    name: tensorboard
    EOF

Create the conda environment from the yaml file generated in the preceeding step

.. code-block:: shell

  conda env create -f tensorboard-dep.yaml

This will create a conda environment called tensorboard. Activate the conda environment by running -

.. code-block:: shell

  conda activate tensorboard


Viewing logs from your experiments
----------------------------------

To launch a TensorBoard session on your local workstation, run -

.. code-block:: shell

    export OCIFS_IAM_KEY=api_key # If you are using resource principal, set resource_principal
    tensorboard --logdir oci://my-bucket@my-namespace/path/to/logs

This will bring up TensorBoard app on your workstation. Access TensorBoard at ``http://localhost:6006/``

**Note**: The logs take some initial time (few minutes) to reflect on the tensorboard dashboard.

Writing TensorBoard logs to Object Storage
------------------------------------------

**Prerequisite**

1. ``tensorboard`` is installed.
2. ``ocifs`` version is 1.1.0 and above.
3. ``oracle-ads`` version 2.6.0 and above.

PyTorch
.......

You could write your logs from your ``PyTorch`` experiements directly to object storage and view the logs on TensorBoard running on your local workstation in real time. Here is an example or running PyTorch experiment and writing TensorBoard logs from ``OCI Data Science Notebook``

1. Create or Open an existing ``OCI Data Science Notebook`` session
2. Run ``odsc conda install -s pytorch110_p37_cpu_v1`` on terminal inside the notebook session
3. Activate conda environment - ``conda activate /home/datascience/conda/pytorch110_p37_cpu_v1``
4. Install TensorBoard - ``python3 -m pip install tensorboard``
5. Upgrade to latest ``ocifs`` - ``python3 -m pip install ocifs --upgrade``
6. Create a notebook and select ``pytorch110_p37_cpu_v1`` kernel
7. Copy the following code into a cell and update the object storage path in the code snippet

.. code-block:: ipython

    # Reference: https://github.com/pytorch/tutorials/blob/master/recipes_source/recipes/tensorboard_with_pytorch.py

    import torch
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("oci://my-bucket@my-namespace/path/to/logs")

    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    def train_model(iter):
        for epoch in range(iter):
            y1 = model(x)
            loss = criterion(y1, y)
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_model(10)
    writer.flush()
    writer.close()

7. Run the cell
8. View the logs from you workstation while the experiement is in progress by lauching TensorBoard with following command -

.. code-block:: shell

    OCIFS_IAM_TYPE=api_key tensorboard --logdir "oci://my-bucket@my-namespace/path/to/logs"

For more possibilities with TensorBoard and PyTorch check this `link <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>`_

TensorFlow
..........

Currently TensorFlow cannot write directly to object storage. However, we can create logs in the local directory and then copy the logs over to object storage, which then can be viewed from the TensorBoard running on your local workstation.

When you run a ``OCI Data Science Job`` with ``ads.jobs.NotebookRuntime`` or ``ads.jobs.GitRuntime``, all the output is automatically copied over to the configured object storage bucket.

OCI Data Science Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of running a TensorFlow experiment in ``OCI Data Science Notebook`` and then viewing the logs from TensorBoard

1. Create or open an existing notebook session.
2. Download notebook - https://raw.githubusercontent.com/mayoor/stats-ml-exps/master/tensorboard_tf.ipynb

.. code-block:: ipython

    !wget https://raw.githubusercontent.com/mayoor/stats-ml-exps/master/tensorboard_tf.ipynb

3. Run ``odsc conda install -s tensorflow27_p37_cpu_v1`` on terminal to install TensorFlow 2.6 environment.
4. Open the downloaded notebook - ``tensorboard_tf.ipynb``
5. Select ``tensorflow27_p37_cpu_v1`` kernel.
6. Run all cells.
7. Copy TensorBoard logs folder - ``tflogs`` to object storage using ``oci-cli``

.. code-block:: shell

     oci os object bulk-upload -bn "<my-bucket>" -ns "<my-namespace>" --src-dir tflogs --prefix myexperiment/tflogs/

View the logs from you workstation once the logs are uploaded by lauching the TensorBoard with following command -

.. code-block:: shell

    OCIFS_IAM_TYPE=api_key tensorboard --logdir "oci://my-bucket@my-namespace/myexperiment/tflogs/"

OCI Data Science Jobs
~~~~~~~~~~~~~~~~~~~~~

Here is an example of running a TensorFlow experiment in ``OCI Data Science Jobs`` and then viewing the logs from TensorBoard

1. Run the following code to submit a notebook to ``OCI Data Science Job``. You could run this code snippet from your local workstation or ``OCI Data Science Notebook`` session. You need ``oracle-ads`` version >= 2.6.0.

.. code-block:: python

    from ads.jobs import Job, DataScienceJob, NotebookRuntime
    # Define an OCI Data Science job to run a jupyter Python notebook
    job = (
        Job(name="<job_name>")
        .with_infrastructure(
            # The same configurations as the OCI notebook session will be used.
            DataScienceJob()
            .with_log_group_id("oci.xxxx.<log_group_ocid>")
            .with_log_id("oci.xxx.<log_ocid>")
            .with_project_id("oci.xxxx.<project_ocid>")
            .with_shape_name("VM.Standard2.1")
            .with_subnet_id("oci.xxxx.<subnet-ocid>")
            .with_block_storage_size(50)
            .with_compartment_id("oci.xxxx.<compartment_ocid>")
        )
        .with_runtime(
            NotebookRuntime()
            .with_notebook("https://raw.githubusercontent.com/mayoor/stats-ml-exps/master/tensorboard_tf.ipynb")
            .with_service_conda("tensorflow27_p37_cpu_v1")
            # Saves the notebook with outputs to OCI object storage.
            .with_output("oci://my-bucket@my-namespace/myexperiment/jobs/")
        )
    ).create()
    # Run and monitor the job
    run = job.run().watch()

View the logs from you workstation once the jobs is complete by lauching the tensorboard with following command -

.. code-block:: shell

    OCIFS_IAM_TYPE=api_key tensorboard --logdir "oci://my-bucket@my-namespace//myexperiment/jobs/tflogs/"