===================
PyTorch Distributed
===================

PyTorch is an open source machine learning framework used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab. ADS supports running PyTorch's native distributed training code (``torch.distributed`` and ``DistributedDataParallel``) with OCI Data Science Jobs. Provided you are following the `official PyTorch distributed data parallel guidelines <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#comparison-between-dataparallel-and-distributeddataparallel>`_, **no changes to your PyTorch code are required**.

PyTorch distributed training requires initialization using the ``torch.distributed.init_process_group()`` function. By default this function collects uses environment variables to initialize the communications for the training cluster. When using ADS to run PyTorch distributed training on OCI data science Jobs, the environment variables, including ``MASTER_ADDR``, ``MASTER_PORT``, ``WORLD_SIZE`` ``RANK``, and ``LOCAL_RANK`` will automatically be set in the job runs. By default ``MASTER_PORT`` will be set to ``29400``.

.. toctree::
    :maxdepth: 3

    creating
