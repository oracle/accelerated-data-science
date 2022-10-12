Creating PyTorch Distributed Workloads
--------------------------------------

.. include:: ../_prerequisite.rst


**Write your training code:**

For this example, the code to run was inspired from an example
`found here <https://github.com/Azure/azureml-examples/blob/main/python-sdk/workflows/train/pytorch/cifar-distributed/src/train.py>`_

Note that ``MASTER_ADDR``, ``MASTER_PORT``, ``WORLD_SIZE``, ``RANK``, and ``LOCAL_RANK`` are environment variables
that will automatically be set.

.. code-block:: python
  :caption: train.py
  :name: train.py

    # Copyright (c) 2017 Facebook, Inc. All rights reserved.
    # BSD 3-Clause License
    #
    # Script adapted from:
    # https://github.com/Azure/azureml-examples/blob/main/python-sdk/workflows/train/pytorch/cifar-distributed/src/train.py
    # ==============================================================================


    import datetime
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import os, argparse

    # define network architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.conv3 = nn.Conv2d(64, 128, 3)
            self.fc1 = nn.Linear(128 * 6 * 6, 120)
            self.dropout = nn.Dropout(p=0.2)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 6 * 6)
            x = self.dropout(F.relu(self.fc1(x)))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    # define functions
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, rank):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_freq == 0:  # print every print_freq mini-batches
                print(
                    "Rank %d: [%d, %5d] loss: %.3f"
                    % (rank, epoch + 1, i + 1, running_loss / print_freq)
                )
                running_loss = 0.0


    def evaluate(test_loader, model, device):
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        model.eval()

        correct = 0
        total = 0
        class_correct = list(0.0 for i in range(10))
        class_total = list(0.0 for i in range(10))
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(10):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # print total test set accuracy
        print(
            "Accuracy of the network on the 10000 test images: %d %%"
            % (100 * correct / total)
        )

        # print test accuracy for each of the classes
        for i in range(10):
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )


    def main(args):
        # get PyTorch environment variables
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        distributed = world_size > 1

        if torch.cuda.is_available():
            print("CUDA is available.")
        else:
            print("CUDA is not available.")

        # set device
        if distributed:
            if torch.cuda.is_available():
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize distributed process group using default env:// method
        if distributed:
            torch.distributed.init_process_group(
                backend=args.backend,
                timeout=datetime.timedelta(minutes=args.timeout)
            )

        # define train and test dataset DataLoaders
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_set = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform
        )

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
        )

        test_set = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

        model = Net().to(device)

        # wrap model with DDP
        if distributed:
            if torch.cuda.is_available():
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank
                )
            else:
                model = nn.parallel.DistributedDataParallel(model)

        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=args.momentum
        )

        # train the model
        for epoch in range(args.epochs):
            print("Rank %d: Starting epoch %d" % (rank, epoch))
            if distributed:
                train_sampler.set_epoch(epoch)
            model.train()
            train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                args.print_freq,
                rank,
            )

        print("Rank %d: Finished Training" % (rank))

        if not distributed or rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, "cifar_net.pt")
            torch.save(model.state_dict(), model_path)

            # evaluate on full test dataset
            evaluate(test_loader, model, device)


    # run script
    if __name__ == "__main__":
        # setup argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data-dir", type=str, help="directory containing CIFAR-10 dataset"
        )
        parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
        parser.add_argument(
            "--batch-size",
            default=16,
            type=int,
            help="mini batch size for each gpu/process",
        )
        parser.add_argument(
            "--workers",
            default=2,
            type=int,
            help="number of data loading workers for each gpu/process",
        )
        parser.add_argument(
            "--learning-rate", default=0.001, type=float, help="learning rate"
        )
        parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
        parser.add_argument(
            "--output-dir", default="outputs", type=str, help="directory to save model to"
        )
        parser.add_argument(
            "--print-freq",
            default=200,
            type=int,
            help="frequency of printing training statistics",
        )
        parser.add_argument(
            "--backend", default="gloo", type=str,
            help="distributed communication backend, should be gloo, nccl or mpi"
        )
        parser.add_argument(
            "--timeout", default=30, type=int,
            help="timeout in minutes for waiting for the initialization of distributed process group."
        )
        args = parser.parse_args()

        # call main function
        main(args)

**Initialize a distributed-training folder:**

At this point you have create a training file (or files) - ``train.py`` in the above
example. Now running the command below will download the artifacts required for building the docker image.
The artifacts will be saved into the ``oci_dist_training_artifacts/pytorch/v1`` directory under your current working directory.

.. code-block:: bash

  ads opctl distributed-training init --framework pytorch --version v1

**Containerize your code and build container:**


Before you can build the image, you must set the following environment variables:

Specify image name and tag

.. code-block:: bash

  export IMAGE_NAME=<region.ocir.io/my-tenancy/image-name>
  export TAG=latest


Build the container image.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
      -df oci_dist_training_artifacts/pytorch/v1/Dockerfile

The code is assumed to be in the current working directory. To override the source code directory, use the ``-s`` flag and specify the code dir. This folder should be within the current working directory.

.. code-block:: bash

  ads opctl distributed-training build-image \
      -t $TAG \
      -reg $IMAGE_NAME \
       -df oci_dist_training_artifacts/horovod/v1/oci_dist_training_artifacts/pytorch/v1/Dockerfile
      -s <code_dir>

If you are behind proxy, ads opctl will automatically use your proxy settings (defined via ``no_proxy``, ``http_proxy`` and ``https_proxy``).


**Define your workload yaml:**

The ``yaml`` file is a declarative way to express the workload.
Following is the YAML for running the example code, you will need to replace the values in the `spec` sections for your project:

- ``infrastructure`` contains ``spec`` for OCI Data Science Jobs. Here you need to specify a subnet that allows communications between nodes. The ``VM.GPU2.1`` shape is used in this example.
- ``cluster`` contains ``spec`` for the image you built and a working directory on OCI object storage, which will be used by job runs to shared internal configurations. Environment variables specified in the ``cluster.spec.config`` will be available in all nodes. Here the ``NCCL_ASYNC_ERROR_HANDLING`` is used to enable the timeout for NCCL backend. The job runs will be terminated if the nodes failed to connect to each other in certain minutes as specified in your training code when calling ``init_process_group()``.
- ``runtime`` contains ``spec`` for the name of your training script, and the command line arguments for running the script. Here the ``nccl`` backend is used for communications between GPUs. For CPU training, you can use the ``gloo`` backend. The ``timeout`` argument specify the maximum minutes for the nodes to wait when calling ``init_process_group()``. This is useful for preventing the job runs to wait forever in case of node failure.

.. code-block:: yaml
  :caption: train.yaml
  :name: train.yaml

  kind: distributed
  apiVersion: v1.0
  spec:
    infrastructure:
      kind: infrastructure
      type: dataScienceJob
      apiVersion: v1.0
      spec:
        projectId: oci.xxxx.<project_ocid>
        compartmentId: oci.xxxx.<compartment_ocid>
        displayName: PyTorch-Distributed
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxx.<log_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.GPU2.1
        blockStorageSize: 50
    cluster:
      kind: pytorch
      apiVersion: v1.0
      spec:
        image: <region.ocir.io/my-tenancy/image-name>
        workDir: "oci://my-bucket@my-namespace/pytorch/distributed"
        config:
          env:
            - name: NCCL_ASYNC_ERROR_HANDLING
              value: '1'
        main:
          name: PyTorch-Distributed-main
          replicas: 1
        worker:
          name: PyTorch-Distributed-worker
          replicas: 3
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
        entryPoint: "train.py"
        args:
          - --data-dir
          - /home/datascience/data
          - --output-dir
          - /home/datascience/outputs
          - --backend
          - gloo
          - --timeout
          - 5


**Use ads opctl to create the cluster infrastructure and dry-run the workload:**

.. code-block:: bash

  ads opctl run -f train.yaml --dry-run

the output from the dry run will show all the actions and infrastructure configuration.

**Use ads opctl to create the cluster infrastructure and run the workload:**

.. include:: ../_test_and_submit.rst

.. _hvd_saving_artifacts:

.. include:: ../_save_artifacts.rst

.. code-block:: python

  model_path = os.path.join(os.environ.get("OCI__SYNC_DIR"),"model.pt")
  torch.save(model, model_path)

**Profiling**

You may want to profile your training setup for optimization/performance tuning. Profiling typically provides a detailed analysis of cpu utilization, gpu utilization,
top cuda kernels, top operators etc. You can choose to profile your training setup using the native Pytorch profiler or using a third party profiler such as `Nvidia Nsights <https://developer.nvidia.com/nsight-systems>`__.

**Profiling using Pytorch Profiler**

Pytorch Profiler is a native offering from Pytorch for Pytorch performance profiling. Profiling is invoked using code instrumentation using the api ``torch.profiler.profile``.

Refer `this link <https://pytorch.org/docs/stable/profiler.html>`__ for changes that you need to do in your training script for instrumentation.
You should choose the ``OCI__SYNC_DIR`` directory to save the profiling logs. For example:

.. code-block:: python

  prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.environ.get("OCI__SYNC_DIR") + "/logs"),
        with_stack=False)
  prof.start()

  # training code
  prof.end()

Also, the sync feature ``SYNC_ARTIFACTS`` should be enabled ``'1'`` to sync the profiling logs to the configured object storage.

You would also need to install the Pytorch Tensorboard Plugin.

.. code-block:: bash

   pip install torch-tb-profiler

Thereafter, use Tensorboard to view logs. Refer the :doc:`Tensorboard setup <../../tensorboard/tensorboard>` for set-up on your computer.


.. include:: ../_profiling.rst