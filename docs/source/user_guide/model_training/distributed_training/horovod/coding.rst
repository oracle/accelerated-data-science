Writing Distributed code with Horovod Framework
-----------------------------------------------

TensorFlow
''''''''''

**To use Horovod in TensorFlow, following modifications are required in the training script:**



1. Import Horovod and initialize it.

.. code-block:: python


  import horovod.tensorflow as hvd
  hvd.init()

2. Pin each GPU to a single process.

With **TensorFlow v1.**

.. code-block:: python


  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(hvd.local_rank())

With **TensorFlow v2.**

.. code-block:: python


  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


3. Scale the learning rate by the number of workers.

.. code-block:: python


  opt = tf.keras.optimizers.SGD(0.0005 * hvd.size())

4. Wrap the optimizer in ``hvd.DistributedOptimizer``.

.. code-block:: python


  opt = hvd.DistributedOptimizer(opt)

5. Modify your code to save checkpoints(and any other artifacts) only in the rank-0 training process to prevent other workers from corrupting them.

.. code-block:: python


  if hvd.rank() == 0:
    tf.keras.callbacks.ModelCheckpoint(ckpts_path)
    tf.keras.callbacks.TensorBoard(tblogs_path)

.. _hvd_state_sync:

6. OCI Data Science Horovod workloads are based on Elastic Horovod. In addition to above changes, the training script also needs
to use `state synchronization <https://horovod.readthedocs.io/en/stable/elastic_include.html#modifying-the-training-script-with-state-synchronization>`_.
In summary, this means:

a. Use the decorator ``hvd.elastic.run`` to wrap the main training process.

b. Use ``hvd.elastic.State`` to add all variables that needs to be sync across workers.

c. Save state periodically, using ``hvd.elastic.State``

A complete example can be found in the :ref:`Write your training code <hvd_training_code>` section.
More examples can be found `here <https://github.com/horovod/horovod/tree/master/examples/elastic/tensorflow2>`_.
Refer `horovod with TensorFlow <https://horovod.readthedocs.io/en/stable/tensorflow.html>`_  and `horovod with Keras <https://horovod.readthedocs.io/en/stable/keras.html>`_ for more details.

PyTorch
'''''''

**To use Horovod in PyTorch, following modifications are required in the training script:**


1. Import Horovod and initialize it.

.. code-block:: python


  import horovod.torch as hvd
  hvd.init()


2. Pin each GPU to a single process. (use ``hvd.local_rank()``)

.. code-block:: python


  torch.manual_seed(args.seed)
  if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)


3. Scale the learning rate by the number of workers. (use ``hvd.size()``)

.. code-block:: python


  optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                      momentum=args.momentum)



4. Wrap the optimizer in ``hvd.DistributedOptimizer``.

.. code-block:: python


  optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    compression=compression,
    op=hvd.Adasum if args.use_adasum else hvd.Average
  )



5. Modify your code to save checkpoints only in the rank-0 training process to prevent other workers from corrupting them.

6. Like TensorFlow, Horovod PyTorch scripts also need to use `state synchronization <https://horovod.readthedocs.io/en/stable/elastic_include.html#modifying-the-training-script-with-state-synchronization>`_.
Refer TensorFlow section :ref:`above <hvd_state_sync>`.


Here is a complete PyTorch sample which is inspired from examples found
`here <https://github.com/horovod/horovod/blob/master/examples/elastic/pytorch/pytorch_mnist_elastic.py>`__ and
`here <https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py>`__.

.. code-block:: python
  :caption: train.py
  :name: train.py

  # Script adapted from https://github.com/horovod/horovod/blob/master/examples/elastic/pytorch/pytorch_mnist_elastic.py

  # ==============================================================================
  import argparse
  import os
  from filelock import FileLock

  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torchvision import datasets, transforms
  import torch.utils.data.distributed
  import horovod.torch as hvd
  from torch.utils.tensorboard import SummaryWriter

  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
                      help='random seed (default: 42)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                      help='use fp16 compression during allreduce')
  parser.add_argument('--use-adasum', action='store_true', default=False,
                      help='use adasum algorithm to do reduction')
  parser.add_argument('--data-dir',
                      help='location of the training dataset in the local filesystem (will be downloaded if needed)')

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  checkpoint_format = 'checkpoint-{epoch}.pth.tar'

  # Horovod: initialize library.
  hvd.init()
  torch.manual_seed(args.seed)

  if args.cuda:
      # Horovod: pin GPU to local rank.
      torch.cuda.set_device(hvd.local_rank())
      torch.cuda.manual_seed(args.seed)


  # Horovod: limit # of CPU threads to be used per worker.
  torch.set_num_threads(1)

  kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
  data_dir = args.data_dir or './data'
  with FileLock(os.path.expanduser("~/.horovod_lock")):
      train_dataset = \
          datasets.MNIST(data_dir, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
  # Horovod: use DistributedSampler to partition the training data.
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

  test_dataset = \
      datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ]))
  # Horovod: use DistributedSampler to partition the test data.
  test_sampler = torch.utils.data.distributed.DistributedSampler(
      test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            sampler=test_sampler, **kwargs)


  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
          self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
          self.conv2_drop = nn.Dropout2d()
          self.fc1 = nn.Linear(320, 50)
          self.fc2 = nn.Linear(50, 10)

      def forward(self, x):
          x = F.relu(F.max_pool2d(self.conv1(x), 2))
          x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
          x = x.view(-1, 320)
          x = F.relu(self.fc1(x))
          x = F.dropout(x, training=self.training)
          x = self.fc2(x)
          return F.log_softmax(x)


  model = Net()

  # By default, Adasum doesn't need scaling up learning rate.
  lr_scaler = hvd.size() if not args.use_adasum else 1

  if args.cuda:
      # Move model to GPU.
      model.cuda()
      # If using GPU Adasum allreduce, scale learning rate by local_size.
      if args.use_adasum and hvd.nccl_built():
          lr_scaler = hvd.local_size()

  # Horovod: scale learning rate by lr_scaler.
  optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                        momentum=args.momentum)

  # Horovod: (optional) compression algorithm.
  compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none


  def metric_average(val, name):
      tensor = torch.tensor(val)
      avg_tensor = hvd.allreduce(tensor, name=name)
      return avg_tensor.item()

  def create_dir(dir):
      if not os.path.exists(dir):
          os.makedirs(dir)
  # Horovod: average metrics from distributed training.
  class Metric(object):
      def __init__(self, name):
          self.name = name
          self.sum = torch.tensor(0.)
          self.n = torch.tensor(0.)

      def update(self, val):
          self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
          self.n += 1

      @property
      def avg(self):
          return self.sum / self.n

  @hvd.elastic.run
  def train(state):
      # post synchronization event (worker added, worker removed) init ...

      artifacts_dir = os.environ.get("OCI__SYNC_DIR") + "/artifacts"
      chkpts_dir = os.path.join(artifacts_dir,"ckpts")
      logs_dir = os.path.join(artifacts_dir,"logs")
      if hvd.rank() == 0:
          print("creating dirs for checkpoints and logs")
          create_dir(chkpts_dir)
          create_dir(logs_dir)

      writer = SummaryWriter(logs_dir) if hvd.rank() == 0 else None

      for state.epoch in range(state.epoch, args.epochs + 1):
          train_loss = Metric('train_loss')
          state.model.train()

          train_sampler.set_epoch(state.epoch)
          steps_remaining = len(train_loader) - state.batch

          for state.batch, (data, target) in enumerate(train_loader):
              if state.batch >= steps_remaining:
                  break

              if args.cuda:
                  data, target = data.cuda(), target.cuda()
              state.optimizer.zero_grad()
              output = state.model(data)
              loss = F.nll_loss(output, target)
              train_loss.update(loss)
              loss.backward()
              state.optimizer.step()
              if state.batch % args.log_interval == 0:
                  # Horovod: use train_sampler to determine the number of examples in
                  # this worker's partition.
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      state.epoch, state.batch * len(data), len(train_sampler),
                      100.0 * state.batch / len(train_loader), loss.item()))
              state.commit()
          if writer:
             writer.add_scalar("Loss", train_loss.avg, state.epoch)
          if hvd.rank() == 0:
              chkpt_path = os.path.join(chkpts_dir,checkpoint_format.format(epoch=state.epoch + 1))
              chkpt = {
                  'model': state.model.state_dict(),
                  'optimizer': state.optimizer.state_dict(),
              }
              torch.save(chkpt, chkpt_path)
          state.batch = 0


  def test():
      model.eval()
      test_loss = 0.
      test_accuracy = 0.
      for data, target in test_loader:
          if args.cuda:
              data, target = data.cuda(), target.cuda()
          output = model(data)
          # sum up batch loss
          test_loss += F.nll_loss(output, target, size_average=False).item()
          # get the index of the max log-probability
          pred = output.data.max(1, keepdim=True)[1]
          test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

      # Horovod: use test_sampler to determine the number of examples in
      # this worker's partition.
      test_loss /= len(test_sampler)
      test_accuracy /= len(test_sampler)

      # Horovod: average metric values across workers.
      test_loss = metric_average(test_loss, 'avg_loss')
      test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

      # Horovod: print output only on first rank.
      if hvd.rank() == 0:
          print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
              test_loss, 100. * test_accuracy))


  # Horovod: wrap optimizer with DistributedOptimizer.
  optimizer = hvd.DistributedOptimizer(optimizer,
                                       named_parameters=model.named_parameters(),
                                       compression=compression,
                                       op=hvd.Adasum if args.use_adasum else hvd.Average)


  # adjust learning rate on reset
  def on_state_reset():
      for param_group in optimizer.param_groups:
          param_group['lr'] = args.lr * hvd.size()


  state = hvd.elastic.TorchState(model, optimizer, epoch=1, batch=0)
  state.register_reset_callbacks([on_state_reset])
  train(state)
  test()


Refer to more examples `here <https://github.com/horovod/horovod/tree/master/examples/elastic/pytorch>`__.
Refer `horovod with PyTorch <https://horovod.readthedocs.io/en/stable/pytorch.html>`_ for more details.

**Next Steps**

Once you have the training code ready (either in TensorFlow or PyTorch), you can proceed to :doc:`creating Horovod workloads<creating>`.
