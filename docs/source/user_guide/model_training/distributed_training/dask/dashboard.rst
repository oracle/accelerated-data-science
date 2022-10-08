==============
Dask dashboard
==============

``Dask`` dashboard allows you to monitor the progress of the tasks. It gives you a real time view of the resource usage, task status, number of workers, task distribution, etc. To learn more about ``Dask`` dashboard refer `this link <https://docs.dask.org/en/stable/diagnostics-distributed.html>`_.


**Prerequisite**

1. IP address of the Main/Scheduler Node. Use ``ads opctl distributed-training show-config`` or find the IP address from the logs of the main job run. 
2. The default port is ``8787``. You can override this port in ``cluster/main/config/startOptions`` in the cluster definition file.
3. Allow ingress to the port ``8787`` in the security list associated with the Subnet of Main/Scheduler node.

The dashboard is accessible over ``<SCHEDULER_IP>:8787``. The IP address may not always be accessible from your workstation especially if you are using a subnet which is not connected to your corporate network. To overcome this, you could setup a bastion host on the private regional subnet that was added to the jobrun and create an ssh tunnel from your workstation to bastion host to the Job Run instance with ``<SCHEDULER_IP>``


++++++++++++
Bastion Host
++++++++++++

Here are the steps to setup a ``Bastion`` host to allow you to connect to the scheduler dashboard - 

1. Launch a compute instance (Linux or Windows) with primary vnic with a public subnet or the subnet that is connected to your corporate network.
2. Attach a secondary VNIC on the subnet used for starting the cluster. Follow the steps detailed `here <https://docs.oracle.com/en-us/iaas/Content/Network/Tasks/managingVNICs.htm#Linux>`_ on how to setup and configure the host to setup the secondary VNIC.
3. Create a public IP if you need access to the dashboard over the internet.

Linux instance
..............

If you setup a Linux instance, you can create ssh tunnel from your workstation and access the scheduler dashboard from your workstation at  ``localhost:8787``. To setup ssh tunnel -  

.. code-block:: shell

  ssh -i <oci-instance-key>.key <ubuntu or opc>@<instance-ip> L 8787:<scheduler jobrun-ip>:8787

If you are using proxy, use this command - 

.. code-block:: shell

  ssh -i <oci-instance-key>.key <ubuntu or opc>@<instance-ip> -o “ProxyCommand=nc -X connect -x $http_proxy:$http_port %h %p” -L 8787:<scheduler jobrun-ip>:8787

Windows instance
................

RDP to the Windows instance and access the dashboard using ``<SCHEDULER_IP>:8787`` from a browser running within the Windows instance.
