Securing with TLS
-----------------

You can setup ``Dask`` cluster to run using ``TLS``. To do so, you need three things - 

1. CA Certificate
2. A Certificate signed by CA
3. Private key of the certificate

For more details refer `Dask documentation <https://distributed.dask.org/en/stable/tls.html>`_



**Self signed Certificate using openssl**

``openssl`` lets you create test CA and certificates required to setup ``TLS`` connectivity for ``Dask`` cluster. Use the commands below to create certificate in your code folder. When the container image is built, all the artifacts in the code folder is copied to ``/code`` directory inside container image.


1. Generate CA Certificate

.. code-block:: shell

  openssl req -x509 -nodes -newkey rsa:4096 -days 10 -keyout dask-tls-ca-key.pem -out dask-tls-ca-cert.pem -subj "/C=US/ST=CA/CN=ODSC CLUSTER PROVISIONER"

2. Generate CSR

.. code-block:: shell
  
  openssl req -nodes -newkey rsa:4096 -keyout dask-tls-key.pem -out dask-tls-req.pem -subj "/C=US/ST=CA/CN=DASK CLUSTER"

3. Sign CSR

.. code-block:: shell

  openssl x509 -req -in dask-tls-req.pem -CA dask-tls-ca-cert.pem -CAkey dask-tls-ca-key.pem -CAcreateserial -out dask-tls-cert.pem

4. Follow the container build instrcutions :doc:`here <creating>` to build, tag and push the image to ``ocir``.
5. Create a cluster definition ``YAML`` and configure the certifacte information under ``cluster/config/startOptions``. Here is an example - 

.. code-block:: yaml

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
        displayName: my_distributed_training
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxx.<log_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.Standard2.4
        blockStorageSize: 50
    cluster:
      kind: dask
      apiVersion: v1.0
      spec:
        image: iad.ocir.io/mytenancy/dask-cluster-examples:dev
        workDir: oci://mybucket@mytenancy/daskexample/001
        name: LGBM Dask
        main:
          config:
              startOptions:
                  - --tls-ca-file /code/dask-tls-ca-cert.pem
                  - --tls-cert /code/dask-tls-cert.pem
                  - --tls-key /code/dask-tls-key.pem
        worker:
          config:
              startOptions:
                  - --tls-ca-file /code/dask-tls-ca-cert.pem
                  - --tls-cert /code/dask-tls-cert.pem
                  - --tls-key /code/dask-tls-key.pem   
          replicas: 2
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
        entryPoint: lgbm_dask.py
        env:
          - name: SIZE
            value: 1000000

**Using OCI Certificate manager**
 
See `OCI Certificates <https://docs.oracle.com/en-us/iaas/Content/certificates/home.htm>`_ for
reference. In this approach, the Admin of the tenancy or the person with the requisite permission can create and manage certificate on OCI console. Sepcify the OCID of the CA Certificate, TLS Certificate and Private Key of the Certificate in `cluster/certificates` option.
 
**Policies Required**::
 
  # Create DG with resource.type='certificateauthority'
 
  Allow dynamic-group certauthority-resource to use keys in compartment <my-compartment-name>
  Allow dynamic-group certauthority-resource to manage objects in compartment <my-compartment-name>
 
 
1. Create certificate authority, certificate and private key inside ``OCI Certificates`` console.
2. Create a cluster definition ``YAML`` and configure the certifacte information under ``cluster/config/startOptions``. Here is an example -
 
.. code-block:: yaml
 
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
        displayName: my_distributed_training
        logGroupId: oci.xxxx.<log_group_ocid>
        logId: oci.xxx.<log_ocid>
        subnetId: oci.xxxx.<subnet-ocid>
        shapeName: VM.Standard2.4
        blockStorageSize: 50
    cluster:
      kind: dask
      apiVersion: v1.0
      spec:
        image: iad.ocir.io/mytenancy/dask-cluster-examples:dev
        workDir: oci://mybucket@mytenancy/daskexample/001
        name: LGBM Dask
        certificate:
          caCert:
              id: ocid1.certificateauthority.oc1.xxx.xxxxxxx
              downloadLocation: /code/dask-tls-ca-cert.pem
          cert:
              id: ocid1.certificate.oc1.xxx.xxxxxxx
              certDownloadLocation: /code/dask-tls-cert.pem
              keyDownloadLocation: /code/dask-tls-key.pem
        main:
          config:
            startOptions:
             - --tls-ca-file /code/dask-tls-ca-cert.pem
             - --tls-cert /code/dask-tls-cert.pem
             - --tls-key /code/dask-tls-key.pem
        worker:
          config:
            startOptions:
              - --tls-ca-file /code/dask-tls-ca-cert.pem
              - --tls-cert /code/dask-tls-cert.pem
              - --tls-key /code/dask-tls-key.pem
          replicas: 2
    runtime:
      kind: python
      apiVersion: v1.0
      spec:
        entryPoint: lgbm_dask.py
        env:
          - name: SIZE
            value: 1000000