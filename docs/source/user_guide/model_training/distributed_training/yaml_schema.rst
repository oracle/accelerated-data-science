===========
YAML Schema
===========

The distributed training workload is defined in ``YAML`` and can be launched by invoking the ``ads opctl run -f path/to/yaml`` command.

.. raw:: html
  :file: schema.html

|

Following is the YAML schema for validating the YAML using `Cerberus <https://docs.python-cerberus.org/en/stable/>`_:

.. code-block:: yaml
  :linenos:

  kind:
    type: string
    allowed:
    - distributed
  apiVersion:
    type: string
  spec:
    type: dict
    schema:
      infrastructure:
        type: dict
        schema:
          kind:
            type: string
            allowed:
            - infrastructure
          type:
            type: string
            allowed:
            - dataScienceJob
          apiVersion:
            type: string
          spec:
            type: dict
            schema:
              displayName:
                type: string
              compartmentId:
                type: string
              projectId:
                type: string
              logGroupId:
                type: string
              logId:
                type: string
              subnetId:
                type: string
              shapeName:
                type: string
              blockStorageSize:
                type: integer
                min: 50
      cluster:
        type: dict
        schema:
          kind:
            type: string
            allowed:
            - PYTORCH
            - DASK
            - HOROVOD
            - dask
            - pytorch
            - horovod
          apiVersion:
            type: string
          spec:
            type: dict
            schema:
              image:
                type: string
              workDir:
                type: string
              name:
                type: string
              config:
                type: dict
                nullable: true
                schema:
                  startOptions:
                    type: list
                    schema:
                      type: string
                  env:
                    type: list
                    nullable: true
                    schema:
                      type: dict
                      schema:
                        name:
                          type: string
                        value:
                          type:
                          - number
                          - string
              main:
                type: dict
                schema:
                  name:
                    type: string
                  replicas:
                    type: integer
                  config:
                    type: dict
                    nullable: true
                    schema:
                      env:
                        type: list
                        nullable: true
                        schema:
                          type: dict
                          schema:
                            name:
                              type: string
                            value:
                              type:
                              - number
                              - string
              worker:
                type: dict
                schema:
                  name:
                    type: string
                  replicas:
                    type: integer
                  config:
                    type: dict
                    nullable: true
                    schema:
                      env:
                        type: list
                        nullable: true
                        schema:
                          type: dict
                          schema:
                            name:
                              type: string
                            value:
                              type:
                              - number
                              - string
      runtime:
        type: dict
        schema:
          kind:
            type: string
          apiVersion:
            type: string
          spec:
            type: dict
            schema:
              entryPoint:
                type: string
              kwargs:
                type: string
              args:
                type: list
                schema:
                  type:
                  - number
                  - string
              env:
                type: list
                nullable: true
                schema:
                  type: dict
                  schema:
                    name:
                      type: string
                    value:
                      type:
                      - number
                      - string
