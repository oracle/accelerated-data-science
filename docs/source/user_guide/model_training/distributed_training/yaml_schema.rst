===========
YAML Schema
===========

The distributed training workload is defined in ``YAML`` and can be launched by invoking the ``ads opctl run -f path/to/yaml`` command.

.. raw:: html
  :file: ../../../yaml_schema/jobs/distributed.html

|

Following is the YAML schema for validating the YAML using `Cerberus <https://docs.python-cerberus.org/en/stable/>`_:

.. literalinclude:: ../../../yaml_schema/jobs/distributed.yaml
  :language: yaml
  :linenos:
