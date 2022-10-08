===================================
Local Development Environment Setup
===================================

**Prerequisite**

* You have completed :doc:`ADS CLI installation <../quickstart>` 
* You have completed :doc:`Configuaration <configure>` 

Setup up your workstation for development and testing your code locally before you submit it as a OCI Data Science Job. This section will guide you on how to setup environment for - 

* Building an OCI Data Science compatible conda environments on your workstation or CICD pipeline and publishing to object storage
* Developing and testing code with a conda environment that is compatible with OCI Data Science Notebooks and OCI Data Science Jobs
* Developing and testing code for running Bring Your Own Container (BYOC) jobs.

**Note**

* In this version you cannot directly access the Service provided conda environments from ADS CLI, but you can publish a service provided conda pack from an OCI Data Science Notebook session to your object storage bucket and then use the CLI to access the published version. 

.. toctree::
  :hidden:
  :maxdepth: 1

  localdev/jobs_container_image
  localdev/vscode
  localdev/condapack
  localdev/jobs


