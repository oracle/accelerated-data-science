+++++++++++++++++++++++++++++++++
Build Development Container Image 
+++++++++++++++++++++++++++++++++

To setup an environment that matches OCI Data Science, a container image must be built. With a Data Science compatible container image you can do the following - 

* Build and Publish custom conda packs that can be used within Data Science environment. Enable building conda packs in your CICD pipeline.
* Install an existing conda pack that was published from an OCI Data Science Notebook.
* Develop code locally against the same conda pack that will be used within an OCID Data Science image.

**Prerequisites**

1. Install docker on your workstation
2. Internet connection to pull dependencies
3. If the access is restricted through proxy - 
    - Setup proxy environment variables ``https_proxy``, ``https_proxy`` and ``no_proxy``
    - For ``Linux`` Workstation - update proxy variables in ``docker.service`` file and restart docker
    - For ``mac`` - update proxy setting in the docker desktop
4. ADS cli is installed. Check CLI Installation section :doc:`here<../../quickstart>`

Build a container image with name ``ml-job``

.. code-block:: shell

  ads opctl build-image job-local

