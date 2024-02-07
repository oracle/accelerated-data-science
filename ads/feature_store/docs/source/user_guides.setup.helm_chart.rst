=================================
Setup using Helm Charts
=================================

We always suggest to deploy feature store via the :doc:`Feature Store Operator <./user_guides.setup.feature_store_operator>` to setup Feature Store API server in OKE Cluster. This method should preferably be used only when the operator can not satisfy your requirements
as it is much simpler to do the setup via the operator.


Prerequisites
_____________

- Helm: Helm is required to be installed on the machine for deploying Feature Store helm chart to the Kubernetes cluster. Ref: `Installing Helm   <https://helm.sh/docs/intro/install/>`_
- Kubectl: Kubectl is required to be installed to deploy the helm chart to the cluster. Ref: `Installing Kubectl <https://kubernetes.io/docs/tasks/tools/>`_
- Setup `MySQL Database <https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/mysql-database/doc/overview-mysql-database-service.html>`_  so that it is reachable from the API server
.. seealso::
   :ref:`Database configuration`
- :ref:`Helm Setup`
- :ref:`Policies`
- `Setup cluster access locally <https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengdownloadkubeconfigfile.htm#:~:text=Under%20Containers%20%26%20Artifacts%2C%20click%20Kubernetes,shows%20details%20of%20the%20cluster>`_

Steps to deploy API server image
________________________________

- Export the package to OCIR using the `Feature store marketplace listing <https://cloud.oracle.com/marketplace/application/ocid1.mktpublisting.oc1.iad.amaaaaaabiudgxya26lzh2dsyvg7cfzgllvdl6xo5phz4mnsoktxeutecrvq>`_
- Wait for export work request to complete
- Identify the Helm chart and API images exported to OCIR:
   - Helm chart image would be of format: ``<ocir-image>:<version>``
   - API image would be of format: ``<ocir-image>:<export-number>-<image-id>-<tenancy-namespace>-feature-store-api-<version>``

- :ref:`Create kubernetes docker secret <Kubernetes secret>`

- :ref:`Generate custom values.yaml for deployment <Helm values>`

- Install the helm chart

.. code-block:: bash

   helm upgrade <app-name> oci://<helm-chart-image-path> --namespace <kubernetes-namespace> --values <path-to-values-yaml> --timeout 300s --wait -i --version <marketplace-version>
- (Optional) `Setup Feature Store API Gateway <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/feature_store/apigw_terraform>`_


Appendix
________

.. _Helm Setup:

Setup Helm to use OCIR
______________________

To login to Container Registry using the Helm CLI:

- If you already have an auth token, go to the next step. Otherwise:
   - In the top-right corner of the Console, open the Profile menu and then click User settings to view the details.
   - On the Auth Tokens page, click Generate Token.
   - Enter a friendly description for the auth token. Avoid entering confidential information.
   - Click Generate Token. The new auth token is displayed.
   - Copy the auth token immediately to a secure location from where you can retrieve it later, because you won't see the auth token again in the Console.
   - Close the Generate Token dialog.

- In a terminal window on the client machine running Docker, log in to Container Registry by entering  ``helm registry login <region-key>.ocir.io``, where <region-key> corresponds to the key for the Container Registry region you're using. For example, ``helm registry login iad.ocir.io``. See `Availability by Region <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryprerequisites.htm#regional-availability>`_.
- When prompted for a username, enter your username in the format <tenancy-namespace>/<username>, where <tenancy-namespace> is the auto-generated Object Storage namespace string of your tenancy (as shown on the Tenancy Information page). For example, ansh81vru1zp/jdoe@acme.com. If your tenancy is federated with Oracle Identity Cloud Service, use the format <tenancy-namespace>/oracleidentitycloudservice/<username>.
- When prompted for a password, enter the auth token you copied earlier.

.. _Kubernetes secret:

Kubernetes Docker Secret Configuration
__________________________________________________________
- If you don't already have an auth token refer :ref:`Helm configuration  <Helm Setup>`
- `Login to Kubernetes cluster <https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengdownloadkubeconfigfile.htm#:~:text=Under%20Containers%20%26%20Artifacts%2C%20click%20Kubernetes,shows%20details%20of%20the%20cluster>`_
- Run command

.. code-block:: bash

   kubectl create secret docker-registry <secret-name> --docker-server=<region-key>.ocir.io --docker-username=<tenancy-namespace>/<username> --docker-password=<auth token>

.. _Helm values:

Helm values configuration:
__________________________________________________________

- Minimal Helm values example for getting started:

.. code-block:: yaml

    db:
      configuredDB: MYSQL
      mysql:
         authType: BASIC
         basic:
            password: #enter-db-password-here
         jdbcURL: jdbc:mysql://<db-ip>:3306/FeatureStore?createDatabaseIfNotExist=true
         username: #enter-db-username-here
    imagePullSecrets:
    - name: #enter secret name containing docker secret here
    oci_meta:
      images:
         api:
            image: #ocir image: The name of image
            tag: #API Image tag
         authoriser:
            image: na
            tag: na
      repo: #ocir repo: <region-key>.ocir.io/<tenancy-namespace>/repository



- All available Helm values

.. code-block:: yaml

    oci_meta:
        repo: #ocir repo: <region-key>.ocir.io/<tenancy-namespace>/repository
        images:
          api:
             image: #ocir image: The name of image
             tag: #API Image tag
          authoriser: # We don't want to deploy this image. This image will be deployed with OCI functions
                image: na
                tag: na

    imagePullSecrets:
    - name:  #name-of-docker-secret-with-credentials

    db:
        configuredDB: #Type of DB configured. Possible values: "MYSQL"
        mysql:
          authType: #Type of authentication to use for connecting to database.
                    # Possible values: 'BASIC', 'VAULT'
          jdbcURL: #JDBC URL of the MySQL server
          username: #Name of the user on MySQL server
          basic:
             password: #Password to mysql server in plain-text format
          vault:
             vaultOcid: #OCID of the vault where the secret is kept
             secretName: #Name of the secret used for connecting to vault

    resources: #https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/

    nameOverride: #Value for label app.kubernetes.io/name

    podSecurityContext: #Pod security #https://kubernetes.io/docs/tasks/configure-pod-container/security-context/

    securityContext: #Container Security context #https://kubernetes.io/docs/tasks/configure-pod-container/security-context/

    deploymentStrategy: #This block is directly inserted into pod spec
                      #https://kubernetes.io/docs/concepts/workloads/controllers/deployment/


    nodeSelector: {} #Pod node selector
                   #https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/

    tolerations: []  #Pod tolerations
                    #https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/

    affinity: {}  #Pod affinity
                 #https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/

    replicaCount: #Pod replicas
                 #https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/

    autoscaling: #Horizontal pod scaling details
                #https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
      enabled:
      minReplicas:
      maxReplicas:
      targetCPUUtilizationPercentage:
      targetMemoryUtilizationPercentage:
    scaleUp:
      stabilizationWindowSeconds:
      periodSeconds:
      podCount:
      percentage:
    scaleDown:
      stabilizationWindowSeconds:
      periodSeconds:
      podCount:
      percentage:

    applicationEnv:
    containerName: #Container name

    livenessProbe: # Liveness probe details
      initialDelaySeconds:
      periodSeconds:
      timeoutSeconds:
      failureThreshold:

    readinessProbe: # Readiness probe details
      initialDelaySeconds:
      periodSeconds:
      timeoutSeconds:
      failureThreshold:
