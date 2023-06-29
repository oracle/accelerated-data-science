===================================
Terraform: Setting up feature store
===================================

Oracle feature store is a stack based solution that is deployed in the customer enclave using OCI resource manager.
Customer can stand up the service with infrastructure in their own tenancy. The service consists of API in customer
tenancy using resource manager.

.. note::

  Blue-green deployment is a strategy for releasing new versions of an application with minimal downtime and risk. It is used in Kubernetes, as well as other deployment environments, to achieve a smooth transition between application versions with the ability to quickly rollback if issues are detected.
  In a blue-green deployment, there are two environments, named "blue" and "green," that run concurrently. One environment is designated as the live or production environment (let's say "blue"), while the other environment ("green") is idle or running a newer version of the application for testing. Both environments have identical configurations and infrastructure.

.. _User Policies:

User policies for stack setup
=============================

Prerequisites (For non admin users only)
#########################################

Feature Store users need to provide the following access permissions in order to deploy the feature store terraform stack. Below mentioned are the policy statements required for terraform stack deployment

..  code-block:: shell

    define tenancy <feature store service tenancy> as <feature store service tenancy ocid>
    endorse group <feature store user group> to read repos in tenancy <feature store service tenancy>
    allow group <feature store user group> to manage orm-stacks in compartment <compartmentId>
    allow group <feature store user group> to manage orm-jobs in compartment <compartmentId>
    allow group <feature store user group> to manage object-family in compartment <compartmentId>
    allow group <feature store user group> to manage users in compartment <compartmentId>
    allow group <feature store user group> to manage instance-family in compartment <compartmentId>
    allow group <feature store user group> to manage tag-namespaces in compartment <compartmentId>
    allow group <feature store user group> to manage groups in compartment <compartmentId>
    allow group <feature store user group> to manage policies in compartment <compartmentId>
    allow group <feature store user group> to manage dynamic-groups in compartment <compartmentId>
    allow group <feature store user group> to manage virtual-network-family in compartment <compartmentId>
    allow group <feature store user group> to manage functions-family in compartment <compartmentId>
    allow group <feature store user group> to inspect compartments in compartment <compartmentId>
    allow group <feature store user group> to manage cluster-family in compartment <compartmentId>
    allow group <feature store user group> to manage mysql-family in compartment <compartmentId>
    allow group <feature store user group> to manage api-gateway-family in compartment <compartmentId>

Deploy Using Oracle Resource Manager
====================================

.. note::

   If you aren't already signed in, when prompted, enter the tenancy and user credentials. Review and accept the terms and conditions. Refer :ref:`Release Notes` for getting the latest conda pack and ``SERVICE_VERSION``.

.. important::

    Refer :ref:`User Policies` to create feature store stack for non admin users. No policies are explicitly required for admin user.


1. Download the stack from ``Terraform Stack`` column in :ref:`Release Notes`. Refer :ref:`User Policies` to create feature store stack for non admin users. No policies are explicitly required for admin user.

2. Click to deploy the stack

3. Select the region and compartment where you want to deploy the stack.

4. Follow the on-screen prompts and instructions to create the stack.

5. After creating the stack, click Terraform Actions, and select Plan.

6. Wait for the job to be completed, and review the plan.

7. To make any changes, return to the Stack Details page, click Edit Stack, and make the required changes. Then, run the Plan action again.

8. If no further changes are necessary, return to the Stack Details page, click Terraform Actions, and select Apply.

Deploy Using the Oracle CLI
==============================

Prerequisites
#############

1. Install `oci-cli <https://docs.oracle.com/en-us/iaas/Content/API/Concepts/cliconcepts.htm>`__ if not installed

2. Download the stack from ``Terraform Stack`` column in :ref:`Release Notes`.

3. (Optional: Skip if default deployment is required) To use this file just copy the example ``terraform.tfvars.example`` and save it in the outermost directory.

4. (Optional: Skip if default deployment is required) Next, rename the file to ``terraform.tfvars``. You can override the example values set in this file.


Steps
#####

.. note::

  Refer :ref:`Release Notes` for getting the latest conda pack and ``SERVICE_VERSION``. Remember to replace the values within angle brackets ("<>" symbols) in the command above with the relevant values for your environment. Also, Refer :ref:`User Policies` to create feature store stack for non admin users. No policies are explicitly required for admin user.


1. Download the stack from ``Terraform Stack`` column in :ref:`Release Notes`.

2. Run the shell command.
  ..  code-block:: shell

    oci resource-manager stack create \
      --compartment-id <compartment-id> \
      --config-source <path-to-downloaded-zip-file> \
      --variables '{
        "service_version": "<SERVICE_VERSION>",
        "tenancy_ocid": "<TENANCY_OCID>",
        "compartment_ocid": "<COMPARTMENT_OCID>",
        "region": "<REGION>",
        "user_ocid": "<USER_OCID>"
      }' \
      --display-name "Feature Store Stack" \
      --working-directory "feature-store-terraform" \
      | tee stack_output.json \
      && stack_id=$(jq -r '.data."id"' stack_output.json) \
      && oci resource-manager job create-apply-job \
        --execution-plan-strategy AUTO_APPROVED \
        --stack-id $stack_id \
        --wait-for-state SUCCEEDED \
        --wait-for-state FAILED


Terraform Variables (Advanced)
===============================

A complete listing of the Terraform variables used in this stack are referenced below:

.. list-table:: Terraform Variables
   :header-rows: 1

   * - Variable Name
     - Value
     - Description
   * - `service_version`
     - `0.1-master.26`
     - The version of API to be deployed in customer tenancy.
   * - `spec_version`
     - `0.1-master.26`
     - The version of API specs to be deployed in customer tenancy.
   * - `deployment_name`
     - `DEFAULT_NAME`
     - Name of the deployment.
   * - `db_name`
     - `DEFAULT_NAME`
     - Name of ATP/MySQL database.
   * - `db_config`
     - `DEFAULT_NAME`
     - Config for db.
   * - `compartment_ocid`
     - `DEFAULT_NAME`
     - OCID of compartment to deploy the feature store stack.
   * - `vcn_details`
     - `DEFAULT_NAME`
     - VCN details required from user in case they are on-boarding the database which has network access within their VCN..
   * - `user_ocid`
     - `ocid1.user..<unique_id>`
     - If you do not have permission to create users, provide the user_ocid of a user that has permission to pull images from OCI Registry.
   * - `tenancy_ocid`
     - `ocid1.tenancy..<unique_id>`
     - OCID of tenancy to deploy the feature store stack.
   * - `ssh_authorized_key`
     - `<SSH AUTHORISED KEY>`
     - OCID of tenancy to deploy the feature store stack.
   * - `ocir_puller_auth_token`
     - `<AUTH TOKEN>`
     - If the user provided above already has an auth_token to use, provide it here. If null a new token will be created. This requires that the user has 1 token at most already (as there is a limit of 2 tokens per user)	.
   * - `ocir_puller_group_ocid`
     - `ocid1.group..<unique_id>`
     - If you have permission to create users, and a group already exists with policies to pull images from OCI Registry, you can provide the group_ocid and a new user will be created and be made a member of this group. Leave null if you are providing a ocir_puller_user_ocid	.
   * - `ocir_puller_user_ocid`
     - `ocid1.user..<unique_id>`
     - If you do not have permission to create users, provide the user_ocid of a user that has permission to create mysql and object storage buckets.
   * - `feature_store_user_group_id`
     - `ocid1.group..<unique_id>`
     - Provide the feature store user group id if the user is not an administrator.


.. note::
    Bring your own database (BYODB): Feature store does not support private access using private endpoint and private access gateway for ATP instances.

    User VCN Deployment
    ###################

    User can provide the existing VCN details in order for the feature store to use the existing VCN. Feature store terraform stack provides a terraform vcn variable which takes VCN details as mentioned below:

    .. list-table:: user_vcn
       :header-rows: 1

       * - Variable Name
         - Value
         - Description
       * - `vcn_id`
         - `ocid1.vcn.oc1.iad.xxxxxxxxxxxx`
         - The ocid of the VCN where user wants to deploy feature store.
       * - `vcn_cidr`
         - `10.0.0.0/16`
         - The VCN CIDR range to be used.
       * - `subnet_suffix`
         - `8`
         - The subnet suffix to be used in order to create  service related subnets(for e.g. 10.0.0.0/16 + 8 => 10.0.0.0/24).
       * - `max_subnet`
         - `16`
         - This is an optional variable which tells how many maximum subnet creations are allowed within the CIDR range.
       * - 'dhcp_options_id'
         - 'ocid1.dhcpoptions.oc1.iad.xxxxxxxxx'
         - DHCP options ocid is required for instance configuration within the VCN
       * - `igw_id`
         - `ocid1.internetgateway.oc1.iad.xxxxxxxxxxxx`
         - This is an optional variable which takes internet gateway ocid as an input. Feature store creates the IGW if not provided.
       * - `nat_gw_id`
         - `ocid1.natgateway.oc1.iad.xxxxxxxxxx`
         - This is an optional variable which takes nat gateway ocid as an input. Feature store creates the NAT gateway if not provided.
       * - `sgw_id`
         - `ocid1.servicegateway.oc1.iad.xxxxxxxxxx`
         - This is an optional variable which takes service gateway ocid as an input. Feature store does not create SGW even when its NULL or does not enable SGW till the time user explicitly using sgw_enable. This is done to ensure that SGW is only enabled for network resources (for e.g. ATP) which allow access through SGW.
       * - `sgw_enable`
         - `false`
         - Enable service gateway usage or creation depending upon the sgw_id provided by the user.
       * - `private_route_table_id`
         - `ocid1.routetable.oc1.iad.xxxxxxxxxxxxxxxx`
         - This is an optional variable which takes private route table ocid as an input. If user provides this then it would be user's reponsibilty to ensure that the nat gateway & service gateway(if applicable) route rules have been added for feature store service access. Feature store creates a private route table with supporting route rules if not provided.
       * - `public_route_table_id`
         - `ocid1.routetable.oc1.iad.xxxxxxxxxxxxxxxx`
         - This is an optional variable which takes public route table ocid as an input. If user provides this then it would be user's reponsibilty to ensure that the internet gateway route rule has been added for feature store service access. Feature store creates a public route table with supporting route rules if not provided.

    Feature store is deployed in feature store specific subnets and security list access are maintained on the basis of details provided by the user.


    User Input
    ##########

    User will need to provide the following details in order to onboard their own database instances.

    1. DB Config: This is general database configuration which is required for the purpose of initial database setup for BYODB or Feature store's own database setup.

    .. list-table:: db_config
       :header-rows: 1

       * - Variable Name
         - Value
         - Description
       * - `vault_ocid`
         - `ocid1.vault.oc1.iad.b5sb3bclaaaog.xxxxxxxxxxxxx`
         - The ocid of the vault where user has kept the  atp / mysql secret. This can be set to null in case of  Feature store's own db setup.
       * - `vault_compartment_id`
         - `ocid1.tenancy.oc1.iad.b5sb3bclaaaog.xxxxxxxxxxxxx`
         - The ocid of the vault compartment where user has created vault. This can be set to null in case of  Feature store's own db setup.
       * - `db_type`
         - `mysql`
         - The database type could be mysql /atp.
       * - `db_secret_source`
         - `VAULT`
         - The database secret source. It should be kept VAULT for BYODB use case. It can be OSS for ATP and LOCAL for MYSQL in case of default feature store deployment without BYODB.
       * - `user_db`
         - `false`
         - set user db to true to enable customer database support (BYODB).

    2. User DB Config: User specific details in order to onboard user database, all of these fields can be kept null if user database is not onboarded

    .. list-table:: user_db_config
       :header-rows: 1

       * - Variable Name
         - Value
         - Description
       * - `db_system_name`
         - `featurestoretestatp_xxxx`
         - The database instance name.
       * - `db_username`
         - `admin`
         - The username for the database.
       * - `db_password_secret`
         - `test_atp_xxxx`
         - Vault database password secret

    .. tabs::

       .. tab:: MySQL DB Config: MySQL database configuration

          MySQL Instance can only be accessed within the network and for that user will need to deploy the Feature store within their VCN. User will also provide the vault secret for the MySQL database password.
          Please ensure that ingress rules are in place to provide VCN access for the MySQL instance.
          Please refer to this link (https://docs.oracle.com/en-us/iaas/mysql-database/doc/networking-setup-mysql-db-systems.html) for more details:

          .. list-table:: mysql_db_config
             :header-rows: 1

             * - Variable Name
               - Value
               - Description
             * - `mysql_db_ip_addr`
               - `192.168.xxx.xxxx`
               - MySQL database IP address.
             * - `mysql_db_port`
               - `3306`
               - mysql db port

       .. tab:: ATP DB Config: ATP database configuration

          The existing ATP instance can have two type of access:

              1. Public Access: In this case the ATP instance can be accessed either with Feature store deployed in its own VCN or in user VCN. User will need to provide the vault secret names which will be used for ATP connection.

              2. Network Access: If ATP instance has network access within VCN only then in cases like these User need to deploy feature store in VCN which has ATP access. User will need to provide the vault secret names which will be used for ATP connection.


          .. list-table:: atp_db_config
             :header-rows: 1

             * - Variable Name
               - Value
               - Description
             * - `wallet_file_secret`
               - `["cwallet.sso", "ewallet.p12", "keystore.jks", "ojdbc.properties", "tnsnames.ora", "truststore.jks", "sqlnet.ora"]`
               - List of ATP Wallet files vault secrets base64 encoded. Please ensure to encode the wallet files to base64 format and then push them as base64 encoded string to Vault.
             * - `wallet_password_secret`
               - `example-secret`
               - Vault wallet password secret
