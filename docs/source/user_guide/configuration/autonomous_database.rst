.. _configuration-autonomous_database:

Autonomous Database
*******************

There are two different configurations of the Autonomous Database (ADB). They are the Autonomous Data Warehouse (ADW) and the Autonomous Transaction Processing (ATP). The steps to connect to ADW and ATP are the same.  To access an instance 
of the ADB from the notebook environment, you need the client credentials and connection information. The client credentials include the wallet, which is required for all types of connections.

Use these steps to access Oracle ADB:

1. From the ADW or ATP instance page that you want to load a dataset from, click ``DB Connection``.

.. figure:: figures/DB-Connection.png
     :align: center

2. Click ``Download Wallet`` to download the wallet file. You need to create a password to for the wallet to complete the download. You don't need this password to connect from the notebook. 
  
3. Unzip the wallet.

.. figure:: figures/Download-Wallet.png
     :align: center

4. Create a ``<path_to_wallet_folder>`` folder for your wallet on the notebook environment environment. 

5. Upload your wallet files into the ``<path_to_wallet_folder>`` folder using the Jupyterlab **Upload Files**:

.. figure:: figures/Upload_Wallet.png
     :align: center

6. Open the ``sqlnet.ora`` file from the wallet files, then configure the ``METHOD_DATA``:

.. code-block:: bash

  METHOD_DATA = (DIRECTORY="<path_to_wallet_folder>")

7. To find the location of the ``sqlnet.ora`` file, the ``TNS_ADMIN`` environment variable must point to that location. We suggest that you create a Python dictionary to store all of the connection information. In this example, this dictionary is called ``creds``. It is generally poor security practice to store credentials in your notebook. We recommend that you use the ``ads-examples/ADB_working_with.ipynb`` notebook example that demonstrates how to store them outside the notebook in a configuration file.

   The environment variable should be set in your notebooks. For example: 

.. code-block:: python3

  # Replace with your TNS_ADMIN value here:
  creds = {}
  creds['tns_admin'] = <path_to_wallet_folder>
  os.environ['TNS_ADMIN'] = creds['tns_admin']

8. You can find SID names from the ``tnsname.ora`` file in the wallet file. Create a dictionary to manage your credentials. In this example, the variable ``creds`` is used. The SID is an identifier that identifies the consumer group of the the Oracle Database:

.. code-block:: python3

  # Replace with your SID name here:
  creds['sid'] = <your_SID_name>

9. Ask your database administrator for the username and password, and then add them to your ``creds`` dictionary. For example:

.. code-block:: python3

  creds['user'] = <database_user>
  creds['password'] = <database_password>

10. Test the connection to the ADB by running these commands:

.. code-block:: python3

  os.environ['TNS_ADMIN'] = creds['tns_admin']
  connect = 'sqlplus ' + creds['user'] + '/' + creds['password'] + '@' + creds['sid']
  print(os.popen(connect).read())

Messages similar to the following display if the connection is successful:

.. figure:: figures/Test_connection.png
     :align: center

An introduction to loading data from ADB into ADS using ``cx_Oracle`` and ``SQLAlchemy`` is in :ref:`Loading Data <loading-data-10>`.

