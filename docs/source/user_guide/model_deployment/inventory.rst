Inventory
*********

List
====

The ``.list_deployments()`` method of the ``ModelDeployer`` class returns a list of ``ModelDeployment`` objects. The optional ``compartment_id`` parameter limits the search to a specific compartment. By default, it uses the same compartment that the notebook is in. The optional ``status`` parameter limits the returned ``ModelDeployment`` objects to those model deployments that have the specified status. Values for the ``status`` parameter would be ‘ACTIVE’, ‘INACTIVE’, or ‘FAILED’.

The code snippet obtains a list of active deployments in the compartment specified by ``compartment_id``, and prints the display name.

.. code-block:: python3

    from ads.model.deployment import ModelDeployer

    deployer = ModelDeployer()
    for active in deployer.list_deployments(status="ACTIVE", compartment_id=compartment_id):
        print(active.properties.display_name)

Show
====

The ``.show_deployments()`` method is a helper function that works the same way as the ``.list_deployments()`` method except it returns a dataframe of the results.

.. code-block:: python3

    from ads.model.deployment import ModelDeployer

    deployer = ModelDeployer()
    deployer.show_deployments(compartment_id=compartment_id, status="ACTIVE")

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>model_id</th>
          <th>deployment_url</th>
          <th>current_state</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ocid1.datasciencemodeldeployment..&l;tunique_ID&gt;</td>
          <td>https://modeldeployment.us-ashburn-1...</td>
          <td>ACTIVE</td>
        </tr>
      </tbody>
    </table>
    </div>

