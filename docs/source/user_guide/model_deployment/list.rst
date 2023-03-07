Inventory
*********

List
====

The ``.list()`` method of the ``ModelDeployment`` class returns a list of ``ModelDeployment`` objects.

The code snippet obtains a list of active deployments in the compartment specified by ``compartment_id``, and prints the display name.

.. code-block:: python3

    from ads.model.deployment import ModelDeployment

    for active in ModelDeployment.list(status="ACTIVE", compartment_id=compartment_id):
        print(active.display_name)

Show
====

The ``.list_df()`` method is a helper function that works the same way as the ``.list()`` method except it returns a dataframe of the results.

.. code-block:: python3

    from ads.model.deployment import ModelDeployment

    ModelDeployment.list_df(compartment_id=compartment_id, status="ACTIVE")

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
          <th>deployment_id</th>
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

