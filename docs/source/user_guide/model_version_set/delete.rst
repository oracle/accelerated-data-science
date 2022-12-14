Delete
______

To delete a model version set, all the associated models must be deleted or in a terminated state. You can set the ``delete_model`` parameter to ``True`` to delete all of the models in the model version set, and then delete the model version set. The ``.delete()`` method on a ``ModelVersionSet`` object initiates an asynchronous delete operation. You can check the ``.status`` method on the ``ModelVersionSet`` object to determine the status of the delete request. 


The following example deletes a model version set and its associated models. 

.. code-block: python3

   mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
   mvs.delete(delete_model=True)


The ``status`` property has the following values:

* ``ModelVersionSet.LIFECYCLE_STATE_ACTIVE``
* ``ModelVersionSet.LIFECYCLE_STATE_DELETED``
* ``ModelVersionSet.LIFECYCLE_STATE_DELETING``
* ``ModelVersionSet.LIFECYCLE_STATE_FAILED``

