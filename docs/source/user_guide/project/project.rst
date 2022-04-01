.. _project-8:

========
Projects
========

``Projects`` is a resource of the Data Science service, and ADS provides an interface to perform operations on the projects.

Listing Projects
----------------

List projects by providing a compartment OCID, and then using the ``list_projects()`` method. Before listing the projects, you must first create or have instances of the ``Project Catalog`` object.

.. code-block:: python3

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    pc = ProjectCatalog(compartment_id=compartment_id)
    pc.list_projects()

This is an example of the output table:

.. image:: figures/project_list.png

Reading a Project Metadata
--------------------------

From the project list, obtain the OCID of a project that you want to retrieve by using ``get_project()``.

.. code-block:: python3

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    pc = ProjectCatalog(compartment_id=compartment_id)
    my_project = pc.get_project(pc.list_projects()[0].id)

Creating a Project
------------------

Using the ``ProjectCatalog`` object, create a project by calling the ``create_project()`` method and specifying the compartment id.

.. code-block:: python3

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    pc = ProjectCatalog(compartment_id=compartment_id)
    new_project = pc.create_project(display_name='new_project',
                               description='this is a test project',
                               compartment_id=compartment_id)

Updating a Project
------------------

Projects can be updated in a similar way as models.  You must call the ``commit()`` function, to push the changes to the project catalog.

.. code-block:: python3

    new_project.description = 'a new description'
    new_project.display_name = 'a new name from ads'
    new_project.commit()

Deleting a Project
------------------

Projects can be deleted by specifying the project id.

.. code-block:: python3

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    pc = ProjectCatalog(compartment_id=compartment_id)
    pc.delete_project(new_project.id)
