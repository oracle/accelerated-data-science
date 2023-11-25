=============
Configure PII
=============

Let's explore each line of the pii.yaml so we can better understand options for extending and customizing the operator to our use case.

Here is an example pii.yaml with every parameter specified:

.. code-block:: yaml

    kind: operator
    type: pii
    version: v1
    spec:
        output_directory:
            url: oci://my-bucket@my-tenancy/results
            name: mydata-out.csv
        report:
            report_filename: report.html
            show_rows: 10
            show_sensitive_content: true
        input_data:
            url: oci://my-bucket@my-tenancy/mydata.csv
        target_column: target
        detectors:
            - name: default.phone
              action: anonymize


* **Kind**: The yaml file always starts with ``kind: operator``. There are many other kinds of yaml files that can be run by ``ads opctl``, so we need to specify this is an operator.
* **Type**: The type of operator is ``pii``.
* **Version**: The only available version is ``v1``.
* **Spec**: Spec contains the bulk of the information for the specific problem.
    * **input_data**: This dictionary contains the details for how to read the input data.
        * **url**: Insert the uri for the dataset if it's on object storage using the URI pattern ``oci://<bucket>@<namespace>/path/to/data.csv``.
    * **target_column**: This string specifies the name of the column where the user data is within the input data.
    * **detectors**: This list contains the details for each detector and action that will be taken.
        * **name**: The string specifies the name of the detector. The format should be ``<type>.<entity>``. Check :ref:`Configure Detector <config_detector>` for more details.
        * **action**: The string specifies the way to process the detected entity. Default to mask.
    * **output_directory**: This dictionary contains the details for where to put the output artifacts. The directory need not exist, but must be accessible by the Operator during runtime.
        * **url**: Insert the uri for the dataset if it's on object storage using the URI pattern ``oci://<bucket>@<namespace>/subfolder/``.
        * **name**: The string specifies the name of the processed data file.

    * **report**: (optional) This dictionary specific details for the generated report.
        * **report_filename**: Placed into output_directory location. Defaults to ``report.html``.
        * **show_sensitive_content**: Whether to show sensitive content in the report. Defaults to ``false``.
        * **show_rows**: The number of rows that shows in the report.


.. _config_detector:

Configure Detector
------------------

A detector consists of ``name`` and ``action``. The **name** parameter defines the detector that will be used, and the **action** parameter defines the way to process the entity.

Configure Name
~~~~~~~~~~~~~~

We currently support the following type of detectors:

* default
* spacy

Default
^^^^^^^

Here scrubadub's pre-defined detector is used. You can designate the name in the format of ``default.<entity>`` (e.g., ``default.phone``). Check the supported detectors from `scrubadub <https://scrubadub.readthedocs.io/en/stable/api_scrubadub_detectors.html>`_.

.. note::

    If you want to de-identify `address` by this tool, `scrubadub_address` is required.
    You will need to follow the `instructions`_ to install the required dependencies.

    .. _instructions: https://scrubadub.readthedocs.io/en/stable/addresses.html/


spaCy
^^^^^

To use spaCy’s NER to identify entity, you can designate the name in the format of ``spacy.<model>.<entity>`` (e.g., ``spacy.en_core_web_sm.person``).
The "entity" value can correspond to any entity that spaCy recognizes. For a list of available models and entities, please refer to the `spaCy documentation <https://spacy.io/models/en>`_.



Configure Action
~~~~~~~~~~~~~~~~

We currently support the following types of actions:

* mask
* remove
* anonymize

Mask
^^^^

The ``mask`` action is used to mask the detected entity with the name of the entity type. It replaces the entity with a placeholder. For example, with the following configured detector:

.. code-block:: yaml

    name: spacy.en_core_web_sm.person
    action: mask

After processing, the input text "Hi, my name is John Doe." will become "Hi, my name is {{NAME}}."

Remove
^^^^^^

The ``remove`` action is used to delete the detected entity from the text. It completely removes the entity without replacement. For example, with the following configured detector:

.. code-block:: yaml

    name: spacy.en_core_web_sm.person
    action: remove

After processing, the input text "Hi, my name is John Doe." will become "Hi, my name is ."


Anonymize
^^^^^^^^^

The ``anonymize`` action can be used to obfuscate the detected sensitive information.
Currently, we provide context-aware anonymization for name, email, and number-like entities.
For example, with the following configured detector:

.. code-block:: yaml

    name: spacy.en_core_web_sm.person
    action: anonymize

After processing, the input text "Hi, my name is John Doe." will become "Hi, my name is Joe Blow."
