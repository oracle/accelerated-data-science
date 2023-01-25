Examples
********

Create a pipeline
=================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    pipeline_step = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
      )

    pipeline.create()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step])
      )

    pipeline.create()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()


Run a job as a step
===================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.jobs import Job, DataScienceJob, ScriptRuntime
    from ads.pipeline import PipelineStep, Pipeline
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = DataScienceJob(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    job = Job(
        infrastructure=infrastructure, 
        runtime=runtime
    )
    job.create() # create a job

    pipeline_step = PipelineStep(
        name="Job_Step",
        description="A step running a job",
        job_id=job.id
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
      )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.jobs import Job, DataScienceJob, ScriptRuntime
    from ads.pipeline import Pipeline, PipelineStep
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        DataScienceJob()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    job = (
        Job()
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )
    job.create() # create a job

    pipeline_step = (
        PipelineStep("Job_Step")
        .with_description("A step running a job")
        .with_job_id(job.id)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step])
      )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    rom ads.jobs import Job, DataScienceJob, ScriptRuntime
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        DataScienceJob()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    job = (
        Job()
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )
    job.create() # create a job

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: dataScienceJob
        spec:
          description: A step running a job
          jobId: {job_id}
          name: Job_Step
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id, job_id=job.id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()


Run a python script as a step
=============================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    pipeline_step = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
      )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step])
      )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()

    
Run a notebook as a step
========================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, NotebookRuntime
    import os 

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = NotebookRuntime(
        notebook_path_uri="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
        output_uri="oci://<bucket_name>@<namespace>/<prefix>",
        env={"GREETINGS": "Welcome to OCI Data Science"}
    )

    pipeline_step = PipelineStep(
        name="Notebook_Step",
        description="A step running a notebook",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
    )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, NotebookRuntime
    import os

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        NotebookRuntime()
        .with_notebook(
            path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
            encoding='utf-8'
        )
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_output("oci://<bucket_name>@<namespace>/<prefix>")
    )

    pipeline_step = (
        PipelineStep("Notebook_Step")
        .with_description("A step running a notebook")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
        Pipeline("A single step pipeline")
        .with_compartment_id(compartment_id)
        .with_project_id(project_id)
        .with_step_details([pipeline_step])
    )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a notebook
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Notebook_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              env:
              - name: GREETINGS
                value: Welcome to OCI Data Science
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
              outputURI: oci://<bucket_name>@<namespace>/<prefix>
            type: notebook
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()



Run two steps with the same infrastructure
==========================================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    step_one_runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "generalml_p37_cpu_v1"}
    )

    pipeline_step_one = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=step_one_runtime
    )

    step_two_runtime = NotebookRuntime(
        notebook_path_uri="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
        output_uri="oci://<bucket_name>@<namespace>/<prefix>",
        env={"GREETINGS": "Welcome to OCI Data Science"}
    )

    pipeline_step_two = PipelineStep(
        name="Notebook_Step",
        description="A step running a notebook",
        infrastructure=infrastructure,
        runtime=step_two_runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step_one, pipeline_step_two],
      )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    step_one_runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step_one = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(step_one_runtime)
    )

    step_two_runtime = (
        NotebookRuntime()
        .with_notebook(
            path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
            encoding='utf-8'
        )
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_output("oci://<bucket_name>@<namespace>/<prefix>")
    )

    pipeline_step_two = (
        PipelineStep("Notebook_Step")
        .with_description("A step running a notebook")
        .with_infrastructure(infrastructure)
        .with_runtime(step_two_runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step_one, pipeline_step_two])
      )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
      - kind: customScript
        spec:
          description: A step running a notebook
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Notebook_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              env:
              - name: GREETINGS
                value: Welcome to OCI Data Science
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
              outputURI: oci://<bucket_name>@<namespace>/<prefix>
            type: notebook
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()




Run two steps in parallel
=========================

In the example below, when DAG is not specified, the steps in the pipeline run in parallel.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    step_one_runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "generalml_p37_cpu_v1"}
    )

    pipeline_step_one = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=step_one_runtime
    )

    step_two_runtime = NotebookRuntime(
        notebook_path_uri="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
        output_uri="oci://<bucket_name>@<namespace>/<prefix>",
        env={"GREETINGS": "Welcome to OCI Data Science"}
    )

    pipeline_step_two = PipelineStep(
        name="Notebook_Step",
        description="A step running a notebook",
        infrastructure=infrastructure,
        runtime=step_two_runtime
    )
    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step_one, pipeline_step_two],
      )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    step_one_runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step_one = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(step_one_runtime)
    )

    step_two_runtime = (
        NotebookRuntime()
        .with_notebook(
            path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
            encoding='utf-8'
        )
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_output("oci://<bucket_name>@<namespace>/<prefix>")
    )

    pipeline_step_two = (
        PipelineStep("Notebook_Step")
        .with_description("A step running a notebook")
        .with_infrastructure(infrastructure)
        .with_runtime(step_two_runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step_one, pipeline_step_two])
      )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
      - kind: customScript
        spec:
          description: A step running a notebook
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Notebook_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              env:
              - name: GREETINGS
                value: Welcome to OCI Data Science
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
              outputURI: oci://<bucket_name>@<namespace>/<prefix>
            type: notebook
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()



Run two steps sequentially
==========================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    step_one_runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "generalml_p37_cpu_v1"}
    )

    pipeline_step_one = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=step_one_runtime
    )

    step_two_runtime = NotebookRuntime(
        notebook_path_uri="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
        output_uri="oci://<bucket_name>@<namespace>/<prefix>",
        env={"GREETINGS": "Welcome to OCI Data Science"}
    )

    pipeline_step_two = PipelineStep(
        name="Notebook_Step",
        description="A step running a notebook",
        infrastructure=infrastructure,
        runtime=step_two_runtime
    )
    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step_one, pipeline_step_two],
        dag=["Python_Script_Step >> Notebook_Step"],
      )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    step_one_runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step_one = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(step_one_runtime)
    )

    step_two_runtime = (
        NotebookRuntime()
        .with_notebook(
            path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
            encoding='utf-8'
        )
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_output("oci://<bucket_name>@<namespace>/<prefix>")
    )

    pipeline_step_two = (
        PipelineStep("Notebook_Step")
        .with_description("A step running a notebook")
        .with_infrastructure(infrastructure)
        .with_runtime(step_two_runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step_one, pipeline_step_two])
          .with_dag(["Python_Script_Step >> Notebook_Step"])
      )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      dag:
      - Python_Script_Step >> Notebook_Step
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
      - kind: customScript
        spec:
          description: A step running a notebook
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Notebook_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              env:
              - name: GREETINGS
                value: Welcome to OCI Data Science
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
              outputURI: oci://<bucket_name>@<namespace>/<prefix>
            type: notebook
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()


Run multiple steps with dependencies specified in DAG
=====================================================

In this example, ``step_1`` and ``step_2`` run in parallel and ``step_3`` runs after ``step_1`` and ``step_2`` are complete. 

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    script_runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    notebook_runtime = NotebookRuntime(
        notebook_path_uri="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    pipeline_step_1 = PipelineStep(
        name="step_1",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=script_runtime
    )

    pipeline_step_2 = PipelineStep(
        name="step_2",
        description="A step running a notebook",
        infrastructure=infrastructure,
        runtime=notebook_runtime
    )

    pipeline_step_3 = PipelineStep(
        name="step_3",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=script_runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="An example pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step_1, pipeline_step_2, pipeline_step_3],
        dag=["(step_1, step_2) >> step_3"],
      )

    pipeline.create()      # create the pipeline
    pipeline.show()        # visualize the pipeline

    pipeline_run = pipeline.run()   # run the pipeline

    pipeline_run.show(wait=True)    # watch the pipeline run status



  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime, NotebookRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    script_runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    notebook_runtime = (
        NotebookRuntime()
        .with_notebook(
            path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
            encoding='utf-8'
        )
        .with_service_conda("tensorflow26_p37_cpu_v2")
    )

    pipeline_step_1 = (
        PipelineStep("step_1")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(script_runtime)
    )

    pipeline_step_2 = (
        PipelineStep("step_2")
        .with_description("A step running a notebook")
        .with_infrastructure(infrastructure)
        .with_runtime(notebook_runtime)
    )

    pipeline_step_3 = (
        PipelineStep("step_3")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(script_runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("An example pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step_1, pipeline_step_2, pipeline_step_3])
          .with_dag(["(step_1, step_2) >> step_3"])
      )

    pipeline.create()      # create the pipeline
    pipeline.show()        # visualize the pipeline

    pipeline_run = pipeline.run()   # run the pipeline

    pipeline_run.show(wait=True)    # watch the pipeline run status


  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: An example pipeline
      projectId: {project_id}
      dag:
      - (step_1, step_2) >> step_3
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: step_1
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
      - kind: customScript
        spec:
          description: A step running a notebook
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: step_2
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
            type: notebook
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: step_3
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()      # create the pipeline
    pipeline.show()        # visualize the pipeline

    pipeline_run = pipeline.run()   # run the pipeline

    pipeline_run.show(wait=True)    # watch the pipeline run status


Set environment variables in a step
===================================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, NotebookRuntime
    import os 

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = NotebookRuntime(
        notebook_path_uri="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"},
        output_uri="oci://<bucket_name>@<namespace>/<prefix>",
        env={"GREETINGS": "Welcome to OCI Data Science"}
    )

    pipeline_step = PipelineStep(
        name="Notebook_Step",
        description="A step running a notebook",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
    )

    pipeline.create()

    pipeline_run = pipeline.run()


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, NotebookRuntime
    import os

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        NotebookRuntime()
        .with_notebook(
            path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
            encoding='utf-8'
        )
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_output("oci://<bucket_name>@<namespace>/<prefix>")
    )

    pipeline_step = (
        PipelineStep("Notebook_Step")
        .with_description("A step running a notebook")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
        Pipeline("A single step pipeline")
        .with_compartment_id(compartment_id)
        .with_project_id(project_id)
        .with_step_details([pipeline_step])
    )

    pipeline.create()

    pipeline_run = pipeline.run()

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a notebook
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Notebook_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: tensorflow26_p37_cpu_v2
                type: service
              env:
              - name: GREETINGS
                value: Welcome to OCI Data Science
              notebookEncoding: utf-8
              notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
              outputURI: oci://<bucket_name>@<namespace>/<prefix>
            type: notebook
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    pipeline_run = pipeline.run()



Watch status update on a pipeline run
=====================================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    pipeline_step = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
      )

    pipeline.create()
    pipeline_run = pipeline.run()

    # pipeline_run.show(mode="text")   # watch pipeline run status in text
    pipeline_run.show(wait=True)   # watch pipeline run status in graph


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step])
      )

    pipeline.create()
    pipeline_run = pipeline.run()
    
    # pipeline_run.show(mode="text")   # watch pipeline run status in text
    pipeline_run.show(wait=True)       # watch pipeline run status in graph


  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()
    pipeline_run = pipeline.run()
    
    # pipeline_run.show(mode="text")   # watch pipeline run status in text
    pipeline_run.show(wait=True)       # watch pipeline run status in graph




Monitor logs of a pipeline run
==============================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
      )

    pipeline.create()
    pipeline_run = pipeline.run()

    # pipeline_run.watch()  # stream the consolidated log of the pipeline run
    pipeline_run.watch(log_type="service_log")      # stream service log of the pipeline run
    pipeline_run.watch("Python_Script_Step", log_type="custom_log") # stream custom log of the step run


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step])
      )

    pipeline.create()
    pipeline_run = pipeline.run()

    # pipeline_run.watch()  # stream the consolidated log of the pipeline run
    pipeline_run.watch(log_type="service_log")      # stream service log of the pipeline run
    pipeline_run.watch("Python_Script_Step", log_type="custom_log") # stream custom log of the step run

  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()
    pipeline_run = pipeline.run()

    # pipeline_run.watch()  # stream the consolidated log of the pipeline run
    pipeline_run.watch(log_type="service_log")      # stream service log of the pipeline run
    pipeline_run.watch("Python_Script_Step", log_type="custom_log") # stream custom log of the step run


Override configurations when creating a pipeline run
====================================================

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os 

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = CustomScriptStep(
        block_storage_size=200,
        shape_name="VM.Standard3.Flex",
        shape_config_details={"ocpus": 4, "memory_in_gbs": 32},
    )

    runtime = ScriptRuntime(
        script_path_uri="script.py",
        conda={"type": "service", "slug": "tensorflow26_p37_cpu_v2"}
    )

    pipeline_step = PipelineStep(
        name="Python_Script_Step",
        description="A step running a python script",
        infrastructure=infrastructure,
        runtime=runtime
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = Pipeline(
        name="A single step pipeline",
        compartment_id=compartment_id,
        project_id=project_id,
        step_details=[pipeline_step],
        command_line_arguments="argument --key value",
        environment_variables={"env": "value"},
      )

    pipeline.create()

    # Override configurations when creating a pipeline run
    display_override_name = "RunOverrideName"
    configuration_override_details = {
        "maximum_runtime_in_minutes": 30,
        "type": "DEFAULT",
        "environment_variables": {"a": "b"},
        "command_line_arguments": "ARGUMENT --KEY VALUE",
    }

    step_override_details = [
    {
        "step_name": "Python_Script_Step",
        "step_configuration_details": {
            "maximum_runtime_in_minutes": 200,
            "environment_variables": {"1": "2"},
            "command_line_arguments": "argument --key value",
        },
    }
    ]
    pipeline_run = pipeline.run(
        display_name=display_override_name,
        configuration_override_details=configuration_override_details,
        step_override_details=step_override_details,
    )

    


  .. code-tab:: Python3
    :caption: Python (Alternative)

    from ads.pipeline import Pipeline, PipelineStep, CustomScriptStep, ScriptRuntime
    import os

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    infrastructure = (
        CustomScriptStep()
        .with_block_storage_size(200)
        .with_shape_name("VM.Standard3.Flex")
        .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )

    runtime = (
        ScriptRuntime()
        .with_source("script.py")
        .with_service_conda("generalml_p37_cpu_v1")
    )

    pipeline_step = (
        PipelineStep("Python_Script_Step")
        .with_description("A step running a python script")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    pipeline = (
          Pipeline("A single step pipeline")
          .with_compartment_id(compartment_id)
          .with_project_id(project_id)
          .with_step_details([pipeline_step])
          .with_argument("argument", key="value")
          .with_environment_variable(env="value")
      )

    pipeline.create()

    # Override configurations when creating a pipeline run
    display_override_name = "RunOverrideName"
    configuration_override_details = {
        "maximum_runtime_in_minutes": 30,
        "type": "DEFAULT",
        "environment_variables": {"a": "b"},
        "command_line_arguments": "ARGUMENT --KEY VALUE",
    }

    step_override_details = [
    {
        "step_name": "Python_Script_Step",
        "step_configuration_details": {
            "maximum_runtime_in_minutes": 200,
            "environment_variables": {"1": "2"},
            "command_line_arguments": "argument --key value",
        },
    }
    ]
    pipeline_run = pipeline.run(
        display_name=display_override_name,
        configuration_override_details=configuration_override_details,
        step_override_details=step_override_details,
    )    
    


  .. code-tab:: Python3
    :caption: YAML
    
    from ads.pipeline import Pipeline
    import os

    compartment_id = os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = os.environ["PROJECT_OCID"]

    with open("script.py", "w") as f:
        f.write("print('Hello World!')")

    yaml_string = """
    kind: pipeline
    spec:
      commandLineArguments: argument --key value
      environmentVariables:
        env: value
      compartmentId: {compartment_id}
      displayName: A single step pipeline
      projectId: {project_id}
      stepDetails:
      - kind: customScript
        spec:
          description: A step running a python script
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
          runtime:
            kind: runtime
            spec:
              conda:
                slug: generalml_p37_cpu_v1
                type: service
              scriptPathURI: script.py
            type: script
    type: pipeline
    """.format(compartment_id=compartment_id, project_id=project_id)

    pipeline = Pipeline.from_yaml(yaml_string)

    pipeline.create()

    # Override configurations when creating a pipeline run
    display_override_name = "RunOverrideName"
    configuration_override_details = {
        "maximum_runtime_in_minutes": 30,
        "type": "DEFAULT",
        "environment_variables": {"a": "b"},
        "command_line_arguments": "ARGUMENT --KEY VALUE",
    }

    step_override_details = [
    {
        "step_name": "Python_Script_Step",
        "step_configuration_details": {
            "maximum_runtime_in_minutes": 200,
            "environment_variables": {"1": "2"},
            "command_line_arguments": "argument --key value",
        },
    }
    ]
    pipeline_run = pipeline.run(
        display_name=display_override_name,
        configuration_override_details=configuration_override_details,
        step_override_details=step_override_details,
    )
    
    


