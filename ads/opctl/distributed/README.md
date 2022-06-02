In order to run your script on distributed jobs:


Getting Started:

0. Make sure you have ads opctl set up:

1. Generate resources required for the distributed dask cluster:
```
ads opctl distributed init --framework dask --version v1
ads opctl distributed init --framework horovod --type tensorflow --arch cpu --version v1
```

2.
This should give you an output something like:
```
docker build -t iad.ocir.io/ociodscdev/my-distr-example:latest -f my-distr-example/Dockerfile .
ads opctl publish-image iad.ocir.io/ociodscdev/my-distr-example:latest
ads opctl run -f my-distr-example/dask_cluster.yaml
```
Before running the above, be sure to:
- Put all of your code into `main.py`.
- If you need more files, be sure to include them in the Dockerfile.
- If you need more libraries, be sure to include them in the environment.yaml.
- Anything you can factor out of main.py as an arg, should be put in the dask_cluster.yaml file.
- Go through dask_cluster.yaml and fill in TODO's with valid parameters
- You shouldn't need to touch any other file, but of course they're there if you need to.


3. Then in separate terminals, watch the statuses:
```
ads opctl watch <main job run id>
ads opctl watch <worker x job run id>
```

4. Next Steps:
If you run into any errors that require changing the image, make those changes and go back to step 2.
The best development approach is to make the script as configurable as possible through args, kwargs and env vars. This will allow for rapid development form the yaml, rather than re-building the image.

## Yaml Example -

```
kind: distributed
apiVersion: v1.0
spec:
  infrastructure: # This section maps to Job definition. Does not include environment variables
    kind: infrastructure
    type: dataScienceJob
    apiVersion: v1.0
    spec:
      projectId: pjocid
      compartmentId: ctocid
      displayName: my_distributed_training
      logGroupId: lgrpid
      logId: logid
      subnetId: subID
      shapeName: VM.Standard2.1
      blockStorageSizeGB: 50GB
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: default
      workDir: "Working directory for the cluster"
      ephemeral: True
      name: cluster name
      config:
        env:
          - name: death_timeout
            value: 10
          - name: death_timeout
            value: 4
        startOptions: # Only named args. Will construct OCI___CLUSTER_START_OPTIONS environment variable from list of args as " ".join(startOptions)
      main:
        image: optional
        name: main-name
        replicas: 1
        config:
          env:
            - name: death_timeout
              value: 20
            - name: death_timeout
              value: 5
          startOptions: # Only named args. Will construct OCI___CLUSTER_START_OPTIONS environment variable from list of args as " ".join(startOptions)
      worker:
        name: worker-name
        image: optional
        replicas: 2 #Name is not decided
        config:
          env:
            - name: death_timeout
              value: 30
            - name: death_timeout
              value: 8
          startOptions: # Only named args. Will construct OCI___CLUSTER_START_OPTIONS environment variable from list of args as " ".join(startOptions)
            - --worker-port 8700:8800
            - --nanny-port 3000:3100
  runtime:
    kind: python
    apiVersion: v1.0
    spec:
      entryPoint: "printhello.py"
      args:
        - 500
      kwargs:
      env:
        - name: TEST
          value: "test"
```
