Overview
********

Oracle Cloud Infrastructure (OCI) Data Science Machine Learning (ML) Pipelines lets you define and run an end-to-end machine learning orchestration covering all the steps of machine learning lifecycle that can be executed in a repeatable, continuous ML pipeline.

The machine learning lifecycle is composed of several steps: data acquisition and extraction, data preparation, featurization, model training including algorithm selection and hyper-parameter tuning, model evaluation, deployment, and finally monitoring the deployed model and possible retraining.


Pipeline Step
=============

A pipeline step is a task in a pipeline. A pipeline step can be either a Data Science Job step or a Custom Script step.

Pipeline
========

A pipeline is a workflow of tasks, called steps. Steps can run in sequence or in parallel resulting in a Directed Acyclic Graph (DAG) of the steps.

In a machine learning context, ML Pipelines provide a workflow of data import, data transformation, model training, and model evaluation.

Pipeline Run
============

Pipeline Run is the execution instance of a pipeline. Each pipeline run includes its step runs.




