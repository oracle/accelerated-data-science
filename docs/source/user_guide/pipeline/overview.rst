Overview
********

Oracle Cloud Infrastructure (OCI) Data Science Machine Learning (ML) Pipeline enables you to define and run an end-to-end machine learning orchestration covering all the steps of machine learning lifecycle that can be executed in a repeatable, continuous ML pipeline. 

Machine learning lifecycle is composed of data acquisition and extraction, data preparation, featurization, training model including algorithm selection and hyper-parameter tuning, model evaluation, deployment, and then monitoring the deployed model and retraining.


Pipeline Step
=============

Pipeline step is a task in a pipeline. A pipeline step can be either a Data Science Job step or a Custom Script step.

Pipeline
========

A pipeline is a workflow of tasks, called steps. Steps can run in sequence or in parallel, creating a Directed Acyclic Graph (DAG) of the steps.

In a machine learning context, ML Pipelines provide a workflow of data import, data transformation, model training, and model evaluation.

Pipeline Run
============

Pipeline Run is the execution instance of a pipeline. Each pipeline run includes its step runs.




