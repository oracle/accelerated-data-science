Overview
========

Machine learning and deep learning are becoming ubiquitous due to:

* The ability to solve complex problems in a variety of different domains.
* The growth in the performance and efficiency of modern computing resources.
* The widespread availability of large amounts of data.

However, as the size and complexity of problems continue to increase, so does the complexity of the machine learning algorithms applied to these problems. The inherent and growing complexity of machine learning algorithms limits the ability to understand what the model has learned or why a given prediction was made, acting as a barrier to the adoption of machine learning. Additionally, there may be legal or regulatory requirements to be able to explain the outcome of a prediction from a machine learning model, resulting in the use of biased models at the cost of accuracy.

Machine learning explainability (MLX) is the process of explaining and interpreting machine learning and deep learning models.

MLX can help machine learning developers to:

* Better understand and interpret the model's behavior.

  - Which features does the model consider important? 
  - What is the relationship between the feature values and the target predictions?

* Debug and improve the quality of the model.

  - Did the model learn something unexpected? 
  - Does the model generalize or did it learn something specific to the training dataset?

* Increase trust in the model and confidence in deploying the model.

MLX can help users of machine learning algorithms to:

* Understand why the model made a certain prediction.

  - Why was my bank loan denied?

Some useful terms for MLX:

* **Explainability**: The ability to explain the reasons behind a machine learning model’s prediction.
* **Global Explanations**: Understand the general behavior of a machine learning model as a whole.
* **Interpretability**: The level at which a human can understand the explanation.
* **Local Explanations**: Understand why the machine learning model made a specific prediction.
* **Model-Agnostic Explanations**: Explanations treat the machine learning model and feature pre-processing as a black box, instead of using properties from the model to guide the explanation.
* **WhatIf Explanations**: Understand how changes in the value of features affects the model's prediction.

The ADS explanation module provides interpretable, model-agnostic, local and global explanations.

