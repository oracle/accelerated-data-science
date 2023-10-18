====================
Forecasting Operator
====================

The Forecasting Operator leverages historical time series data to generate accurate forecasts for future trends. This operator aims to simplify and expedite the data science process by automating the selection of appropriate models and hyperparameters, as well as identifying relevant features for a given prediction task.


Overview
--------

**Introduction to Forecasting with the Python Library Module**

Forecasting is a crucial component of decision-making in various fields, from finance and supply chain management to weather prediction and demand forecasting. Accurate forecasts enable organizations to allocate resources efficiently, plan for the future, and respond proactively to changing circumstances. The Operators framework is OCI's most extensible, low-code, managed ecosystem for building and deploying forecasting models.

This technical documentation introduces using ``ads opctl`` for forecasting tasks. This module is engineered with the principles of low-code development in mind, making it accessible to users with varying degrees of technical expertise. It operates on managed infrastructure, ensuring reliability and scalability, while its configurability through YAML allows users to tailor forecasts to their specific needs.

**Multivariate vs. Univariate Forecasting**

One of the fundamental decisions in forecasting is whether to employ multivariate or univariate models. Univariate forecasting involves predicting a single variable, typically based on its historical values, making it suitable for straightforward time series analysis. In contrast, multivariate forecasting takes into account multiple interrelated variables, allowing for a more comprehensive understanding of complex systems.

**Global vs. Local Models for Multivariate Forecasts**

When dealing with multivariate forecasts, the choice between global and local models is pivotal. Global models assume that the relationships between variables are uniform across all data points, providing a consolidated forecast for the entire dataset. In contrast, local models consider localized relationships, allowing forecasts to adapt to variations within the dataset.

**Strengths and Weaknesses of Global and Local Models**

Global models are advantageous when relationships between variables remain relatively stable over time. They offer simplicity and ease of interpretation, making them suitable for a wide range of applications. However, they may struggle to capture nuances in the data when relationships are not consistent throughout the dataset.

Local models, on the other hand, excel in capturing localized patterns and relationships, making them well-suited for datasets with varying dynamics. They can provide more accurate forecasts in cases where global models fall short.

**Auto Model Selection**

Some users know which modeling frameworks (this can be a specific model, such as ARIMA and Prophet or it can be an automl library like Oracle's AutoMLX) they want to use right already, the forecasting operator allows these more advanced users to configure this through the ``model`` parameter. For those newer users who don't know, or want to explore multiple, the forecasting operator sets the ``model`` parameter to  "auto" by default. "auto" will select the framework that looks most appropriate given the dataset.

**Forecasting Documentation**

This documentation will explore these concepts in greater depth, demonstrating how to leverage the flexibility and configurability of the Python library module to implement both multivariate and univariate forecasting models, as well as global and local approaches. By the end of this guide, users will have the knowledge and tools needed to make informed decisions when designing forecasting solutions tailored to their specific requirements.


.. toctree::
  :hidden:
  :maxdepth: 1

  ./getting_started
  ./yaml_schema
  ./examples
  ./advanced_use_cases
  ./faq
