============
PII Operator
============

The PII operator aims to detect and redact Personally Identifiable Information(PII) in datasets by combining pattern match and machine learning solution.

Overview
--------

**Introduction to PII**

Personal Identifiable Information (PII) refers to any information that can identify an individual, encompassing financial, medical, educational, and employment records. Failure to protect Personal Identifiable Information (PII) can lead to identity theft, financial loss, and reputational damage of individuals and businesses alike, highlighting the importance of taking appropriate measures to safeguard sensitive information. The Operators framework is OCI's most extensible, low-code, managed ecosystem for detecting and redacting pii in dataset.

This technical documentation introduces using ``ads opctl`` for detecting and redacting pii tasks. This module is engineered with the principles of low-code development in mind, making it accessible to users with varying degrees of technical expertise. It operates on managed infrastructure, ensuring reliability and scalability, while its configurability through YAML allows users to customize redaction to their specific needs.

**Automated Detection and Classification**

By leveraging pattern matching and AI-powered solution, the ADS PII Operator efficiently identifies sentitive data on free form texts.

**Intelligent Co-reference Resolution**

A standout feature of the ADS PII Operator is its ability to maintain co-reference entity relationships even after anonymization, this not only anonymizes the data, but peserves the statistical properties of the data.

**PII Operator Documentation**

This documentation will explore the key concepts and capabilities of the PII operator, providing examples and practical guidance on how to use its various functions and modules. By the end of this guide, users will have a solid understanding of the PII operator and its capabilities, as well as the knowledge and tools needed to make informed decisions when designing solutions tailored to their specific requirements.

.. versionadded:: 2.9.0

.. toctree::
  :maxdepth: 1

  ./install
  ./getting_started
  ./pii
  ./examples
  ./yaml_schema
