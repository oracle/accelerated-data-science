.. _string-overview:

Overview
********

Text analytics uses a set of powerful tools to understand the content of unstructured data, such as text. It’s becoming an increasingly more important tool in feature engineering as product reviews, media content, research papers, and more are being mined for their content. In many data science areas, such as marketing analytics, the use of unstructured text is becoming as popular as structured data. This is largely due to the relatively low cost of collection of the data. However, the downside is the complexity of working with the data. To work with unstructured that you need to clean, summarize, and create features from it before you create a model. The ``ADSString`` class provides tools that allow you to quickly do this work. More importantly, you can expand the tool to meet your specific needs.

Data scientists need to be able to quickly and easily manipulate strings. ADS SDK provides an enhanced string class, called ``ADSString``. It adds functionality like regular expression (RegEx) matching and natural language processing (NLP) parsing. The class can be expanded by registering custom plugins so that you can process a string in a way that it fits your specific needs. For example, you can register the `OCI Language service <https://docs.oracle.com/iaas/language/using/overview.htm>`__ plugin to bind functionalities from the `OCI Language service <https://docs.oracle.com/iaas/language/using/overview.htm>`__ to ``ADSString``.

