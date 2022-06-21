.. _string-regex_match:

RegEx Match
***********

Text documents are often parsed looking for specific patterns to extract information like emails, dates, times, web links, and so on. This pattern matching is often done using RegEx, which is hard to write, modify, and understand. Custom written RegEx often misses the edge cases. ``ADSString`` provides a number of common RegEx patterns so that your work is simplified. You can use the following patterns:

* ``credit_card``: Credit card number.
* ``dates``: Dates in a variety of standard formats.
* ``email``: Email address.
* ``ip``: IP addresses, versions IPV4 and IPV6.
* ``link``: Text that appears to be a link to a website.
* ``phone_number_US``: USA phone numbers including those with extensions.
* ``price``: Text that appears to be a price.
* ``ssn``: USA social security number.
* ``street_address``: Street address.
* ``time``: Text that appears to be a time and less than 24 hours.
* ``zip_code``: USA zip code.

The preceding ``ADSString`` properties return an array with each pattern that in matches. The following examples demonstrate how to extract email addresses, dates ,and links from the text. Note that the text is extracted as is. For example, the dates aren't converted to a standard format. The returned value is the text as it is represented in the input text. Use the ``datetime.strptime()`` method to convert the date to a date time stamp.

.. code-block:: python3

    s = ADSString("Get in touch with my associates john.smith@example.com and jane.johnson@example.com to schedule")
    s.email

.. parsed-literal::

    ['john.smith@example.com', 'jane.johnson@example.com']

.. code-block:: python3

    s = ADSString("She is born on Jan. 19th, 2014 and died 2021-09-10")
    s.date

.. parsed-literal::

    ['Jan. 19th, 2014', '2021-09-10']

.. code-block:: python3

    s = ADSString("Follow the link www.oracle.com to Oracle's homepage.")
    s.link

.. parsed-literal::

    ['www.oracle.com']


