.. _string-still_a_string:

Still a String
**************

While ``ADSString`` expands your feature engineering capabilities, it can still be treated as a ``str`` object. Any standard operation on ``str`` is preserved in ``ADSString``. For instance, you can convert it to lowercase:

.. code-block:: python3

    hello_world = "HELLO WORLD"
    s = ADSString(hello_world)
    s.lower()

.. parsed-literal::

    'hello world'

You could split a text string.

.. code-block:: python3

    s.split()

.. parsed-literal::

    ['HELLO', 'WORLD']

You can use all the ``str`` methods, such as the ``.replace()`` method, to replace text.

.. code-block:: python3

    s.replace("L", "N")

.. parsed-literal::

    'HENNO WORND'

You can perform a number of ``str`` manipulation operations, such as ``.lower()`` and ``.upper()`` to get an ``ADSString`` object back.

.. code-block:: python3

    isinstance(s.lower().upper(), ADSString)

.. parsed-literal::

    True

While a new ``ADSString`` object is created with ``str`` manipulation operations, the equality operation holds.

.. code-block:: python3

    s.lower().upper() == s

.. parsed-literal::

    True

The equality operation even holds between ``ADSString`` objects (``s``) and ``str`` objects (``hello_world``).

.. code-block:: python3

    s == hello_world

.. parsed-literal::

    True

