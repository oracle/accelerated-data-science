#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest

from ads.tests import logger
from ads.tests import utils as testutils
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


# ref: https://nbconvert.readthedocs.io/en/latest/api/preprocessors.html
# ref: https://stackoverflow.com/a/3772008
class TestNotebooks(unittest.TestCase):
    def __init__(self, notebook_filename, kernel=None):
        super(TestNotebooks, self).__init__()
        self.notebook_filename = notebook_filename

        # We create a new logger for each notebook so that messages can be
        # appropriately tagged with the notebook they arise from.
        self.kernel = kernel

    @runtime_dependency(module="nbformat", install_from=OptionalDependency.OPCTL)
    @runtime_dependency(module="nbconvert", install_from=OptionalDependency.OPCTL)
    def runTest(self):
        from nbformat import v4 as nbf

        error_flag = False
        msg = None
        with open(self.notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
            nb.cells.insert(
                0, nbf.new_code_cell("import ads; ads.set_debug_mode(True)")
            )
            logger.info("Executing " + self.notebook_filename)

            from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

            if self.kernel:
                ep = ExecutePreprocessor(timeout=600, kernel_name=self.kernel)
            else:
                ep = ExecutePreprocessor(timeout=600)

            try:
                ep.log = logger
                ep.preprocess(
                    nb, {"metadata": {"path": os.path.dirname(self.notebook_filename)}}
                )
            except CellExecutionError as e:
                msg = "Error {0} in executing notebook {1}".format(
                    str(e), self.notebook_filename
                )
                error_flag = True
        self.assertFalse(error_flag, msg=msg)
        return


# We will dynamically create a test case for each notebook in the stable
# notebooks folder. All these test cases will be held in a test suite.
# Hence, whenever we add a new stable notebook, a test case for it will
# be automatically created in this test suite.
def get_suite(kernel=None):
    stable_notebook_folder = testutils.get_stable_notebooks_folder()
    notebooks = (
        os.path.join(stable_notebook_folder, nb)
        for nb in os.listdir(stable_notebook_folder)
        if nb.endswith(".ipynb")
    )
    notebook_suite = unittest.TestSuite()

    for nb in notebooks:
        notebook_suite.addTest(TestNotebooks(nb, kernel=kernel))

    return notebook_suite


# We need custom test-discovery protocol for this module. Without this, the
# default test discovery mechanism will use the testcase classes defined
# in this module instead of using the test suite. When this method is
# present, invocation of unittests using the "python -m unittest
# <TestModuleName>" convention will work as expected.
# ref: https://docs.python.org/3/library/unittest.html#load-tests-protocol
def load_tests(loader, tests, pattern):
    return get_suite()


if __name__ == "__main__":
    unittest.TextTestRunner().run(get_suite())
