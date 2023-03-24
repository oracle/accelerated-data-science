#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import patch

import pytest
from ads.common.decorator.runtime_dependency import runtime_dependency
from tests.unitary.default_setup.runtime_dependency import mock_package
from tests.unitary.default_setup.runtime_dependency.mock_package import (
    mock_module,
    mock_package_function,
)
from tests.unitary.default_setup.runtime_dependency.mock_package.mock_module import (
    mock_module_function,
)


class TestRuntimeDependencyDecorator:

    MOCK_PACKAGE_PATH = "tests.unitary.default_setup.runtime_dependency.mock_package"

    @pytest.mark.parametrize(
        "module, short_name, object, expected_result",
        [
            (MOCK_PACKAGE_PATH, "mp", None, mock_package),
            (MOCK_PACKAGE_PATH, None, None, mock_package),
            (f"{MOCK_PACKAGE_PATH}.mock_module", "mm", None, mock_module),
            (
                f"{MOCK_PACKAGE_PATH}.mock_module",
                "mf",
                "mock_module_function",
                mock_module_function,
            ),
            (MOCK_PACKAGE_PATH, "tpf", "mock_package_function", mock_package_function),
        ],
    )
    def test_positive(self, module, short_name, object, expected_result):
        """Ensures modules can be imported via decorator and injected to the original function."""

        @runtime_dependency(module=module, short_name=short_name, object=object)
        def mock_function():
            _module_name = module.split(".")[-1]
            return globals()[short_name or object or _module_name]

        assert mock_function() == expected_result

    @pytest.mark.parametrize(
        "module, object, expected_result",
        [
            (f"{MOCK_PACKAGE_PATH}_fake", None, "module was not found"),
            (f"{MOCK_PACKAGE_PATH}.mock_module_fake", None, "module was not found"),
            (
                f"{MOCK_PACKAGE_PATH}.mock_module",
                "mock_function_fake",
                "Cannot import name",
            ),
            (MOCK_PACKAGE_PATH, "mock_package_function_fake", "Cannot import name"),
        ],
    )
    def test_negative(self, module, object, expected_result):
        """Ensures the decorator throws error in case if module not installed."""

        @runtime_dependency(module=module, object=object)
        def mock_function():
            pass

        with pytest.raises((ModuleNotFoundError, ImportError), match=expected_result):
            mock_function()

    def test_imported_module_can_be_used(self):
        """Ensures the imported modules can be used in the original function."""

        @runtime_dependency(module=self.MOCK_PACKAGE_PATH, short_name="mp")
        def mock_function1():
            return mp.mock_package_function()

        @runtime_dependency(
            module=f"{self.MOCK_PACKAGE_PATH}.mock_module", short_name="mm"
        )
        def mock_function2():
            return mm.mock_module_function()

        @runtime_dependency(
            module=f"{self.MOCK_PACKAGE_PATH}.mock_module",
            object="mock_module_function",
        )
        def mock_function3():
            return mock_module_function()

        @runtime_dependency(
            module=f"{self.MOCK_PACKAGE_PATH}", object="mock_package_function"
        )
        def mock_function4():
            return mock_package_function()

        assert mock_function1() == mock_package_function()
        assert mock_function4() == mock_package_function()
        assert mock_function2() == mock_module_function()
        assert mock_function3() == mock_module_function()

    def test_multi_import(self):
        """Ensures the decorator supports multiple imports."""

        @runtime_dependency(
            module=f"{self.MOCK_PACKAGE_PATH}.mock_module", short_name="mm"
        )
        @runtime_dependency(module=self.MOCK_PACKAGE_PATH, short_name="mp")
        def mock_function():
            return mp.mock_package_function(), mm.mock_module_function()

        assert mock_function() == (mock_package_function(), mock_module_function())

    def test_custom_error_message(self):
        """Ensures that custom error can be specified."""

        @runtime_dependency(
            module=f"{self.MOCK_PACKAGE_PATH}_fake", err_msg="custom error msg"
        )
        def mock_function():
            pass

        with pytest.raises(ModuleNotFoundError, match="custom error msg"):
            mock_function()

    def test_install_from_message(self):
        """Ensures that `install_from` parameter can be specified and used in error message."""

        @runtime_dependency(
            module=f"{self.MOCK_PACKAGE_PATH}_fake", install_from="oracle-ads[data]"
        )
        def mock_function():
            pass

        with pytest.raises(
            ModuleNotFoundError, match=r".*pip install oracle-ads[data]*"
        ):
            mock_function()

    @pytest.mark.parametrize(
        "module, expected_error",
        [
            ("", "The parameter `module` must be provided."),
            ({"test"}, "The parameter `module` must be a string."),
        ],
    )
    def test_module_not_provided(self, module, expected_error):
        """Ensures that the decorator throws an error in case of module name not provided."""

        @runtime_dependency(module=module)
        def mock_function():
            pass

        with pytest.raises(AssertionError, match=expected_error):
            mock_function()

    def test_is_for_notebook_only(self):
        """Tests the `is_for_notebook_only` option."""

        @runtime_dependency(module="test_module", is_for_notebook_only=True)
        @runtime_dependency(
            module=self.MOCK_PACKAGE_PATH, short_name="tp", is_for_notebook_only=False
        )
        def mock_function():
            return globals()["tp"]

        with patch("ads.common.utils.is_notebook") as mock_is_notebook:
            mock_is_notebook.return_value = True
            with pytest.raises(ModuleNotFoundError, match=r".*test_module*"):
                mock_function()

            mock_is_notebook.return_value = False
            mock_function() == mock_package_function
