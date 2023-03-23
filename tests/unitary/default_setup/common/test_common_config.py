#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from collections import defaultdict
from unittest.mock import MagicMock, mock_open, patch

import pytest
from ads.common import auth
from ads.common.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONFIG_PROFILE,
    Config,
    ConfigSection,
    Eventing,
    EventType,
    ExtendedConfigParser,
)


class TestEventing:
    """Tests for the class helper to register event handlers."""

    def setup_method(self):
        self.test_eventing = Eventing()

    def test__init__(self):
        """Ensures that Eventing instance can be initialized."""
        expected_result = defaultdict(set)
        assert hasattr(self.test_eventing, "_events")
        assert self.test_eventing._events == expected_result

    def test_on(self):
        """Ensures that new events can be registered."""
        mock_callback = MagicMock()
        self.test_eventing.on("new_event", callback=mock_callback)
        self.test_eventing.on("new_event", callback=mock_callback)
        assert len(self.test_eventing._events) == 1
        assert "new_event" in self.test_eventing._events

    def test_trigger(self):
        """Ensures that registered callbacks can be triggered."""
        mock_callback = MagicMock()
        self.test_eventing.on("new_event", callback=mock_callback)
        self.test_eventing.trigger("new_event")
        mock_callback.assert_called_once()


class TestConfigSection:
    """Tests the class representing a config section."""

    def setup_method(self):
        self.test_config_section = ConfigSection()

    def test_init(self):
        """Ensures the instance of ConfigSection can be initialized."""
        assert hasattr(self.test_config_section, "events")
        assert hasattr(self.test_config_section, "_info")
        assert isinstance(self.test_config_section.events, Eventing)
        assert self.test_config_section._info == {}

    @patch.object(Eventing, "trigger")
    def test_clear(self, mock_eventing_trigger):
        """Test clearing the config section values."""
        self.test_config_section._info = {"test": "value"}
        self.test_config_section.clear()
        mock_eventing_trigger.assert_called_with(EventType.CHANGE.value)
        assert self.test_config_section._info == {}

    def test_copy(self):
        """Tests making a copy of a config section."""
        test_info = {"key": "value"}
        self.test_config_section._info = test_info
        result = self.test_config_section.copy()
        assert isinstance(result, ConfigSection)
        assert result._info == test_info

    def test__with_dict(self):
        """Tests populating config section info from a dictionary."""

        with pytest.raises(TypeError, match="The `info` must be a dictionary."):
            self.test_config_section._with_dict("111")

        self.test_config_section._info = {"k1": "v1", "k2": "v2", "k3": "v3"}
        common_keys = ["k1"]
        with pytest.raises(ValueError) as ex:
            self.test_config_section._with_dict({"k1": "v11"})
        assert str(ex.value) == (
            "The config section is already contain "
            f"fields: {common_keys}. Use `replace=True` to overwrite."
        )

        self.test_config_section._with_dict({"k1": "v11", "k3": "v31"}, replace=True)
        assert self.test_config_section._info == {"k1": "v11", "k2": "v2", "k3": "v31"}

    @patch.object(Eventing, "trigger")
    @patch.object(ConfigSection, "_with_dict")
    def test_with_dict(self, mock_congig_section_with_dict, mock_eventing_trigger):
        """Tests populating config section info from a dictionary."""
        test_dict = {"key": "value"}
        self.test_config_section.with_dict(test_dict, replace=True)
        mock_congig_section_with_dict.assert_called_with(info=test_dict, replace=True)
        mock_eventing_trigger.assert_called_with(EventType.CHANGE.value)

    def test_keys(self):
        """Tests getting the list of the keys of a config section."""
        self.test_config_section._info = {"k1": "v11", "k3": "v31"}
        assert self.test_config_section.keys() == ("k1", "k3")

    def test_to_dict(self):
        """Tests converting config section to a dictionary."""
        self.test_config_section._info = {"k1": "v1", "k3": "v3"}
        assert self.test_config_section.to_dict() == {"k1": "v1", "k3": "v3"}

    def test_get(self):
        """Tests getting the config section value by key."""
        self.test_config_section._info = {"k1": "v1", "k3": "v3"}
        self.test_config_section.get("k1") == "v1"
        self.test_config_section.get("k0") == None

    def test__set(self):
        """Tests setting the config section value by key."""
        self.test_config_section._info = {"k1": "v1", "k3": "v3"}

        with pytest.raises(ValueError) as ex:
            self.test_config_section.set("k1", "v11")

        assert str(ex.value) == (
            "The field with key `k1` already exists. "
            "Use `replace=True` to overwrite."
        )

        self.test_config_section.set("k1", "v11", replace=True)
        self.test_config_section.set("k2", "v2")

        assert self.test_config_section._info["k1"] == "v11"
        assert self.test_config_section._info["k2"] == "v2"

    @patch.object(Eventing, "trigger")
    @patch.object(ConfigSection, "_set")
    def test_set(self, mock_config_section_set, mock_eventing_trigger):
        """Tests setting the config section value by key."""
        self.test_config_section.set("k1", "v1", replace=True)
        mock_config_section_set.assert_called_with("k1", "v1", True)
        mock_eventing_trigger.assert_called_with(EventType.CHANGE.value)

    @patch.object(Eventing, "trigger")
    def test_remove(self, mock_eventing_trigger):
        """Tests removing the config section field by key."""
        self.test_config_section._info = {"k1": "v1", "k3": "v3"}
        self.test_config_section.remove("k1")
        self.test_config_section.remove("k1111")
        assert self.test_config_section._info == {"k3": "v3"}
        mock_eventing_trigger.assert_called_with(EventType.CHANGE.value)

    def test__getitem__(self):
        self.test_config_section._info = {"k1": "v1", "k3": "v3"}
        assert self.test_config_section["k1"] == "v1"
        assert self.test_config_section["k111"] == None

    def test__setitem__(self):
        self.test_config_section["k1"] = "v1"
        self.test_config_section["k2"] = {"1": "2"}
        assert self.test_config_section["k1"] == "v1"
        assert self.test_config_section["k2"] == {"1": "2"}


class TestConfig:
    """Tests the class representing a config."""

    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))

    def setup_method(self):
        with patch.object(auth, "default_signer"):
            self.test_config = Config()

    @patch("ads.common.auth.default_signer")
    def test__init___with_default_params(self, mock_default_signer):
        """Ensures that Config instance can be initialized with default params."""
        mock_default_signer.return_value = {"config": "value"}
        test_config = Config()

        assert test_config.auth == {"config": "value"}
        assert DEFAULT_CONFIG_PROFILE in test_config._config
        assert len(test_config._config) == 1
        assert test_config.uri == os.path.expanduser(DEFAULT_CONFIG_PATH)
        assert isinstance(test_config._config_parser, ExtendedConfigParser)

    def test__init__(self):
        """Ensures that Config instance can be initialized."""
        test_config = Config(uri="/custom/config/location/", auth={"test": "test"})
        assert test_config.auth == {"test": "test"}
        assert DEFAULT_CONFIG_PROFILE in test_config._config
        assert len(test_config._config) == 1
        assert test_config.uri == "/custom/config/location/"
        assert isinstance(test_config._config_parser, ExtendedConfigParser)

    @patch.object(Config, "section_get")
    def test_default(self, mock_config_section_get):
        """Tests getting default config section."""
        self.test_config.default()
        mock_config_section_get.assert_called_with(DEFAULT_CONFIG_PROFILE)

    def tests_section_exists(self):
        """Tests checkoing if a config section exists."""
        assert self.test_config.section_exists(DEFAULT_CONFIG_PROFILE) == True

    def test_section_get(self):
        """Tests getting the config section by key."""
        fake_key = "fake_key"
        with pytest.raises(
            KeyError, match=f"The config section `{fake_key.upper()}` not found."
        ):
            self.test_config.section_get(key=fake_key)
        assert isinstance(
            self.test_config.section_get(DEFAULT_CONFIG_PROFILE), ConfigSection
        )

    def test_section_set(self):
        """Tests setting a config section to config."""

        # ensures adding new config section fails if section with
        # given name is already exists and replace flag is False
        with pytest.raises(ValueError) as exc:
            self.test_config.section_set(DEFAULT_CONFIG_PROFILE, {"key", "value"})
        assert str(exc.value) == (
            f"A config section `{DEFAULT_CONFIG_PROFILE}` is already exist. "
            "Use `replace=True` if you want to overwrite."
        )

        # ensures adding new config section fails in case of wrong input attributes
        with pytest.raises(TypeError):
            self.test_config.section_set("new_profile", "wrong_data")

        # ensures the new config section can be added
        self.test_config.section_set("from_dict", {"key": "value"})
        self.test_config.section_set(
            "from_config_section", ConfigSection().with_dict({"key1": "value1"})
        )

        assert len(self.test_config._config.keys()) == 3

        assert isinstance(self.test_config._config["from_dict".upper()], ConfigSection)
        assert self.test_config._config["from_dict".upper()].to_dict() == {
            "key": "value"
        }
        assert isinstance(
            self.test_config._config["from_config_section".upper()], ConfigSection
        )
        assert self.test_config._config["from_config_section".upper()].to_dict() == {
            "key1": "value1"
        }

        # test replace
        self.test_config.section_set(
            "from_config_section", {"key2": "value2"}, replace=True
        )
        assert self.test_config._config["from_config_section".upper()].to_dict() == {
            "key1": "value1",
            "key2": "value2",
        }

    @patch.object(Config, "_on_change")
    def test_section_remove(self, mock_config_onchange):
        """Tests removing config section form config."""
        self.test_config.section_set("new_section", {"key": "value"})
        assert self.test_config.section_exists("new_section") == True
        self.test_config.section_remove("new_section")
        assert self.test_config.section_exists("new_section") == False
        mock_config_onchange.assert_called()

    @patch.object(ExtendedConfigParser, "save")
    @patch.object(ExtendedConfigParser, "with_dict")
    def test_save(self, mock_config_parser_with_dict, mock_config_parser_save):
        """Tests saving config data to a config file."""
        mock_config_parser_with_dict.return_value = self.test_config._config_parser
        expected_result = {DEFAULT_CONFIG_PROFILE: {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.section_set("new_section", {"key": "value"})
        self.test_config.save()
        mock_config_parser_with_dict.assert_called_with(expected_result)
        mock_config_parser_save.assert_called_once()

    @patch.object(ExtendedConfigParser, "save")
    @patch.object(ExtendedConfigParser, "with_dict")
    def test_save_uri(self, mock_config_parser_with_dict, mock_config_parser_save):
        """Tests saving config data to a remote config file."""
        mock_config_parser_with_dict.return_value = self.test_config._config_parser
        expected_result = {DEFAULT_CONFIG_PROFILE: {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.section_set("new_section", {"key": "value"})
        self.test_config.save(uri="test_uri", auth={"config": ""}, force_overwrite=True)
        mock_config_parser_with_dict.assert_called_with(expected_result)
        mock_config_parser_save.assert_called_with(
            uri="test_uri", auth={"config": ""}, force_overwrite=True
        )

    @patch.object(ExtendedConfigParser, "with_dict")
    def test_save_uri_fail(self, mock_config_parser_with_dict):
        """Tests saving config data to a remote config file."""
        mock_config_parser_with_dict.return_value = self.test_config._config_parser
        expected_result = {DEFAULT_CONFIG_PROFILE: {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.section_set("new_section", {"key": "value"})

        with pytest.raises(FileExistsError):
            self.test_config.save(
                uri=os.path.join(self.curr_dir, "test_files/config/test_config"),
                auth={"config": "value"},
                force_overwrite=False,
            )
            mock_config_parser_with_dict.assert_called_with(expected_result)

    @patch.object(ExtendedConfigParser, "read")
    @patch.object(ExtendedConfigParser, "to_dict")
    def test_load(self, mock_config_parser_to_dict, mock_config_parser_read):
        """Tests loading config data from a config file."""
        expected_result = {"NEW_SECTION": {"key": "value"}}
        mock_config_parser_read.return_value = self.test_config._config_parser
        mock_config_parser_to_dict.return_value = expected_result
        self.test_config.load()
        mock_config_parser_read.assert_called_once()
        mock_config_parser_to_dict.assert_called_once()
        self.test_config._config == expected_result

    @patch.object(ExtendedConfigParser, "read")
    @patch.object(ExtendedConfigParser, "to_dict")
    def test_load_uri(self, mock_config_parser_to_dict, mock_config_parser_read):
        """Tests loading config data from a config file."""
        expected_result = {"NEW_SECTION": {"key": "value"}}
        mock_config_parser_read.return_value = self.test_config._config_parser
        mock_config_parser_to_dict.return_value = expected_result
        self.test_config.load(uri="test_uri", auth={"config": ""})
        mock_config_parser_read.assert_called_with(uri="test_uri", auth={"config": ""})
        mock_config_parser_to_dict.assert_called_once()
        self.test_config._config == expected_result

    @patch.object(Config, "validate")
    @patch.object(Config, "_on_change")
    def with_dict(self, mock_config_on_change, mock_config_validate):
        """Tests merging dictionary to config."""
        expected_result = {"DEFAULT": {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.with_dict(expected_result, replace=True)
        mock_config_on_change.assert_called()
        mock_config_validate.assert_called()
        assert self.test_config._config == expected_result

    def test_to_dict(self):
        """Converts config to a dictionary format."""
        expected_result = {"DEFAULT": {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.with_dict(expected_result, replace=True)
        assert self.test_config.to_dict() == expected_result

    def test_keys(self):
        """Test getting the all registered config section keys."""
        test_data = {"DEFAULT": {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.with_dict(test_data, replace=True)
        assert self.test_config.keys() == test_data.keys()

    def test__getitem__(self):
        expected_result = {"DEFAULT": {}, "NEW_SECTION": {"key": "value"}}
        self.test_config.with_dict(expected_result, replace=True)
        assert self.test_config["NEW_SECTION"].to_dict() == {"key": "value"}

    def test__setitem__(self):
        self.test_config["NEW_SECTION"] = {"key": "value"}
        assert self.test_config["NEW_SECTION"].to_dict() == {"key": "value"}

    def test__validate(self):
        """Tests validating input dictionary."""
        with pytest.raises(TypeError, match="The input data should be a dictionary."):
            self.test_config._validate("wrong_value")
        with pytest.raises(
            ValueError,
            match=f"The `WRONG_SECTION` must be a dictionary or a `ConfigSection` instance.",
        ):
            self.test_config._validate({"WRONG_SECTION": "wrong_value", "DEFAULT": {}})


class TestExtendedConfigParser:
    """Tests class helper to read/write information to the config file."""

    def setup_class(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))

    def setup_method(self):
        self.test_config_parser = ExtendedConfigParser(
            uri="test_uri", auth={"config": "value"}
        )

    @patch("ads.common.auth.default_signer")
    def test__init___with_default_params(self, mock_default_signer):
        """Ensures the TestExtendedConfigParser instance can be initialized with default params."""
        mock_default_signer.return_value = {"config": "value"}

        config_parser = ExtendedConfigParser()
        assert config_parser.auth == {"config": "value"}
        assert config_parser.uri == DEFAULT_CONFIG_PATH

    def test__init__(self):
        """Ensures the TestExtendedConfigParser instance can be initialized."""
        config_parser = ExtendedConfigParser(uri="test_uri", auth={"config": "value"})
        assert config_parser.auth == {"config": "value"}
        assert config_parser.uri == "test_uri"

    def test_with_dict(self):
        """Tests populating config with values from a dictionary."""
        test_config = {"MY_PROFILE": {"key": "value"}}
        self.test_config_parser.read_dict(test_config)
        assert self.test_config_parser.to_dict() == test_config

    def test_save(self):
        """Tests saving the config to the file."""
        mock_file_path = "oci://bucket-name@namespace/test_config"
        mock_storage_options = {"config": {"test": "value"}}

        test_config = {"MY_PROFILE": {"key": "value"}}
        test_config_parser = ExtendedConfigParser(
            uri=mock_file_path, auth=mock_storage_options
        )
        test_config_parser.read_dict(test_config)
        open_mock = mock_open()

        with patch("fsspec.open", open_mock, create=True):
            test_config_parser.save(force_overwrite=True)
        open_mock.assert_called_with(mock_file_path, mode="w", **mock_storage_options)
        open_mock.return_value.write.assert_called()

    def test_read(self):
        """Tests reading config file."""
        expected_result = {
            "DEFAULT": {"some_key1": "some_value_1", "some_key2": "some_value_2"},
            "MY_PROFILE": {"some_key1": "some_value_11", "some_key2": "some_value_22"},
        }
        test_config_parser = ExtendedConfigParser(
            uri=os.path.join(self.curr_dir, "test_files/config/test_config"),
            auth={"config": {}},
        )
        test_config_parser.read()
        assert test_config_parser.to_dict() == expected_result
