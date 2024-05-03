#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import sys
import traceback
from dataclasses import is_dataclass

import fire

from ads.common import logger

try:
    import click

    import ads.jobs.cli
    import ads.opctl.cli
    import ads.opctl.operator.cli
    import ads.pipeline.cli
except Exception as ex:
    print(
        "Please run `pip install oracle-ads[opctl]` to install "
        "the required dependencies for ADS CLI. \n"
        f"{str(ex)}"
    )
    logger.debug(ex)
    logger.debug(traceback.format_exc())
    exit()

# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-package-version
if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata


ADS_VERSION = metadata.version("oracle_ads")


@click.group()
@click.version_option(version=ADS_VERSION, prog_name="ads")
@click.help_option("--help", "-h")
def click_cli():
    pass


@click.command
def aqua_cli():
    """CLI for AQUA."""
    # This is a dummy entry for click.
    # The `ads aqua` commands are handled by AquaCommand


click_cli.add_command(ads.opctl.cli.commands)
click_cli.add_command(ads.jobs.cli.commands)
click_cli.add_command(ads.pipeline.cli.commands)
click_cli.add_command(ads.opctl.operator.cli.commands)
click_cli.add_command(aqua_cli, name="aqua")


# fix for fire issue with --help
# https://github.com/google/python-fire/issues/258
def _SeparateFlagArgs(args):
    try:
        index = args.index("--help")
        args = args[:index]
        return args, ["--help"]
    except ValueError:
        return args, []


fire.core.parser.SeparateFlagArgs = _SeparateFlagArgs


def serialize(data):
    """Serialize dataclass objects or lists of dataclass objects.
    Parameters:
        data: A dataclass object or a list of dataclass objects.
    Returns:
        None
    Prints:
        The string representation of each dataclass object.
    """
    if isinstance(data, list):
        [print(str(item)) for item in data]
    else:
        print(str(data))


def exit_program(ex: Exception, logger: "logging.Logger") -> None:
    """
    Logs the exception and exits the program with a specific exit code.

    This function logs the full traceback and the exception message, then terminates
    the program with an exit code. If the exception object has an 'exit_code' attribute,
    it uses that as the exit code; otherwise, it defaults to 1.

    Parameters
    ----------
    ex (Exception):
        The exception that triggered the program exit. This exception
        should ideally contain an 'exit_code' attribute, but it is not mandatory.
    logger (Logger):
        A logging.Logger instance used to log the traceback and the error message.

    Returns
    -------
    None:
        This function does not return anything because it calls sys.exit,
        terminating the process.

    Examples
    --------

    >>> import logging
    >>> logger = logging.getLogger('ExampleLogger')
    >>> try:
    ...     raise ValueError("An error occurred")
    ... except Exception as e:
    ...     exit_program(e, logger)
    """

    logger.debug(traceback.format_exc())
    logger.error(str(ex))

    exit_code = getattr(ex, "exit_code", 1)
    logger.error(f"Exit code: {exit_code}")
    sys.exit(exit_code)


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == "aqua":
        from ads.aqua import logger as aqua_logger
        from ads.aqua.cli import AquaCommand

        try:
            fire.Fire(
                AquaCommand, command=sys.argv[2:], name="ads aqua", serialize=serialize
            )
        except Exception as err:
            exit_program(err, aqua_logger)
    else:
        click_cli()


if __name__ == "__main__":
    cli()
