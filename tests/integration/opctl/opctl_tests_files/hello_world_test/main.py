#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import argparse
import os
import json
from test_mod import test_import
from folder.test_mod import test_import_nested
import ads

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=False, default="ADS")
    args = parser.parse_args()
    print(f"ADS {ads.__version__}")
    print("Running user script...")
    print(f"Hello World from {args.name}")
    print("Printing environment variables...")
    print(json.dumps(dict(os.environ), indent=2))
    test_import()
    test_import_nested()
