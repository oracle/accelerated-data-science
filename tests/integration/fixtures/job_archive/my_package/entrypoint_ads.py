#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import os
import sys
print(os.getcwd())
print(sys.path)

import ads
print(ads.__version__)

from my_package import utils
from my_module import my_function_in_module

def write_output():
    output_dir = os.environ.get("OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(
        output_dir,
        "python_test.txt"
    )
    print(f"Writing {os.path.abspath(filename)}...")
    with open(filename, "w") as f:
        f.write(datetime.datetime.now().strftime("%Y%m%d_%H%M%s.txt"))

print("This is the entrypoint inside a package.")

my_function_in_module()
utils.my_function_in_package()
write_output()
