#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import time
import sys


print("Starting the job")
argv = sys.argv[1:]

args = []
kwargs = {}

i = 0
while i < len(argv):
    if argv[i].startswith("-"):
        kwargs[argv[i]] = argv[i + 1]
        i += 2
    else:
        args.append(argv[i])
        i += 1

print(f"Positional Arguments: {args}")
print(f"Keyword Arguments: {kwargs}")

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

print("Sleeping for 60 seconds...")
time.sleep(60)
print("After 60 seconds...")

print("Finishing the job")
