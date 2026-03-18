"""
IPython magics for running Slurm commands via SSH.

Load with:
  %load_ext ads.slurm.magics
"""

from .magics import load_ipython_extension, unload_ipython_extension

__all__ = ["load_ipython_extension", "unload_ipython_extension"]
