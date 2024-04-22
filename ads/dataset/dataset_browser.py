#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from __future__ import print_function, absolute_import

import re, pathlib, os
import urllib.parse
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, isdir, join, getsize
from typing import List, Set, Tuple, Dict

import requests

import pandas as pd
import sklearn.datasets as sk_datasets

from ads.dataset import helper
from ads.common.utils import inject_and_copy_kwargs
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class DatasetBrowser(ABC):
    #
    # anything that lists & loads datasets
    #

    def __init__(self):  # pragma: no cover
        pass

    @staticmethod
    def list(filter_pattern="*") -> List[str]:
        """
        Return a list of dataset browser strings.
        """
        return ["web", "sklearn", "seaborn", "GitHub"]

    @abstractmethod
    def open(name: str, **kwargs):  # pragma: no cover
        """
        Return new dataset for the given name.

        Parameters
        ----------
        name : str
            the name of the dataset to open.

        Returns
        -------
        ds: Dataset

        Examples
        --------
        ds_browser = DatasetBrowser("sklearn")

        ds = ds_browser.open("iris")
        """
        pass

    #
    # helper to filter list of dataset names
    #
    def filter_list(self, L, filter_pattern) -> List[str]:  # pragma: no cover
        """
        Filters a list of dataset names.
        """
        return list(filter(re.compile(filter_pattern).match, L))

    #
    #
    # Create subclasses for each type of DatasetBrowser
    #
    #

    @staticmethod
    def web(index_url: str):
        """
        Returns a WebDataset.
        """
        return WebDatasets(index_url)  # pragma: no cover

    @staticmethod
    def sklearn():
        """
        Returns a SklearnDataset.
        """
        return SklearnDatasets()

    @staticmethod
    def seaborn():
        """
        Returns a SeabornDataset.
        """
        return SeabornDatasets()

    @staticmethod
    def filesystem(folder: str):
        """
        Returns a LocalFilesystemDataset.
        """
        return LocalFilesystemDatasets(folder)

    @staticmethod
    def GitHub(user: str, repo: str, branch: str = "master"):
        """
        Returns a GitHubDataset
        """
        return GitHubDatasets(user, repo, branch)


class GitHubDatasets(DatasetBrowser):
    def __init__(self, user: str, repo: str, branch: str):
        super(DatasetBrowser, self).__init__()
        self.user = user
        self.repo = repo
        self.branch = branch
        self.api_url = (
            f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"
        )

    def _generate_filelist(self):

        response = requests.get(self.api_url)
        if response.status_code == 200:
            listing = []
            j = response.json()
            if "tree" in j:
                for n in j["tree"]:
                    filepath = n["path"]
                    fileurl = f"https://raw.githubusercontent.com/{self.user}/{self.repo}/{self.branch}/{filepath}"
                    pl = pathlib.Path(filepath)
                    format = pl.suffix[1:]
                    if format.lower() in ["csv", "tsv", "json"]:
                        listing.append(
                            {
                                "url": fileurl,
                                "name": os.path.splitext(filepath)[0],
                                "format": format,
                                "size": n["size"],
                                "description": f"""
                                                Origin: GitHub {self.user}@{self.repo}#{self.branch}\n
                                                Name: {filepath}, size: {n['size']/1024}kb\n"
                                                """.strip(),
                            }
                        )

            return listing

        else:
            raise ValueError(
                f"Error accessing GitHub API ({self.api_url}): {response.status_code}"
            )

    def list(self, filter_pattern: str = ".*") -> List[str]:
        return super().filter_list(
            [x["name"] for x in self._generate_filelist()], filter_pattern
        )

    def open(self, name: str, **kwargs):
        #
        # lookup the name
        #

        for obj in self._generate_filelist():
            if obj["name"] == name:
                return helper.open(
                    **inject_and_copy_kwargs(
                        kwargs,
                        **{
                            "source": obj["url"],
                            "format": obj["format"],
                            "name": obj["name"],
                            "description": obj["description"],
                        },
                    )
                )

        raise ValueError(f"dataset [{name}] does not exist, use .list() to display all")


class LocalFilesystemDatasets(DatasetBrowser):
    def __init__(self, folder: str):
        super(DatasetBrowser, self).__init__()
        self.folder = folder

    def list(self, filter_pattern: str = ".*") -> List[str]:
        return super().filter_list(
            [x["name"] for x in self._generate_filelist()], filter_pattern
        )

    def open(self, name: str, **kwargs):
        #
        # lookup the name
        #

        for obj in self._generate_filelist():
            if obj["name"] == name:
                return helper.open(
                    **inject_and_copy_kwargs(
                        kwargs,
                        **{
                            "source": obj["path"],
                            "format": obj["format"],
                            "name": obj["name"],
                            "description": obj["description"],
                        },
                    )
                )

        raise ValueError(f"dataset [{name}] does not exist, use .list() to display all")

    def _generate_filelist(self):
        if not isdir(self.folder):
            raise ValueError(
                f'The path "{self.folder}" does not exist, or is not a folder'
            )
        else:
            onlyfiles = [
                f for f in listdir(self.folder) if isfile(join(self.folder, f))
            ]
            listing = []
            for f in onlyfiles:
                i = f.rfind(".")
                if i > 0:
                    name, format = f[0:i].strip(), f[i + 1 :].strip()
                    path = join(self.folder, f)
                    if format.lower() in ["json", "csv", "tsv", "hdf"]:
                        listing.append(
                            {
                                "path": path,
                                "size": getsize(path),
                                "name": f"{name}.{format}",
                                "format": format.strip(),
                                "description": f"""
                                                Origin: {self.folder}\n
                                                Name: {f} ({getsize(path)} bytes)\n"
                                                """.strip(),
                            }
                        )

            return listing


class WebDatasets(DatasetBrowser):
    @runtime_dependency(module="htmllistparse", install_from=OptionalDependency.DATA)
    def __init__(self, index_url: str):  # pragma: no cover

        self.index_url = index_url
        self.listing = []
        super(DatasetBrowser, self).__init__()

        try:
            _, raw_listing = htmllistparse.fetch_listing(self.index_url, timeout=30)
        except Exception as e:
            raise ValueError(str(e))

        for f in [x for x in raw_listing if x.size]:
            i = f.name.rfind(".")
            filename, format = f.name[0:i].strip(), f.name[i + 1 :]
            if format.lower() in ["json", "csv", "tsv", "hdf"]:
                d = {
                    "url": urllib.parse.urljoin(self.index_url, f.name),
                    "size": f.size,
                    "name": filename,
                    "format": format,
                    "description": f"""
                                        Origin: {self.index_url}\n
                                        Name: {f.name.strip()} ({f.size} bytes)\n"
                                        """.strip(),
                }
                self.listing.append(d)

    def list(self, filter_pattern: str = ".*") -> List[str]:
        return super().filter_list([x["name"] for x in self.listing], filter_pattern)

    def open(self, name: str, **kwargs):
        #
        # lookup the name
        #
        for obj in self.listing:
            if obj["name"] == name:
                return helper.open(
                    obj["url"],
                    format=obj["format"],
                    name=obj["name"],
                    description=obj["description"],
                )
        raise ValueError(f"dataset [{name}] does not exist, use .list() to display all")


class SeabornDatasets(DatasetBrowser):
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def __init__(self):
        super(DatasetBrowser, self).__init__()
        self.dataset_names = list(seaborn.get_dataset_names())

    def list(self, filter_pattern: str = ".*") -> List[str]:
        return super().filter_list(self.dataset_names, filter_pattern)

    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def open(self, name: str, **kwargs):
        if name in self.dataset_names:
            return helper.open(
                seaborn.load_dataset(name), name=name, description="from seaborn"
            )
        else:
            raise ValueError(
                "dataset [{name}] does not exist, use .list() to display all"
            )


class SklearnDatasets(DatasetBrowser):

    sklearn_datasets = ["breast_cancer", "diabetes", "iris", "wine", "digits"]

    def __init__(self):
        super(DatasetBrowser, self).__init__()

    def list(self, filter_pattern: str = ".*") -> List[str]:
        return super().filter_list(SklearnDatasets.sklearn_datasets, filter_pattern)

    def open(self, name: str, **kwargs):
        if name in SklearnDatasets.sklearn_datasets:
            data = getattr(sk_datasets, "load_%s" % (name))()
            description = data["DESCR"]
            if "images" in data:
                # special case digits
                n_samples = len(data.images)
                cols = data.images.reshape((n_samples, -1))
                df = pd.DataFrame(
                    cols, columns=["f%d" % (i) for i in range(cols.shape[1])]
                )
                for col in df.columns:
                    df[col] = df[col].astype(float)
                df["target"] = pd.Series(data.target).astype("category")
            elif "target_names" in data:
                # inverse transform the target labels for categorical types
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = pd.Series(
                    [data.target_names[x] for x in data.target]
                ).astype("category")
            else:
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = pd.Series(data.target)

            return helper.open(
                df, target="target", name=name, description=description
            )

        else:
            raise ValueError(
                f"dataset [{name}] does not exist, use .list() to display all"
            )
