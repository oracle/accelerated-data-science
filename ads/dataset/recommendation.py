#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import importlib

from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.dataset import logger


class Recommendation:
    recommendation_types = [
        "constant_column",
        "primary_key",
        "imputation",
        "strong_correlation",
        "positive_class",
        "fix_imbalance",
    ]
    recommendation_type_labels = [
        "Constant Columns",
        "Potential Primary Key Columns",
        "Imputation",
        "Multicollinear Columns",
        "Identify positive label for target",
        "Fix imbalance in dataset",
    ]

    @runtime_dependency(module="ipywidgets", install_from=OptionalDependency.NOTEBOOK)
    def __init__(self, ds, recommendation_transformer):
        self.recommendation_transformer = recommendation_transformer
        self.reco_dict = recommendation_transformer.reco_dict_
        self.orig_reco_dict = copy.deepcopy(self.reco_dict)
        self.ds = ds
        self._clear_output()
        self.fill_nan_dict_values = {}
        self.fill_nan_dict = {}
        self.recommendation_type_index = 0
        self.out = ipywidgets.widgets.Output()

    def _clear_output(self):
        self.boxes = []
        self.column_list = []
        self.message_list = []
        self.action_list = {}
        self.control_buttons = None
        self.info = []

    def _get_on_action_change(self, recommendation_type, column):
        def on_action_change(change):
            for message in self.info:
                message.close()
            if change["type"] == "change" and change["name"] == "value":
                if change["new"] == "Fill missing values with constant":
                    self._show_constant_fill_widget(column)
                else:
                    if change["old"] == "Fill missing values with constant":
                        text = self.fill_nan_dict.pop(column, None)
                        if text is not None:
                            text.close()
                self.reco_dict[recommendation_type][column]["Selected Action"] = change[
                    "new"
                ]

        return on_action_change

    @runtime_dependency(module="ipywidgets", install_from=OptionalDependency.NOTEBOOK)
    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def _show_constant_fill_widget(self, column):
        if column in self.ds.numeric_columns:
            text = ipywidgets.widgets.FloatText(
                description=column,
                disabled=False,
                style={"description_width": "initial"},
                layout=ipywidgets.Layout(width="auto"),
            )
        else:
            text = ipywidgets.widgets.Text(
                description=column,
                disabled=False,
                style={"description_width": "initial"},
                layout=ipywidgets.Layout(width="auto"),
            )
        # copy previously entered value
        if column in self.fill_nan_dict:
            text.value = self.fill_nan_dict[column].value
        self.fill_nan_dict[column] = text

        from IPython.core.display import display

        display(text)
        if self.control_buttons is not None:
            # self.control_buttons.close()
            self.control_buttons = ipywidgets.HBox(self.control_buttons.children)
            display(self.control_buttons)

    def _get_apply_on_click(self):
        def apply(change):
            self.fill_nan_dict_values = {
                k: v.value for k, v in self.fill_nan_dict.items()
            }
            if self.recommendation_type_index < len(self.recommendation_types):
                self._display()
            else:
                self.recommendation_transformer.reco_dict = self.reco_dict
                self.recommendation_transformer.fill_nan_dict_ = (
                    self.fill_nan_dict_values
                )
                self.ds.recommendation_transformer = self.recommendation_transformer
                with self.out:
                    self.ds.new_ds = self.ds.auto_transform()

        return apply

    def _get_reset_on_click(self):
        def reset(change):
            recommendation_type = self.recommendation_types[
                self.recommendation_type_index
            ]
            self.fill_nan_dict_values = None
            self.reco_dict = copy.deepcopy(self.orig_reco_dict)
            for index, column in enumerate(self.reco_dict[recommendation_type].keys()):
                self.action_list[recommendation_type][index].value = self.reco_dict[
                    recommendation_type
                ][column]["Selected Action"]
            for constant_text in self.fill_nan_dict.values():
                constant_text.close()
            for message in self.info:
                message.close()
            self.ds.new_ds = None

        return reset

    def _repr_html_(self):
        self.show_in_notebook()
        return ""

    def show_in_notebook(self):
        if len(self.reco_dict) == 0:
            logger.info("No recommendations.")
            return

        self._display()

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    @runtime_dependency(module="ipywidgets", install_from=OptionalDependency.NOTEBOOK)
    def _display(self):

        from IPython.core.display import display

        if self.recommendation_type_index != len(self.recommendation_types):
            if (
                self.recommendation_types[self.recommendation_type_index]
                in self.reco_dict
            ):
                recommendation = self.reco_dict[
                    self.recommendation_types[self.recommendation_type_index]
                ]
                if self.recommendation_type_index == 0:
                    for column in recommendation:
                        print(
                            "Column '{0}' is constant and will be dropped".format(
                                column
                            )
                        )
                    self.recommendation_type_index += 1
                    self._display()
                    return
                else:
                    with self.out:
                        display(
                            ipywidgets.HTML(
                                """
                    <font size=4>
                        {}
                    </font>
                    </body""".format(
                                    self.recommendation_type_labels[
                                        self.recommendation_type_index
                                    ]
                                )
                            ),
                            layout=ipywidgets.Layout(display="flex"),
                        )
                        self.action_list[
                            self.recommendation_types[self.recommendation_type_index]
                        ] = []
                        self.column_list = []
                        self.message_list = []
                        self.extra_info_list = []
                        for column in recommendation.keys():
                            messages = ipywidgets.Label(
                                recommendation[column]["Message"],
                                layout=ipywidgets.Layout(
                                    flex="auto", justify_content="flex-start"
                                ),
                            )
                            actions = ipywidgets.widgets.Dropdown(
                                options=recommendation[column]["Action"],
                                value=recommendation[column]["Selected Action"],
                                width="auto",
                            )
                            if (
                                "Up-sample" in recommendation[column]["Action"]
                                and importlib.util.find_spec("imblearn") is None
                            ):
                                extra_info = ipywidgets.HTML(
                                    """Up-sample requires imbalanced-learn python package. <br>
                                            To install run <code> pip install imbalanced-learn </code>.
                                        """,
                                    layout=ipywidgets.Layout(
                                        flex="auto", flex_flow="wrap"
                                    ),
                                )
                                self.extra_info_list.append(extra_info)
                            else:
                                self.extra_info_list.append(
                                    ipywidgets.Label(
                                        "",
                                        layout=ipywidgets.Layout(
                                            flex="auto", justify_content="flex-start"
                                        ),
                                    )
                                )

                            actions.observe(
                                self._get_on_action_change(
                                    self.recommendation_types[
                                        self.recommendation_type_index
                                    ],
                                    column,
                                )
                            )
                            self.column_list.append(
                                ipywidgets.Label(
                                    column
                                    + "(type: {0})".format(
                                        self.ds.sampled_df[column].dtype
                                    ),
                                    layout=ipywidgets.Layout(flex="auto"),
                                    color="grey",
                                )
                            )
                            self.message_list.append(messages)
                            self.action_list[
                                self.recommendation_types[
                                    self.recommendation_type_index
                                ]
                            ].append(actions)
                        display(
                            ipywidgets.HBox(
                                [
                                    ipywidgets.VBox(self.column_list),
                                    ipywidgets.VBox(self.message_list),
                                    ipywidgets.VBox(
                                        self.action_list[
                                            self.recommendation_types[
                                                self.recommendation_type_index
                                            ]
                                        ]
                                    ),
                                    ipywidgets.VBox(self.extra_info_list),
                                ],
                                layout=ipywidgets.Layout(
                                    display="flex",
                                    flex_flow="row",
                                    border="solid 2px",
                                    align_content="stretch",
                                    width="auto",
                                ),
                            )
                        )
            else:
                self.recommendation_type_index += 1
                self._display()
                return
        if self.control_buttons is not None:
            self.control_buttons.close()
            self.control_buttons = None
        else:
            display(self.out)
        self.recommendation_type_index += 1
        for constant_text in self.fill_nan_dict.values():
            constant_text.close()
        for constant_text in self.fill_nan_dict.values():
            self._show_constant_fill_widget(constant_text.description)
        if self.recommendation_type_index < len(self.recommendation_types):
            apply_button = ipywidgets.widgets.Button(
                description="Next", button_style="primary"
            )
        else:
            apply_button = ipywidgets.widgets.Button(
                description="Apply", button_style="primary"
            )
        apply_button.on_click(self._get_apply_on_click())
        reset_button = ipywidgets.widgets.Button(description="Reset All")
        reset_button.on_click(self._get_reset_on_click())
        self.control_buttons = ipywidgets.HBox([apply_button, reset_button])
        with self.out:
            display(self.control_buttons)
