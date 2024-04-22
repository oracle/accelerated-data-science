#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.feature_store.common.enums import EntityType

try:
    import graphviz
    from graphviz import Digraph
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `graphviz` module was not found. Please run `pip install "
        f"{OptionalDependency.GRAPHVIZ}`."
    )

logger = logging.getLogger(__name__)
GRAPH_BOX_COLOR = "#DEDEDE"
CONSTRUCT_COLOR_MAP = {
    "FEATURE_STORE": "#747E7E",
    "ENTITY": "#F26B1D",
    "TRANSFORMATION": "#3E975D",
    "FEATURE_GROUP": "#9146C2",
    "DATASET": "#2C6CBF",
}

STEP_WITH_STATUS_LABEL_TEMPLATE = """<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD><FONT COLOR="black" POINT-SIZE="11.0" FACE="Helvetica,Arial,sans-serif">{step_name}</FONT></TD></TR>
        <TR BORDER="1"><TD><FONT COLOR="black" POINT-SIZE="7.0" FACE="Helvetica,Arial,sans-serif">{step_kind}</FONT></TD></TR>
        <TR><TD><FONT COLOR="black" POINT-SIZE="7.0"  FACE="Courier New">{step_id}</FONT></TD></TR>
        </TABLE>>"""


class GraphOrientation:
    TOP_BOTTOM = "TB"
    LEFT_RIGHT = "LR"


class GraphService:
    """GraphService is used to plot the lineage tree graph using GraphViz,
    Please refer to this documentation for more details around graphviz : https://graphviz.org/about/
    """

    @staticmethod
    def __graphviz_node(label, graph_root: Digraph, id, color):
        graph_root.node(
            name=id,
            label=label,
            shape=None,
            style="filled, rounded",
            fontsize="14.0",
            color=color,
            # fillcolor=CONSTRUCT_COLOR_MAP[construct_type],
        )

    @staticmethod
    def __add_node_entry(lineage_element, graph_root: Digraph, construct_type):
        if lineage_element:
            label = STEP_WITH_STATUS_LABEL_TEMPLATE.format(
                step_name=lineage_element.display_name,
                step_id=lineage_element.id,
                step_kind=construct_type,
            )
            GraphService.__graphviz_node(
                label, graph_root, lineage_element.id, GRAPH_BOX_COLOR
            )

    @staticmethod
    def __add_edge_entry(
        lineage_source, lineage_destination, graph_root, visited_edges
    ):
        if (
            lineage_source
            and lineage_destination
            and f"{lineage_source.id}.{lineage_destination.id}" not in visited_edges
        ):
            graph_root.edge(lineage_source.id, lineage_destination.id)
            visited_edges[f"{lineage_source.id}.{lineage_destination.id}"] = True

    @staticmethod
    def __add_model_nodes(model, graph_root: Digraph, construct_type):
        if model and model.items:
            for model_item in model.items:
                label = STEP_WITH_STATUS_LABEL_TEMPLATE.format(
                    step_name=" ",
                    step_id=model_item,
                    step_kind=construct_type,
                )
                GraphService.__graphviz_node(
                    label, graph_root, model_item, GRAPH_BOX_COLOR
                )

    @staticmethod
    def __add_model_edges(
        lineage_source, model_items, graph_root: Digraph, visited_edges
    ):
        if lineage_source and model_items and model_items.items:
            for model_item in model_items.items:
                if f"{lineage_source.id}.{model_item}" not in visited_edges:
                    graph_root.edge(lineage_source.id, model_item)
                    visited_edges[f"{lineage_source.id}.{model_item}"] = True

    @staticmethod
    def view_lineage(
        lineage_data,
        lineage_type,
        rankdir: str = GraphOrientation.LEFT_RIGHT,
    ):
        lineage_composite = lineage_data.lineage
        visited_edges = {}
        if lineage_type == EntityType.FEATURE_GROUP:
            graph_root = graphviz.Digraph(
                graph_attr={"rankdir": rankdir},
            )
        else:
            graph_root = graphviz.Digraph(graph_attr={"rankdir": rankdir})
        graph_root.attr("node", shape="box")
        for lineage_item in lineage_composite.items:
            GraphService.__add_node_entry(
                lineage_item.feature_store, graph_root, "Feature Store"
            )
            GraphService.__add_node_entry(lineage_item.entity, graph_root, "Entity")
            GraphService.__add_edge_entry(
                lineage_item.feature_store,
                lineage_item.entity,
                graph_root,
                visited_edges,
            )
            GraphService.__add_node_entry(
                lineage_item.transformation, graph_root, "Transformation"
            )
            GraphService.__add_edge_entry(
                lineage_item.feature_store,
                lineage_item.transformation,
                graph_root,
                visited_edges,
            )
            GraphService.__add_node_entry(
                lineage_item.feature_group, graph_root, "Feature Group"
            )
            GraphService.__add_edge_entry(
                lineage_item.entity,
                lineage_item.feature_group,
                graph_root,
                visited_edges,
            )
            GraphService.__add_edge_entry(
                lineage_item.transformation,
                lineage_item.feature_group,
                graph_root,
                visited_edges,
            )
            GraphService.__add_node_entry(lineage_item.dataset, graph_root, "Dataset")
            GraphService.__add_edge_entry(
                lineage_item.entity, lineage_item.dataset, graph_root, visited_edges
            )
            GraphService.__add_edge_entry(
                lineage_item.feature_group,
                lineage_item.dataset,
                graph_root,
                visited_edges,
            )
            GraphService.__add_model_nodes(
                lineage_item.model_details, graph_root, "Model"
            )
            GraphService.__add_model_edges(
                lineage_item.dataset,
                lineage_item.model_details,
                graph_root,
                visited_edges,
            )
        try:
            from IPython.core.display import display

            display(graph_root)
        except:
            pass
