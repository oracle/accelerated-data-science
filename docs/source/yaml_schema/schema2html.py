"""This script generates the HTML page containing tables to show the YAML schema.
Usage:
python schema2html path/to/yaml
The HTML file will be saved under the same directory containing YAML file.
The basename of the HTML file will be the same as the YAML file.
"""

import logging
import os
import sys

import yaml
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def load_template():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    environment = Environment(
        loader=FileSystemLoader(os.path.join(working_dir, "templates"))
    )
    template = environment.get_template("schema.html")
    return template


def list_options(values):
    """Lists the values as options in the description."""
    if not values:
        return ""
    if not isinstance(values, list):
        return values
    values = [f"<code>{v}</code>" for v in values]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} or {values[1]}"
    return ", ".join(values[:-1]) + ", or " + values[-1]


def process_schemas(schema_dict):
    schema_kind = schema_dict["kind"]["allowed"][0]
    # Each schema will have 3 keys: "name", "schema" and "items".
    # name: stores the name of the schema, which will be used as the title of the HTML table.
    # schema: stores the schema as a dictionary
    # items: a list of "item" dictionaries. Each "item" has 3 keys: "key", "type", "description".
    # At the beginning, put the "root" schema into the list.
    schemas = [dict(name=schema_kind, schema=schema_dict)]

    i = 0
    # For each schema in the list, put the top level key/value into a table.
    # If value contains additional schema, the schema will be pushed into the list.
    while i < len(schemas):
        items = []
        if not schemas[i].get("schema"):
            logger.error("Empty schema for %s", schemas[i].get("name"))
        for key, val in schemas[i].get("schema", {}).items():
            # Schema should be a dictionary
            if not isinstance(val, dict):
                logger.error(
                    "Invalid schema value:\nkey=%s\nval=%s\n%s", key, val, schemas[i]
                )
                continue
            if "meta" in val:
                description = f"{val['meta']} "
            else:
                description = ""
            val_type = val.get("type")
            if val_type in ["dict", "list"]:
                # If the value type is dict or list,
                # there should be a schema for the dict or elements in the list.
                if not "schema" in val:
                    logger.warning(
                        "Missing schema for type: %s\nkey=%s\nval=%s\n%s",
                        val_type,
                        key,
                        val,
                        schemas[i],
                    )

                if val_type == "list":
                    # When value type is a list,
                    # the schema for each element should have a type
                    if "type" not in val["schema"]:
                        logger.error(
                            "Missing type in list schema\nkey=%s\nval=%s\n%s",
                            key,
                            val,
                            schemas[i],
                        )
                        continue
                    description += f'List of {list_options(val["schema"]["type"])}. '
                    if "schema" in val["schema"]:
                        name = f'{schemas[i].get("name")}.{key}'
                        schemas.append(dict(name=name, schema=val["schema"]["schema"]))
                        description += f'For each element, see <a href="#{name}"><code>{name}</code></a> schema.'

                if val_type == "dict" and "schema" in val:
                    name = f'{schemas[i].get("name")}.{key}'
                    schemas.append(dict(name=name, schema=val["schema"]))
                    description += (
                        f'See <a href="#{name}"><code>{name}</code></a> schema.'
                    )

            else:
                rules = []
                if "allowed" in val:
                    allowed_values = val["allowed"]
                    if len(allowed_values) == 1:
                        rules.append(f"Must be <code>{allowed_values[0]}</code>")
                    else:
                        options = list_options(allowed_values)
                        rules.append(options)
                if "min" in val:
                    rules.append(f"Minimum: <code>{val['min']}</code>")
                if "max" in val:
                    rules.append(f"Maximum: <code>{val['max']}</code>")
                description += ", ".join(rules)

            items.append(
                {
                    "key": key,
                    "type": list_options(val_type),
                    "description": description,
                }
            )
        schemas[i]["rows"] = items
        i += 1
    return schemas


def main():
    yaml_path = sys.argv[1]

    with open(yaml_path, "r", encoding="utf-8") as f:
        schema_dict = yaml.safe_load(f)

    template = load_template()
    schemas = process_schemas(schema_dict)
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(yaml_path)),
        os.path.splitext(os.path.basename(yaml_path))[0] + ".html",
    )
    html = template.render(schema_list=schemas)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved to {output_file}")


if __name__ == "__main__":
    main()
