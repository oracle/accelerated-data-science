import json
import re

from pyparsing import ParseException

from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton

"""
USER_TRANSFORMATION_FUNCTION template: It is used to transform the user provided SQL query to the 
transformation function

Args:
    function_name: Transformation function name 
    input : The input placeholder for the FROM clause
"""
USER_TRANSFORMATION_FUNCTION = """def {function_name}(input):
    sql_query = f\"\"\"{query}\"\"\"
    return sql_query"""


class TransformationQueryValidator:
    @staticmethod
    def __verify_sql_query_plan(parser_plan, input_symbol: str):
        """
        Once the sql parser has parsed the query,
        This function takes the parser plan as an input, It checks for the table names
        and verifies to ensure that there should only be single table and that too should have the placeholder name
        A regex has been added to cater to common table expressions
        Args:
            parser_plan: A Spark sqlParser ParsePlan object.
                parser_plan contain the project and unresolved relation items
                project: list of unresolved attributes - table field names
                UnresolvedRelation: list of unresolved relation attributes - table names
                e.g. : Project ['user_id, 'credit_score], 'UnresolvedRelation [DATA_SOURCE_INPUT], [], false
            input_symbol (Transformation): The table name to be matched.
        """
        plan_items = json.loads(parser_plan.toJSON())
        plan_string = parser_plan.toString()
        cte = re.findall(r"CTE \[(.*?)\]", plan_string)
        table_names = []
        for plan_item in plan_items:
            if (
                plan_item["class"]
                == "org.apache.spark.sql.catalyst.analysis.UnresolvedRelation"
            ):
                table = plan_item["multipartIdentifier"]
                res = table.strip("][").split(", ")
                if len(res) >= 2:
                    raise ValueError("FROM Clause has invalid input {0}".format(table))
                else:
                    if res[0].lower() != input_symbol.lower():
                        raise ValueError(
                            f"Incorrect table template name, It should be {input_symbol}"
                        )
                    if table not in cte:
                        table_names.append(f"{table}")
                        if len(table_names) > 1:
                            raise ValueError("Multiple tables are not supported ")

    @staticmethod
    def verify_sql_input(query_input: str, input_symbol: str):
        """
        verifies the query provided by user to ensure that its a valid sql query.

        Args:
            query_input: A Spark Sql query
            input_symbol (Transformation): The table name to be matched.
        """
        spark = SparkSessionSingleton().get_spark_session()
        parser = spark._jsparkSession.sessionState().sqlParser()
        try:
            parser_plan = parser.parsePlan(query_input)
        except ParseException as pe:
            raise ParseException(
                f"Unable to parse the sql expression, exception occurred:  {pe}"
            )

        # verify if the parser plan has only FROM DATA_SOURCE_INPUT template
        TransformationQueryValidator.__verify_sql_query_plan(parser_plan, input_symbol)

    @staticmethod
    def create_transformation_template(
        query: str, input_symbol: str, function_name: str
    ):
        """
        Creates the query transformation function to ensure backend integrity
        Args:
            query: A Spark Sql query
            input_symbol (Transformation): The table name to be used.
            function_name : The name of the transformation function
        """
        transformation_query = query.replace(input_symbol, "{input}")
        output = USER_TRANSFORMATION_FUNCTION.format(
            query=transformation_query, function_name=function_name
        )
        return output
