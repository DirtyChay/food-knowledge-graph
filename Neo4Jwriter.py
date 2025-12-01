from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import os
import ast
from sqlalchemy import create_engine


# Update with your credentials
URI = "bolt://67.58.49.84:7687"
USER_N4J = "neo4j"
PASSWORD_N4J = "x7t4p2ks"  # or the password you set


#POSTGRES INFO
HOST = "awesome-hw.sdsc.edu"
PORT = 5432
DATABASE = "nourish"
USER_PG = "b6hill"
PASSWORD_PG = "dse203#2025"


class PostgresCon:

    engine = None
    def __init__(self):
        self.get_engine()

    def get_engine(self):
        if self.engine:
            return self.engine
        else:
            try:
                self.init_engine()
                return self.engine
            except Exception as e:
                print(f"Failed to initialize engine due to {e}")
                return None

    def init_engine(self):
        self.engine = create_engine(f"postgresql+psycopg2://{USER_PG}:{PASSWORD_PG}@{HOST}:{PORT}/{DATABASE}")
        return


class GraphDB:
    
    driver = None
    enable_logging = True
    
    def __init__(self):
        self.get_driver()
        
    def get_driver(self):
        if self.driver:
            return self.driver()
        else:
            try:
                self.init_driver()
                return self.driver
            except Exception as e:
                print(f"Failed to initialize driver due to {e}")
                return None
                
                
                
    def init_driver(self):
        self.driver = GraphDatabase.driver(URI, auth=(USER_N4J, PASSWORD_N4J))
        return

    def run_query(self, query, parameters=None, single=False):
        """
        Run a Cypher query with optional parameters.

        Args:
            query (str): The Cypher query to execute.
            parameters (dict, optional): Query parameters.
            single (bool): If True, return only the first result.

        Returns:
            list | dict | None: Query results.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            records = [r.data() for r in result]
            if single:
                return records[0] if records else None
            return records
        

    def create_node(self, label, properties):
        if self.node_exists(label, properties):
            if self.enable_logging:
                print("Node already exists")
            return self.get_node(label, properties)
        
        query = f"CREATE (n:{label} $props) RETURN n"
        return self.run_query(query, {"props": properties}, single=True)

    def get_nodes_matching_label(self, label):
        query = f"MATCH (n:{label}) RETURN n"
        return self.run_query(query)
    
    def get_node(self, label, properties):
        prop_keys = list(properties.keys())
        prop_str = ", ".join([f"{key}: ${key}" for key in prop_keys])

        query = f"MATCH (n:{label} {{{prop_str}}}) RETURN n LIMIT 1"
        return self.run_query(query, properties, single=True)
    
    def node_exists(self, label, properties):
        prop_keys = list(properties.keys())
        prop_str = ", ".join([f"{key}: ${key}" for key in prop_keys])

        query = f"MATCH (n:{label} {{{prop_str}}}) RETURN n LIMIT 1"
        result = self.run_query(query, properties, single=True)
        return result is not None
    
    def relationship_exists(self, label1, prop1, rel_type, label2, prop2):
        prop1_str = ", ".join([f"{key}: $prop1.{key}" for key in prop1.keys()])
        prop2_str = ", ".join([f"{key}: $prop2.{key}" for key in prop2.keys()])

        query = f"""
        MATCH (a:{label1} {{{prop1_str}}})-[r:{rel_type}]->(b:{label2} {{{prop2_str}}})
        RETURN r LIMIT 1
        """
        result = self.run_query(query, {"prop1": prop1, "prop2": prop2}, single=True)
        return result is not None
    
    def get_relationship(self, label1, prop1, rel_type, label2, prop2):
        prop1_str = ", ".join([f"{key}: $prop1.{key}" for key in prop1.keys()])
        prop2_str = ", ".join([f"{key}: $prop2.{key}" for key in prop2.keys()])

        query = f"""
        MATCH (a:{label1} {{{prop1_str}}})-[r:{rel_type}]->(b:{label2} {{{prop2_str}}})
        RETURN type(r) AS relationship, properties(r) AS edge_properties LIMIT 1
        """
        return self.run_query(query, {"prop1": prop1, "prop2": prop2}, single=True)
    

    def create_nutrient_nodes(self, nutrient_records):
        """
        nutrient_records : list[dict]
            Example:
            [
                {
                    "nutrientId": 1003,
                    "nutrientName": "Protein",
                    "unitName": "g"
                },
                {
                    "nutrientId": 1004,
                    "nutrientName": "Total Fat",
                    "unitName": "g"
                }
            ]
        """
        query = """
        UNWIND $nutrient_records AS record
        MERGE (n:Nutrient {nutrientId: record.nutrientId})
        SET n.nutrientName = record.nutrientName,
            n.unitName = record.unitName
        """
    
        self.run_query(query, {"nutrient_records":nutrient_records})

    def create_ingredient_nodes(self, ingredient_records):
        """
        ingredient_records should look like this:
    
            ingredient_records = [
                {"ingredientName": "brown sugar"},
                ...
                {"ingredientName": "butter"}
            ]
        """
    
        query = """
            UNWIND $ingredients AS ing
            MERGE (i:Ingredient {ingredientName: ing.ingredientName})
        """
        return self.run_query(query, {"ingredients": ingredient_records})
    
    def create_product_nodes(self, product_records):
        """
        product_records should look like this:
    
            product_records = [
                {
                    "productId": 100001,
                    "productDescription": "Organic Whole Milk"
                },
                {
                    "productId": 100002,
                    "productDescription": "Greek Yogurt, Vanilla"
                },
                ...
            ]
        """
    
        query = """
            UNWIND $product_records AS p
            MERGE (prod:Product {productId: p.productId})
            SET prod.productDescription = p.productDescription
        """
    
        return self.run_query(query, {"product_records": product_records})

    def create_recipe_nodes(self, recipe_records):
        """
        recipe_records should look like this:
    
            recipe_records = [
                {
                    "recipeId": 0,
                    "recipeName": "No-Bake Nut Cookies",
                    "originalIngredients": "["1 c. firmly packed brown sugar",..."2 tbsp butter"]"
                },
                ...
                {
                    "recipeId": 2,
                    "recipeName": "Creamy Corn",
                    "originalIngredients": "["2 (16 oz.) pkg. frozen corn",..."1/2 c. butter"]""
                }
            ]
    
        """
        
        query = """
            UNWIND $recipes AS r
            MERGE (rec:Recipe {recipeId: r.recipeId})
            SET rec.recipeName = r.recipeName,
                rec.originalIngredients = r.originalIngredients
            """
        return self.run_query(query, {"recipes": recipe_records})


    def create_hasNutrient_edges(self, edge_records):
        """ 
        Example:
            edge_records = [
                {"productId": 100001, "nutrientId": 1003, "amount": 5.0},
                {"productId": 100001, "nutrientId": 1004, "amount": 10.0},
                {"productId": 100002, "nutrientId": 1003, "amount": 8.0}
            ]
        """
    
        query = """
            UNWIND $edges AS e
            MATCH (p:Product {productId: e.productId})
            MATCH (n:Nutrient {nutrientId: e.nutrientId})
            MERGE (p)-[r:HAS_NUTRIENT]->(n)
            SET r.amount = e.amount
        """
    
        return self.run_query(query, {"edges": edge_records})

    def create_hasProduct_edges(self, edge_records):
        """
            edge_records = [
                {"ingredientName": "butter", "productId": 100001},
                {"ingredientName": "milk", "productId": 100001},
                {"ingredientName": "butter", "productId": 100002}
            ]
        """
    
        query = """
            UNWIND $edges AS e
            MATCH (ing:Ingredient {ingredientName: e.ingredientName})
            MATCH (p:Product {productId: e.productId})
            MERGE (ing)-[:HAS_PRODUCT]->(p)
        """
    
        return self.run_query(query, {"edges": edge_records})
    
    def create_hasIngredient_edges(self, edge_records):
        """
        edge_records should look like this:
    
            edge_records = [
                {
                    "recipeId": 0,
                    "ingredientName": "brown sugar"
                },
                {
                    "recipeId": 0,
                    "ingredientName": "evaporated milk"
                },
                {
                    "recipeId": 1,
                    "ingredientName": "chicken"
                },
                ...
            ]

        """
    
        query = """
            UNWIND $edges AS e
            MATCH (r:Recipe {recipeId: e.recipeId})
            MATCH (i:Ingredient {ingredientName: e.ingredientName})
            MERGE (r)-[:HAS_INGREDIENT]->(i)
        """
    
        return self.run_query(query, {"edges": edge_records})

    
    def create_relationship(self, label1, prop1, rel_type, label2, prop2, edge_prop=None):
        """
        Create a relationship between two nodes:
          1. Create nodes if they don’t exist (using existing helpers)
          2. If the relationship exists, return it and print a message
          3. Otherwise, create and return it
        """
        # If node1 doesn't exist, Create it
        if not self.node_exists(label1, prop1):
            if self.enable_logging:
                print(f"{label1} node does not exist — creating it.")
            self.create_node(label1, prop1)

        # If node2 doesn't exist, Create it
        if not self.node_exists(label2, prop2):
            if self.enable_logging:
                print(f"{label2} node does not exist — creating it.")
            self.create_node(label2, prop2)

        # Check if relationship already exists
        if self.relationship_exists(label1, prop1, rel_type, label2, prop2):
            if self.enable_logging:
                print(f"Relationship '{rel_type}' already exists between {label1} and {label2}.")
            return self.get_relationship(label1, prop1, rel_type, label2, prop2)

        # Build relationship creation query
        prop1_str = ", ".join([f"{key}: $prop1.{key}" for key in prop1.keys()])
        prop2_str = ", ".join([f"{key}: $prop2.{key}" for key in prop2.keys()])

        if edge_prop:
            edge_keys = list(edge_prop.keys())
            edge_str = ", ".join([f"{key}: $edge_prop.{key}" for key in edge_keys])
            edge_prop_clause = f"{{{edge_str}}}"
        else:
            edge_prop_clause = ""

        query = f"""
        MATCH (a:{label1} {{{prop1_str}}}), (b:{label2} {{{prop2_str}}})
        CREATE (a)-[r:{rel_type} {edge_prop_clause}]->(b)
        RETURN type(r) AS relationship, properties(r) AS edge_properties
        """

        params = {"prop1": prop1, "prop2": prop2}
        if edge_prop:
            params["edge_prop"] = edge_prop

        result = self.run_query(query, params, single=True)
        if self.enable_logging:
            print(f"Created new relationship '{rel_type}' between {label1} and {label2}.")
        return result

#RECIPE-INGREDIENTS TABLE##################################################################################################
def create_recipe_nodes(graph, recipe_df, batch_size=100000):
    n_rows = len(recipe_df)
    for i in range(0, n_rows, batch_size):
        records = recipe_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_recipe_nodes(records)

def create_ingredient_nodes(graph, ingredient_df, batch_size=100000):
    n_rows = len(ingredient_df)
    for i in range(0, n_rows, batch_size):
        records = ingredient_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_ingredient_nodes(records)

def create_recipe_ingredient_relationship(graph, hasIngredient_df, batch_size=100000):
    n_rows = len(hasIngredient_df)
    for i in range(0, n_rows, batch_size):
        records = hasIngredient_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_hasIngredient_edges(records)

def write_recipe_ingredient_rel(graph, csv_path):

    #Process the df
    df = pd.read_csv(csv_path)
    df["ingredients_normalized"] = df["ingredients_normalized"].apply(ast.literal_eval)
    df["ingredients_normalized"] = df["ingredients_normalized"].apply(ast.literal_eval)

    
    recipe_df = df[["id", "title", "ingredients"]].drop_duplicates().rename(columns={"id": "recipeId", "title": "recipeName", "ingredients": "originalIngredients"})
    df_exploded = df.explode("ingredients_normalized").reset_index(drop=True).rename(columns={"ingredients_normalized": "ingredientName"})
    ingredient_df = df_exploded[["ingredientName"]]
    ingredient_df.dropna(inplace = True)
    ingredient_df.drop_duplicates(subset=["ingredientName"], inplace=True)

    hasIngredient_df = df_exploded[["recipeId", "ingredientName"]]
    ingredient_df.dropna(inplace = True)
    ingredient_df.drop_duplicates(inplace=True)
    ################

    create_recipe_nodes(graph, recipe_df)
    create_ingredient_nodes(graph, ingredient_df)
    create_recipe_ingredient_relationship(graph, hasIngredient_df)


#PRODUCTS-INGREDIENTS TABLE################################################################################################
def create_product_nodes(graph, product_df, batch_size=100000):
    n_rows = len(product_df)
    for i in range(0, n_rows, batch_size):
        records = product_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_product_nodes(records)


def create_product_ingredient_relationship(graph, hasProduct_df, batch_size=100000):
    n_rows = len(hasProduct_df)
    for i in range(0, n_rows, batch_size):
        records = hasProduct_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_hasProduct_edges(records)

def write_product_ingredient_rel(graph, csv_path):

    #Process the df
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"fdc_id": "productId", "description": "productDescription", "mapped_ingredient": "ingredientName"})

    product_df = df[["productId", "productDescription"]].drop_duplicates()
    ingredient_df = df[["ingredientName"]].drop_duplicates()
    hasProduct_df = df[["productId", "ingredientName"]].drop_duplicates()
    ################

    create_product_nodes(graph, product_df)
    create_ingredient_nodes(graph, ingredient_df)
    create_product_ingredient_relationship(graph, hasProduct_df)

#PRODUCTS-NUTRIENTS TABLE##################################################################################################
def create_nutrient_nodes(graph, nutrient_df, batch_size=100000):
    n_rows = len(nutrient_df)
    for i in range(0, n_rows, batch_size):
        records = nutrient_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_nutrient_nodes(records)


def create_product_nutrient_relationship(graph, hasNutrient_df, batch_size=100000):
    n_rows = len(hasNutrient_df)
    for i in range(0, n_rows, batch_size):
        records = hasNutrient_df[i:min(n_rows, i+batch_size)].to_dict("records")
        graph.create_hasNutrient_edges(records)


def write_product_nutrient_rel(graph, postgres_conn):

    query = """
            SELECT
                fn.fdc_id AS productId,
                fb.description AS productDescription,
                fn.nutrient_id AS nutrientId,
                fn.amount AS amount,
                nm.name AS nutrientName,
                nm.unit_name AS unitName
            FROM
                usda_2022_branded_food_nutrients fn,
                usda_2022_nutrient_master nm,
                usda_2022_food_branded_experimental fb
            WHERE
                fn.nutrient_id = nm.id AND
                fb.fdc_id = fn.fdc_id;
        """

    df = pd.read_sql(query, postgres_conn.get_engine())
    df = df.rename(columns = dict(zip(['productid', 'productdescription', 'nutrientid', 'amount',
       'nutrientname', 'unitname'], ['productId', 'productDescription', 'nutrientId', 'amount',
       'nutrientName', 'unitName'])))

    nutrient_df = df[["nutrientId", "nutrientName", "unitName"]].drop_duplicates()
    product_df = df[["productId", "productDescription"]].drop_duplicates()
    hasNutrient_df = df[["nutrientId", "productId", "amount"]].drop_duplicates()

    create_nutrient_nodes(graph, nutrient_df)
    create_product_nodes(graph, product_df)
    create_product_nutrient_relationship(graph, hasNutrient_df)



#Neo4JWriter Main

postgres_conn = PostgresCon()
graph = GraphDB()
write_recipe_ingredient_rel(graph, <csv_path>)
write_product_ingredient_rel(graph, <csv_path>)
write_product_nutrient_rel(graph, postgres_conn)
