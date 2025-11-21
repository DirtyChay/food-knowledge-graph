def createRecipeNodeQuery(recipe_name, recipe_id, ingredients):
    """
    Takes in recipe info and returns Cypher query string to create
    Recipe node.
    
    Args:
        recipe_name (str): Name of recipe.
        recipe_id (int): ID of recipe.
        ingredients (list): List of ingredients.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    CREATE (r:Recipe {{
        name: {repr(recipe_name)}, 
        id: {repr(recipe_id)}, 
        ingredients: {repr(ingredients)}
    }})
    RETURN r
    """

    return query


def createIngredientNodeQuery(ingredient_name):
    """
    Takes in ingredient info and returns Cypher query string
    to create Ingredient node.
    
    Args:
        ingredient_name (str): Name of ingredient.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    CREATE (i:Ingredient {{
        name: {repr(ingredient_name)}
    }})
    RETURN i
    """

    return query


def createProductNodeQuery(product_name, product_id):
    """
    Takes in product info and returns Cypher query string
    to create Product node.
    
    Args:
        product_name (str): Name of product.
        product_id (int): ID of product.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    CREATE (p:Product {{
        name: {repr(product_name)}, 
        id: {repr(product_id)}
    }})
    RETURN p
    """

    return query


def createNutrientNodeQuery(nutrient_name, nutrient_id, unit_name):
    """
    Takes in nutrient info and returns Cypher query string
    to create Nutrient node.
    
    Args:
        nutrient_name (str): Name of nutrient.
        nutrient_id (int): ID of nutrient.
        unit_name (str): Name of unit ('g', 'mg', etc.).
    
    Returns:
        str: String Cypher query.
    """

    query = f"""
    CREATE (n:Nutrient {{
        name: {repr(nutrient_name)}, 
        id: {repr(nutrient_id)}, 
        unit: {repr(unit_name)}
    }})
    RETURN n
    """

    return query


def createRecipeToIngredientEdge(recipe_id, ingredient_name):
    """
    Takes in a recipe ID and an ingredient name and returns a Cypher 
    query string to create a HAS_INGREDIENT relationship between them.
    
    Args:
        recipe_id (int): ID of recipe.
        ingredient_name (str): Name of ingredient.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    MATCH (r:Recipe {{id: {repr(recipe_id)}}})
    MATCH (i:Ingredient {{name: {repr(ingredient_name)}}})
    CREATE (r)-[rel:HAS_INGREDIENT]->(i)
    RETURN r, rel, i
    """

    return query


def createIngredientToProductEdge(ingredient_name, product_id):
    """
    Takes in an ingredient name and a product ID and returns a Cypher 
    query string to create a FOUND_IN relationship between them.
    
    Args:
        ingredient_name (str): The name of the ingredient to match.
        product_id (int): The ID of the product to match.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    MATCH (i:Ingredient {{name: {repr(ingredient_name)}}})
    MATCH (p:Product {{id: {repr(product_id)}}})
    CREATE (i)-[rel:FOUND_IN]->(p)
    RETURN i, rel, p
    """

    return query


def createProductToNutrientEdge(product_id, nutrient_id, amount):
    """
    Takes in a product ID, nutrient ID, and amount, and returns a
    Cypher query string to create a HAS_NUTRIENT relationship
    between them with an amount.
    
    Args:
        product_id (int): ID of product.
        nutrient_id (int): ID of nutrient.
        amount (int): Amount of nutrient.

    Returns:
        str: String Cypher query.
    """

    query = f"""
    MATCH (p:Product {{id: {repr(product_id)}}})
    MATCH (n:Nutrient {{id: {repr(nutrient_id)}}})
    CREATE (p)-[rel:HAS_NUTRIENT {{amount: {repr(amount)}}}]->(n)
    RETURN p, rel, n
    """

    return query


def checkRecipeExists(recipe_id):
    """
    Generates a Cypher query to check if Recipe node exists by its ID.
    
    Args:
        recipe_id (int): ID of recipe.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    OPTIONAL MATCH (r:Recipe {{id: {repr(recipe_id)}}})
    RETURN r IS NOT NULL AS nodeExists
    """
    return query


def checkIngredientExists(ingredient_name):
    """
    Generates a Cypher query to check if Ingredient node exists by its name.
    
    Args:
        ingredient_name (str): Name of ingredient to check.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    OPTIONAL MATCH (i:Ingredient {{name: {repr(ingredient_name)}}})
    RETURN i IS NOT NULL AS nodeExists
    """
    return query


def checkProductExists(product_id):
    """
    Generates a Cypher query to check if Product node exists by its ID.
    
    Args:
        product_id (int): ID of product.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    OPTIONAL MATCH (p:Product {{id: {repr(product_id)}}})
    RETURN p IS NOT NULL AS nodeExists
    """
    return query


def checkNutrientExists(nutrient_id):
    """
    Generates a Cypher query to check if Nutrient node exists by its ID.
    
    Args:
        nutrient_id (int): ID of nutrient.
        
    Returns:
        str: String Cypher query.
    """

    query = f"""
    OPTIONAL MATCH (n:Nutrient {{id: {repr(nutrient_id)}}})
    RETURN n IS NOT NULL AS nodeExists
    """
    return query


print(createRecipeNodeQuery('Cake', 4, ['flour', 'egg']))
print(createIngredientNodeQuery('Egg'))
print(createProductNodeQuery('Costco Organic Eggs', 5000))
print(createNutrientNodeQuery('Protein', 301, 'g'))
print(createRecipeToIngredientEdge(4, 'Egg'))
print(createIngredientToProductEdge('Egg', 5001))
print(createProductToNutrientEdge(5000, 301, 10))
print(checkRecipeExists(4))
print(checkIngredientExists('Egg'))
print(checkProductExists(5000))
print(checkNutrientExists(301))
