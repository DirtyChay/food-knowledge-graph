CREATE INDEX recipe_id_index IF NOT EXISTS
FOR (r:Recipe)
ON (r.recipeId);

CREATE INDEX ingredient_name_index IF NOT EXISTS
FOR (i:Ingredient)
ON (i.ingredientName);

CREATE INDEX product_id_index IF NOT EXISTS
FOR (p:Product)
ON (p.productId);

CREATE INDEX nutrient_id_index IF NOT EXISTS
FOR (n:Nutrient)
ON (n.nutrientId);
