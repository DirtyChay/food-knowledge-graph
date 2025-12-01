"""
Simple Spacy-based ingredient processor for FoodKG data.
Connects to PostgreSQL database and processes ingredients using NLP.
"""

import spacy
import re
import pandas as pd
from sqlalchemy import create_engine, text

# Database credentials (same as Neo4Jwriter.py)
HOST = "awesome-hw.sdsc.edu"
PORT = 5432
DATABASE = "nourish"
USER_PG = "akrish"
PASSWORD_PG = "dse203#2025"

# Load Spacy model
print("Loading Spacy model...")
nlp = spacy.load("en_core_web_sm")

# Common measurement units to remove
units = [
    'c', 'cup', 'cups', 'tbsp', 'tsp', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons',
    'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds', 'g', 'gram', 'grams',
    'kg', 'kilogram', 'kilograms', 'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters',
    'qt', 'quart', 'quarts', 'pt', 'pint', 'pints', 'gal', 'gallon', 'gallons',
    'pkg', 'package', 'packages', 'can', 'cans', 'jar', 'jars', 'bottle', 'bottles',
    'box', 'boxes', 'bag', 'bags', 'container', 'containers',
    'pinch', 'dash', 'slice', 'slices', 'clove', 'cloves', 'piece', 'pieces',
    'bunch', 'bunches', 'head', 'heads', 'stalk', 'stalks', 'strip', 'strips',
    'small', 'medium', 'large', 'extra'
]
pattern_units = r'\b(?:' + '|'.join(units) + r')\b'


def connect_to_database():
    """Create database connection."""
    connection_string = f"postgresql+psycopg2://{USER_PG}:{PASSWORD_PG}@{HOST}:{PORT}/{DATABASE}"
    engine = create_engine(connection_string)
    return engine


def get_total_count(engine):
    """Get total number of recipes in FoodKG table."""
    query = 'SELECT COUNT(*) FROM "FoodKG"'
    result = pd.read_sql(query, engine)
    return result.iloc[0, 0]


def fetch_foodkg_batch(engine, offset, batch_size):
    """Fetch a batch of FoodKG data from database."""
    query = f'SELECT id, ingredients FROM "FoodKG" ORDER BY id LIMIT {batch_size} OFFSET {offset}'
    df = pd.read_sql(query, engine)
    return df


def singularize(word):
    """Simple singularization of English words."""
    # Don't singularize very short words
    if len(word) <= 3:
        return word
    
    # Regular patterns
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    elif word.endswith('ves'):
        return word[:-3] + 'f'
    elif word.endswith('oes'):
        return word[:-2]
    elif word.endswith('ses'):
        return word[:-2]
    elif word.endswith('xes'):
        return word[:-2]
    elif word.endswith('ches'):
        return word[:-2]
    elif word.endswith('shes'):
        return word[:-2]
    elif word.endswith('s') and not word.endswith('ss') and not word.endswith('us'):
        return word[:-1]
    
    return word


def process_ingredient_list(ingredient_list):
    """Process a single recipe's ingredient list."""
    import ast
    
    # Convert string to list if needed
    if isinstance(ingredient_list, str):
        ingredient_list = ast.literal_eval(ingredient_list)
    
    recipe_ingredients = []
    
    for ing in ingredient_list:
        # Lowercase
        ing_clean = ing.lower()
        
        # Remove numbers and fractions
        ing_clean = re.sub(r'\d+\/\d+|\d+', '', ing_clean)
        
        # Remove units
        ing_clean = re.sub(pattern_units, '', ing_clean, flags=re.IGNORECASE)
        
        # Remove punctuation and extra spaces
        ing_clean = re.sub(r'[^a-zA-Z\s]', '', ing_clean)
        ing_clean = re.sub(r'\s+', ' ', ing_clean).strip()
        
        # Skip if empty after cleaning
        if not ing_clean:
            continue
        
        # Use SpaCy to extract nouns
        doc = nlp(ing_clean)
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        
        if nouns:
            # Singularize each noun
            singular_nouns = [singularize(noun) for noun in nouns]
            # Join nouns and add to recipe ingredients
            ingredient_name = ' '.join(singular_nouns)
            recipe_ingredients.append(ingredient_name)
    
    return recipe_ingredients


def main():
    """Main processing function with batch processing."""
    import os
    
    # Connect to database
    print("Connecting to database...")
    engine = connect_to_database()
    
    # Get total count
    total_recipes = get_total_count(engine)
    print(f"Total recipes to process: {total_recipes}")
    
    # Set batch size
    batch_size = 200
    
    # Output file
    output_path = 'data/output/foodkg_spacy_processed.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs('data/output', exist_ok=True)
    
    # Initialize or clear output file with headers
    with open(output_path, 'w') as f:
        f.write('recipeId,original_ingredients,processed_ingredients\n')
    
    # Process in batches
    processed_count = 0
    
    for offset in range(0, total_recipes, batch_size):
        # Fetch batch
        print(f"\nFetching batch: recipes {offset} to {offset + batch_size}...")
        df_batch = fetch_foodkg_batch(engine, offset, batch_size)
        
        if df_batch.empty:
            break
        
        # Process each recipe in batch
        results = []
        for idx, row in df_batch.iterrows():
            recipe_id = row['id']
            original_ingredients = row['ingredients']
            
            # Process ingredients
            processed = process_ingredient_list(original_ingredients)
            
            results.append({
                'recipeId': recipe_id,
                'original_ingredients': original_ingredients,
                'processed_ingredients': processed
            })
            
            processed_count += 1
        
        # Append batch results to CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_path, mode='a', header=False, index=False)
        
        # Print progress
        percentage = (processed_count / total_recipes) * 100
        print(f"Progress: {processed_count}/{total_recipes} recipes ({percentage:.1f}%)")
    
    print(f"\n✓ Processing complete! Results saved to {output_path}")
    print(f"Total recipes processed: {processed_count}")


if __name__ == "__main__":
    main()
