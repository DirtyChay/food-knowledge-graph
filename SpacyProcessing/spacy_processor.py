import spacy
import re
import ast
import pandas as pd
from sqlalchemy import create_engine

# Database credentials
HOST = "awesome-hw.sdsc.edu"
PORT = 5432
DATABASE = "nourish"
USER_PG = "akrish"
PASSWORD_PG = "dse203#2025"

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Configuration constants
BATCH_SIZE = 4000
SPACY_BATCH_SIZE = 256

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
    'small', 'medium', 'large', 'extra', 'stick', 'sticks', 'whole', 'env', 'envelope'
]
pattern_units = r'\b(?:' + '|'.join(units) + r')\b'

# Expanded ingredient noise words
noise_words = {
    'optional', 'bite', 'size', 'taste', 'garnish', 'topping', 'filling', 'coating',
    'spread', 'sauce', 'dip', 'glaze', 'frosting', 'icing', 'powder', 'dust', 'sprinkle',
    'drizzle', 'splash', 'dash', 'pinch', 'touch', 'hint', 'bit', 'drop', 'dollop',
    'scoop', 'handful', 'portion', 'serving', 'side', 'accompaniment', 'mixture', 'blend',
    'half', 'halve', 'inch', 'sized', 'supreme', 'san', 'very',
    'hot', 'cold', 'warm', 'soft', 'hard', 'fresh', 'dried', 'cooked', 'raw', 'ripe',
    'chopped', 'sliced', 'diced', 'minced', 'shredded', 'grated', 'crushed', 'melted', 
    'frozen', 'thawed', 'boned', 'boneless', 'skinless', 'softened', 'beaten', 'sifted', 
    'divided', 'separated', 'ground', 'all-purpose', 'plain', 'self-rising', 'firmly', 
    'packed', 'light', 'heavy', 'white', 'broken', 'chipped', 'wedged', 'skinned', 
    'trimmed', 'boiling', 'finely', 'coarsely', 'thinly', 'lean'
}

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

def clean_ingredient(ing):
    """Clean a single ingredient string."""
    ing_clean = ing.lower()
    ing_clean = re.sub(r'\s*\([^)]*\)\s*', ' ', ing_clean)  # remove parenthesis
    ing_clean = re.sub(r'\b(and|or)\b', ',', ing_clean)  # split and/or with commas
    ing_clean = re.sub(r'\bof\b', ' ', ing_clean)  # remove of
    ing_clean = re.sub(r'\d+\/\d+|\d+', '', ing_clean)  # remove numbers and fractions
    ing_clean = re.sub(pattern_units, '', ing_clean, flags=re.IGNORECASE)  # remove units
    ing_clean = re.sub(r'[^a-zA-Z\s,]', '', ing_clean)  # remove punctuation except commas
    ing_clean = re.sub(r'\s+', ' ', ing_clean).strip()  # fix spacing
    return ing_clean

def process_ingredient_list(ingredient_list):
    """Process a single recipe's ingredient list."""
    if isinstance(ingredient_list, str):
        try:
            ingredient_list = ast.literal_eval(ingredient_list)
        except (ValueError, SyntaxError):
            return []
    
    recipe_ingredients = []
    seen = set()
    
    texts_to_process = []
    if ingredient_list:
        for ing in ingredient_list:
            cleaned = clean_ingredient(ing)
            if cleaned:
                texts_to_process.append(cleaned)
    
    if texts_to_process:
        for doc in nlp.pipe(texts_to_process, batch_size=SPACY_BATCH_SIZE):
            for chunk in doc.noun_chunks:
                clean_tokens = []
                for token in chunk:
                    if token.text.lower() not in noise_words and not token.is_punct and not token.is_stop:
                        if token == chunk.root:
                            clean_tokens.append(token.lemma_.lower())  # lemmatize root noun
                        else:
                            clean_tokens.append(token.text.lower())
                
                if clean_tokens:
                    processed = " ".join(clean_tokens)
                    if processed and processed not in noise_words and processed not in seen:
                        recipe_ingredients.append(processed)
                        seen.add(processed)
    
    return recipe_ingredients

def main():
    """Main processing function with batch processing."""
    engine = connect_to_database()
    total_recipes = get_total_count(engine)
    print(f"Total recipes to process: {total_recipes}")
    
    output_path = 'SpacyProcessing/foodkg_spacy_processed.csv'
    
    with open(output_path, 'w') as f:
        f.write('recipe_id,original_ingredients,processed_ingredients\n')
    
    processed_count = 0
    for offset in range(0, total_recipes, BATCH_SIZE):
        df_batch = fetch_foodkg_batch(engine, offset, BATCH_SIZE)
        if df_batch.empty:
            break
        
        df_batch['processed_ingredients'] = df_batch['ingredients'].apply(process_ingredient_list)
        df_results = df_batch[['id', 'ingredients', 'processed_ingredients']].copy()
        df_results.columns = ['recipe_id', 'original_ingredients', 'processed_ingredients']
        df_results.to_csv(output_path, mode='a', header=False, index=False)
        
        processed_count += len(df_batch)
        percentage = (processed_count / total_recipes) * 100
        print(f"Progress: {processed_count}/{total_recipes} ({percentage:.1f}%)")
    
    print(f"\nComplete! Processed {processed_count} recipes")

if __name__ == "__main__":
    main()