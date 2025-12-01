# Spacy Processing for FoodKG Ingredients

This folder contains a simple Spacy-based NLP processor for normalizing ingredient lists from the FoodKG recipe database.

## Overview

The `spacy_processor.py` script:
1. Connects to the PostgreSQL database
2. Fetches recipes from the FoodKG table
3. Uses Spacy NLP to extract core ingredient names
4. Saves the processed results to CSV

## How It Works

The processor performs the following steps on each ingredient:

1. **Lowercase** the text
2. **Remove numbers and fractions** (e.g., "1/2", "2")
3. **Remove measurement units** (cups, tbsp, oz, etc.)
4. **Remove punctuation** and extra spaces
5. **Extract nouns** using Spacy's part-of-speech tagging
6. **Join nouns** to form the normalized ingredient name

### Example

```
Input:  ['1 c. firmly packed brown sugar', '1/2 c. evaporated milk', '1/2 tsp. vanilla']
Output: ['brown sugar', 'evaporated milk', 'vanilla']
```

## Requirements

Install required packages:

```bash
pip install spacy pandas sqlalchemy psycopg2-binary
python -m spacy download en_core_web_sm
```

## Usage

### Run the full processing:

```bash
python spacy_processor.py
```

This will:
- Connect to the database
- Process all recipes in the FoodKG table
- Save results to `data/output/foodkg_spacy_processed.csv`

### Test with limited data:

Edit the `main()` function and change:
```python
df = fetch_foodkg_data(engine, limit=100)  # Process only 100 recipes
```

## Output

The script creates a CSV file with columns:
- `id`: Recipe ID
- `title`: Recipe name
- `ingredients`: Original ingredient list
- `ingredients_normalized`: Processed ingredient list

## Database Connection

The script uses the same database credentials as `Neo4Jwriter.py`:
- Host: awesome-hw.sdsc.edu
- Database: nourish
- User: b6hill

## Notes

- The processor is intentionally simple and rule-based
- It focuses on extracting nouns as the core ingredient identifiers
- More complex ingredient names (like "cream of mushroom soup") may be simplified to just the nouns
- For more sophisticated processing, consider the LLM-based approach in `llm_util.py`
