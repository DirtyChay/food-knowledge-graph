# Spacy Processing for FoodKG Ingredients

This folder contains Spacy-based NLP processing for normalizing ingredient lists from the FoodKG recipe table.

## Overview

The `spacy_processor.py` script:

1. Connects to PostgreSQL database
2. Fetches recipes from FoodKG table
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

## Output

The script creates a CSV file with columns:

- `id`: Recipe ID
- `title`: Recipe name
- `ingredients`: Original ingredient list
- `ingredients_normalized`: Processed ingredient list

## spaCy Output Analysis

The `test_spacy.ipynb` notebook tests and benchmarks the output. To run it, simply run all the
cells and make sure the spaCy output is in the data folder of this project.
