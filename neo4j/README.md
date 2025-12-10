# food-knowledge-graph

DSE 203 Food Knowledge Graph

## LLM Processing Flow

1. Acquire tables via Postgres and save to CSV
2. Read the recipe CSV and call LLM with prompt on every row to generated processed_ingredients.csv
3. Create a "unique_ingredients" list from the processed_ingredients.csv
4. Read the product CSV and provide to an LLM a prompt, the unique_ingredients list, and the row of data. Save the
   resulting processed_products.csv
5. Test the outputs of the processed tables with a human-annotated test set

