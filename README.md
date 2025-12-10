# food-knowledge-graph

DSE 203 Food Knowledge Graph

## LLM Processing Flow

1. Acquire tables via Postgres and save to CSV
2. Read the recipe CSV and call LLM with prompt on every row to generated processed_ingredients.csv
3. Create a "unique_ingredients" list from the processed_ingredients.csv
4. Read the product CSV and provide to an LLM a prompt, the unique_ingredients list, and the row of data. Save the
   resulting processed_products.csv
5. Test the outputs of the processed tables with a human-annotated test set




END 2 END INSTRUCTIONS:

1) SPACY CODE, RECIPE PREPROCESSING - Navigate to SpacyProcessing folder and execute according to the readme.md file provided in that folder. Save the output csv as "foodkg_spacy_processed.csv" to be used later
2) LLM CODE, PRODUCTS PREPROCESSING -
3) NEO4J GRAPH CREATION - Navigate to neo4j folder and execute according to the readme.md file provided in that folder. Make sure to copy output CSVs from steps 1 and 2 to the local NEO4J folder path where you are running the graph creation script

