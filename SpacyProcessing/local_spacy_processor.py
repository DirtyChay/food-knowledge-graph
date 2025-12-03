"""
Simple Spacy-based ingredient processor for FoodKG data.
Connects to local data and processes ingredients using NLP.
"""

import os

import pandas as pd

from spacy_processor import process_ingredient_list

# Set to a number to limit, set to None for all
num_rows_to_run = None
# num_rows_to_run = 100
# Configuration constants
BATCH_SIZE = 4000
SPACY_BATCH_SIZE = 256


def main():
    """Main processing function with batch processing."""
    # Load df and cut down to number of relevant rows
    foodkg_df = pd.read_csv('../data/raw/nourish_public_FoodKG.csv')
    foodkg_df.sort_values(by=['id'], inplace=True)
    if num_rows_to_run is not None:
        foodkg_df = foodkg_df.head(num_rows_to_run)

    total_recipes = len(foodkg_df)
    print(f"Total recipes to process: {total_recipes}")

    # Output file
    output_path = '../data/SpacyProcessing'
    os.makedirs(output_path, exist_ok=True)
    output_file_path = output_path + "/foodkg_spacy_processed.csv"
    # Initialize or clear output file with headers
    with open(output_file_path, 'w') as f:
        f.write('recipe_id,original_ingredients,processed_ingredients\n')

    processed_count = 0
    for offset in range(0, total_recipes, BATCH_SIZE):
        df_batch = foodkg_df.iloc[offset: offset + BATCH_SIZE].copy()
        if df_batch.empty:
            break

        df_batch['processed_ingredients'] = df_batch['ingredients'].apply(process_ingredient_list)
        df_results = df_batch[['id', 'ingredients', 'processed_ingredients']].copy()
        df_results.columns = ['recipe_id', 'original_ingredients', 'processed_ingredients']
        df_results.to_csv(output_file_path, mode='a', header=False, index=False)

        processed_count += len(df_batch)
        percentage = (processed_count / total_recipes) * 100
        print(f"Progress: {processed_count}/{total_recipes} ({percentage:.1f}%)")

    print(f"\nComplete! Processed {processed_count} recipes")


if __name__ == "__main__":
    main()
