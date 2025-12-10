"""
A script for processing rows of data asynchronously using the OpenAI GPT model with resumable logic.

This script processes a large dataset in chunks, sending concurrent asynchronous requests to the
OpenAI API. Completed rows are saved to disk, and interrupted runs can resume without reprocessing
already-completed rows. Special features include exponential backoff on retries, request concurrency
limiting, and resumable writes to a CSV file.
"""
import asyncio
import os
import time
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

# Load environment variables from .env file (contains API keys)
load_dotenv()

# ------------- CONFIG -------------
# The specific GPT model to use for processing
MODEL_NAME = "gpt-4o-mini-2024-07-18"
# OpenAI API key loaded from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Column name in the CSV that contains unique identifiers for each row
ID_COLUMN = "fdc_id"
# Column name in the CSV that contains the text to be processed
TEXT_COLUMN = "description"

# File path where processed results will be saved
OUTPUT_PATH = "../data/output/processed_products_no_batch.csv"
# File path to the system prompt that defines the AI's behavior
SYSTEM_PROMPT_PATH = "prompts/system_message_products.txt"
# File path to unique ingredients list that will be appended to the system prompt
UNIQUE_INGREDIENTS_PATH = "../SpacyProcessing/spacy_unique_ingredients.txt"

# Maximum number of concurrent API requests allowed at once (to avoid rate limits)
CONCURRENCY = 35
# Number of rows to process before saving results to disk (enables incremental saves)
CHUNK_SIZE = 5000
# Maximum number of retry attempts for failed API requests
MAX_RETRIES = 5

# ------------- LOAD PROMPT -------------
# Read the unique ingredients list from file
with open(UNIQUE_INGREDIENTS_PATH, "r", encoding="utf-8") as f:
    UNIQUE_INGREDIENTS = f.read()

# Read the system prompt and append the unique ingredients to it
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_MSG_PRODUCTS = f.read() + UNIQUE_INGREDIENTS

# ------------- CLIENT -------------
# Initialize the asynchronous OpenAI client with the API key
client = AsyncOpenAI(api_key=API_KEY)


# ------------- RESUME LOGIC -------------
def get_completed_ids(output_path: str, id_column: str) -> set:
    """
    If OUTPUT_PATH exists, load completed IDs so we can skip them.

    This function checks if an output file already exists from a previous run.
    If it does, it loads all the IDs from that file so we can skip reprocessing them.
    This enables resumable processing after interruptions.

    Args:
        output_path: Path to the output CSV file
        id_column: Name of the column containing unique identifiers

    Returns:
        A set of string IDs that have already been processed
    """
    # If no output file exists yet, return an empty set (nothing has been processed)
    if not os.path.exists(output_path):
        return set()

    # For large files, only load the ID column (memory efficient)
    print(f"Loading completed IDs from {output_path}...")
    completed = pd.read_csv(output_path, usecols=[id_column])
    # Convert all IDs to strings and return as a set for fast lookups
    return set(completed[id_column].astype(str).tolist())


# ------------- ASYNC REQUEST + RETRIES -------------
async def call_model_single(
        row: pd.Series,
        system_prompt: str,
        model: str,
        semaphore: asyncio.Semaphore,
        id_column: str,
        text_column: str,
        max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Call the chat model for a single row with retries.

    This function sends a single row's data to the OpenAI API with automatic retry logic.
    It uses exponential backoff to handle rate limits and transient errors gracefully.

    Args:
        row: A pandas Series representing one row of data
        system_prompt: The system message that defines the AI's behavior
        model: The name of the OpenAI model to use
        semaphore:  Asyncio semaphore to limit concurrent requests
        id_column: Name of the column containing the row's unique ID
        text_column: Name of the column containing the text to process
        max_retries: Maximum number of retry attempts for failed requests

    Returns:
        A dictionary containing the row ID, input text, and API response (or error info)
    """
    # Extract the unique ID and text content from this row
    row_id = str(row[id_column])
    user_text = str(row[text_column])

    # Initialize retry tracking variables
    attempt = 0
    backoff = 1.0  # Initial backoff delay in seconds

    # Infinite loop that will break on success or after max_retries
    while True:
        attempt += 1
        try:
            # Use semaphore to limit concurrent API requests
            async with semaphore:
                # Make the API call to OpenAI
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    max_tokens=10,  # Limit response length
                )
                # Optionally log token usage for monitoring (currently commented out)
                # usage = resp.usage
                # print(
                #     f"[{row_id}] prompt_tokens={usage.prompt_tokens}, "
                #     f"cached_tokens={usage.prompt_tokens_details.cached_tokens}"
                # )

            # Extract the text content from the API response
            content = resp.choices[0].message.content
            # Return a successful result dictionary
            return {
                id_column: row_id,
                text_column: user_text,
                "response": content,
            }

        except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as e:
            # Check if we've exhausted all retry attempts
            if attempt >= max_retries:
                print(f"[{row_id}] giving up after {attempt} attempts:  {e}")
                # Return a failed record with error information for debugging
                return {
                    id_column: row_id,
                    text_column: user_text,
                    "response": None,
                    "error": str(e),
                }
            else:
                # Log the retry attempt and wait before retrying
                print(f"[{row_id}] error ({type(e).__name__}), retry {attempt}/{max_retries} in {backoff:. 1f}s...")
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff:  double the wait time for next retry


async def process_chunk_async(
        df_chunk: pd.DataFrame,
        system_prompt: str,
        model: str,
        id_column: str,
        text_column: str,
        concurrency: int,
) -> List[Dict[str, Any]]:
    """
    Process one chunk of the DataFrame asynchronously and return list of result dicts.

    This function takes a subset of the DataFrame and processes all rows concurrently
    using asyncio tasks. It creates one task per row and waits for all to complete.

    Args:
        df_chunk: A subset of the DataFrame to process
        system_prompt: The system message for the AI
        model: The OpenAI model name
        id_column: Name of the ID column
        text_column: Name of the text column
        concurrency: Maximum number of concurrent requests

    Returns:
        A list of dictionaries, one per row, containing results or errors
    """
    # Create a semaphore to limit concurrent API requests
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    # Create an async task for each row in the chunk
    for _, row in df_chunk.iterrows():
        task = asyncio.create_task(
            call_model_single(
                row=row,
                system_prompt=system_prompt,
                model=model,
                semaphore=semaphore,
                id_column=id_column,
                text_column=text_column,
            )
        )
        tasks.append(task)

    # Wait for all tasks to complete and gather their results
    results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
    return results


def append_results_to_csv(
        results: List[Dict[str, Any]],
        output_path: str,
        header: bool,
):
    """
    Append results to CSV on disk.

    This function takes a list of result dictionaries and appends them to the output CSV file.
    It supports incremental writes, which allows the process to be resumed if interrupted.

    Args:
        results: List of dictionaries containing processing results
        output_path: Path to the output CSV file
        header: Whether to write the CSV header row (True for first write, False for appends)
    """
    # If no results to write, exit early
    if not results:
        return

    # Convert list of dictionaries to a DataFrame
    df_out = pd.DataFrame(results)
    # Append to the CSV file (mode="a" for append)
    df_out.to_csv(
        output_path,
        mode="a",
        header=header,
        index=False,
    )


def process_dataframe_resumable(df: pd.DataFrame):
    """
    Main driver:
      - Filters out already-completed IDs (if OUTPUT_PATH exists)
      - Processes remaining rows in chunks (async)
      - Appends to CSV after each chunk

    This function orchestrates the entire processing pipeline.  It checks for previously
    processed rows, divides remaining work into chunks, processes each chunk asynchronously,
    and saves results incrementally.

    Args:
        df: The full DataFrame to process
    """
    # --- Resume: skip rows already processed ---
    # Load IDs that have already been processed in previous runs
    completed_ids = get_completed_ids(OUTPUT_PATH, ID_COLUMN)
    print(f"Found {len(completed_ids)} completed rows from previous runs.")

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    # Ensure ID column is string type for consistent comparison
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    # Filter out rows that have already been processed
    df_to_do = df[~df[ID_COLUMN].isin(completed_ids)]

    # Calculate and display processing statistics
    total_remaining = len(df_to_do)
    print(f"Total rows in DF: {len(df)}")
    print(f"Remaining to process: {total_remaining}")

    # Exit if there's nothing left to process
    if total_remaining == 0:
        print("Nothing to do, all rows already processed.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Determine if this is the first write (need to write CSV header)
    first_write = not os.path.exists(OUTPUT_PATH)

    # --- Process in chunks ---
    # Loop through the DataFrame in chunks of CHUNK_SIZE
    for start in range(0, total_remaining, CHUNK_SIZE):
        # Calculate the end index for this chunk (don't exceed total_remaining)
        end = min(start + CHUNK_SIZE, total_remaining)
        # Extract the current chunk from the DataFrame
        df_chunk = df_to_do.iloc[start:end]
        print(f"\nProcessing rows {start}–{end} (chunk size {len(df_chunk)})...")

        # Record the start time for performance monitoring
        start_time = time.time()
        # Run this chunk asynchronously (processes all rows in chunk concurrently)
        results = asyncio.run(
            process_chunk_async(
                df_chunk=df_chunk,
                system_prompt=SYSTEM_MSG_PRODUCTS,
                model=MODEL_NAME,
                id_column=ID_COLUMN,
                text_column=TEXT_COLUMN,
                concurrency=CONCURRENCY,
            )
        )

        # Append results to CSV (incremental save enables resumability)
        append_results_to_csv(
            results=results,
            output_path=OUTPUT_PATH,
            header=first_write,
        )
        # After first write, don't write header again
        first_write = False

        # Calculate and display elapsed time for this chunk
        end_time = time.time()
        print(f"Chunk {start}–{end} done, wrote {len(results)} rows to {OUTPUT_PATH}.")
        print(f"Time elapsed: {end_time - start_time:.4f} seconds")

    print("\nAll remaining rows processed.")


if __name__ == "__main__":
    # We ran 4 batches that consisted of the first 100,000 rows of the dataset.
    # Therefore, we remove these rows. If not using batches, you can skip this step.

    # Number of batches already processed (each batch contains 25,000 rows)
    num_batches_run = 0  # ex. if 4 batches uploaded successfully, then set to 4
    # Calculate how many rows to skip (ex. 4 batches * 25,000 rows per batch = 100,000 rows)
    start_offset = 25000 * num_batches_run

    # Define the path to the input CSV file
    INPUT_CSV = "../data/raw/usda_2022_food_branded_experimental_DESCRIPTION_ONLY.csv"
    print("Loading full DataFrame...")
    # Load the CSV but skip the first 100,000 rows that were already processed in batches
    # skiprows=range(1, start_offset + 1) skips rows 1 through 100,000 (row 0 is the header)
    unbatched_df = pd.read_csv(INPUT_CSV, skiprows=range(1, start_offset + 1))
    print("Dataframe size:", len(unbatched_df))
    # Start the resumable processing pipeline
    process_dataframe_resumable(unbatched_df)
