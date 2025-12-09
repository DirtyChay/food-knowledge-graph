import asyncio
import os
import time
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

load_dotenv()

# ------------- CONFIG -------------
MODEL_NAME = "gpt-4o-mini-2024-07-18"
API_KEY = os.getenv("OPENAI_API_KEY")

ID_COLUMN = "fdc_id"  # unique identifier per row
TEXT_COLUMN = "description"  # text you send in the user message

OUTPUT_PATH = "results_async.csv"  # where we store results
SYSTEM_PROMPT_PATH = "../llm_processing/prompts/system_message_products.txt"
UNIQUE_INGREDIENTS_PATH = "../SpacyProcessing/spacy_unique_ingredients.txt"

CONCURRENCY = 35  # how many requests in-flight at once
CHUNK_SIZE = 5000  # how many rows to schedule before flushing to disk
MAX_RETRIES = 5  # per-request retry attempts

# ------------- LOAD PROMPT -------------
with open(UNIQUE_INGREDIENTS_PATH, "r", encoding="utf-8") as f:
    UNIQUE_INGREDIENTS = f.read()

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_MSG_PRODUCTS = f.read() + UNIQUE_INGREDIENTS

# ------------- CLIENT -------------
client = AsyncOpenAI(api_key=API_KEY)


# ------------- RESUME LOGIC -------------
def get_completed_ids(output_path: str, id_column: str) -> set:
    """
    If OUTPUT_PATH exists, load completed IDs so we can skip them.
    """
    if not os.path.exists(output_path):
        return set()

    # For large files, just load that one column
    print(f"Loading completed IDs from {output_path}...")
    completed = pd.read_csv(output_path, usecols=[id_column])
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
    Returns a dict with ID, input text, and response text.
    """
    row_id = str(row[id_column])
    user_text = str(row[text_column])

    attempt = 0
    backoff = 1.0  # seconds

    while True:
        attempt += 1
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    max_tokens=10,
                )
                # usage = resp.usage
                # print(
                #     f"[{row_id}] prompt_tokens={usage.prompt_tokens}, "
                #     f"cached_tokens={usage.prompt_tokens_details.cached_tokens}"
                # )

            # Extract main text; adjust if you want JSON, etc.
            content = resp.choices[0].message.content
            return {
                id_column: row_id,
                text_column: user_text,
                "response": content,
            }

        except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as e:
            if attempt >= max_retries:
                print(f"[{row_id}] giving up after {attempt} attempts: {e}")
                # Return a failed record with error for post-mortem
                return {
                    id_column: row_id,
                    text_column: user_text,
                    "response": None,
                    "error": str(e),
                }
            else:
                print(f"[{row_id}] error ({type(e).__name__}), retry {attempt}/{max_retries} in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff


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
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

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

    results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
    return results


def append_results_to_csv(
        results: List[Dict[str, Any]],
        output_path: str,
        header: bool,
):
    """
    Append results to CSV on disk.
    """
    if not results:
        return

    df_out = pd.DataFrame(results)
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
    """
    # --- Resume: skip rows already processed ---
    completed_ids = get_completed_ids(OUTPUT_PATH, ID_COLUMN)
    print(f"Found {len(completed_ids)} completed rows from previous runs.")

    # Filter df to only unprocessed rows
    df = df.copy()
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)  # ensure same type
    df_to_do = df[~df[ID_COLUMN].isin(completed_ids)]

    total_remaining = len(df_to_do)
    print(f"Total rows in DF: {len(df)}")
    print(f"Remaining to process: {total_remaining}")

    if total_remaining == 0:
        print("Nothing to do, all rows already processed.")
        return

    first_write = not os.path.exists(OUTPUT_PATH)

    # --- Process in chunks ---
    for start in range(0, total_remaining, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_remaining)
        df_chunk = df_to_do.iloc[start:end]
        print(f"\nProcessing rows {start}–{end} (chunk size {len(df_chunk)})...")

        start_time = time.time()
        # Run this chunk asynchronously
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

        # Append results to CSV
        append_results_to_csv(
            results=results,
            output_path=OUTPUT_PATH,
            header=first_write,
        )
        first_write = False

        end_time = time.time()
        print(f"Chunk {start}–{end} done, wrote {len(results)} rows to {OUTPUT_PATH}.")
        print(f"Time elapsed: {end_time - start_time:.4f} seconds")

    print("\nAll remaining rows processed.")


# ------------- EXAMPLE USAGE -------------
if __name__ == "__main__":
    num_batches_run = 4  # 4 uploaded successfully, 100k rows
    start_offset = 25000 * num_batches_run
    # Example: df loaded from CSV with 1.8M rows
    # You can replace this with however you construct your DataFrame.
    INPUT_CSV = "../data/raw/usda_2022_food_branded_experimental_DESCRIPTION_ONLY.csv"
    print("Loading full DataFrame...")
    df_big = pd.read_csv(INPUT_CSV, skiprows=range(1, start_offset + 1))  # dont process what was batched
    print("Dataframe size:", len(df_big))
    process_dataframe_resumable(df_big)
