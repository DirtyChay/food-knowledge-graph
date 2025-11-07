import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_last_id(CKPT):
    if CKPT.exists():
        with CKPT.open() as f:
            try:
                return json.load(f).get("last_id")
            except json.JSONDecodeError:
                return None
    return None


def set_last_id(v, CKPT):
    # ensure directory exists
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    tmp = CKPT.with_suffix(CKPT.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump({"last_id": v}, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(CKPT)


def process_food_kg_df(df, client, model="qwen/qwen3-4b-2507", batch_size=100, restart=False):
    # Setup
    CKPT = Path("checkpoints/.foodkg_checkpoint.json")
    OUT_DIR = Path("outputs/ingredients_dataset")  # directory of many part files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Pick up where left off
    last_id = get_last_id(CKPT)  # your function
    if last_id is None or restart:
        last_id = -1
    id_column = "id"  # stable and increasing identifier
    # Work only on remaining rows, sorted by id for deterministic batches
    todo = df[df[id_column] > last_id].sort_values(id_column)
    if todo.empty:
        print("Nothing to do. All caught up.")
        return

    # Split into size-limited batches
    n = len(todo)
    num_batches = int(np.ceil(n / batch_size))

    for i in range(num_batches):
        start = i * batch_size
        stop = min((i + 1) * batch_size, n)
        batch = todo.iloc[start:stop].copy()

        # Apply extractor on just this batch
        batch["ingredients_normalized"] = batch["ingredients"].apply(
            extract_ingredients,
            model=model,
            client=client,
        )

        # Write to CSV safely
        part_file = OUT_DIR / f"part_{int(batch[id_column].min())}_{int(batch[id_column].max())}.csv"
        tmp = part_file.with_suffix(part_file.suffix + ".tmp")
        batch[[id_column, "ingredients", "ingredients_normalized"]].to_csv(tmp, index=False)
        tmp.replace(part_file)

        # Advance checkpoint atomically
        set_last_id(int(batch[id_column].max()), CKPT)

        # Optional: log progress
        print(f"Wrote {part_file.name} ({start}:{stop}) — checkpoint={get_last_id(CKPT)}")

    print("Done.")


def process_branded_food_experimental_df(df, client, model="qwen/qwen3-4b-2507", batch_size=100, restart=False):
    # Setup
    CKPT = Path("checkpoints/.food_branded_experimental_checkpoint.json")
    OUT_DIR = Path("outputs/food_branded_experimental")  # directory of many part files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Pick up where left off
    last_id = get_last_id(CKPT)  # your function
    if last_id is None or restart:
        last_id = -1
    id_column = "fdc_id"  # stable and increasing identifier
    # Work only on remaining rows, sorted by id for deterministic batches
    todo = df[df[id_column] > last_id].sort_values(id_column)
    if todo.empty:
        print("Nothing to do. All caught up.")
        return

    # Split into size-limited batches
    n = len(todo)
    num_batches = int(np.ceil(n / batch_size))

    for i in range(num_batches):
        start = i * batch_size
        stop = min((i + 1) * batch_size, n)
        batch = todo.iloc[start:stop].copy()

        # Apply extractor on just this batch
        batch["mapped_ingredient"] = batch["description"].apply(
            map_to_ingredient,
            model=model,
            client=client,
        )

        # Write to CSV safely
        part_file = OUT_DIR / f"part_{int(batch[id_column].min())}_{int(batch[id_column].max())}.csv"
        tmp = part_file.with_suffix(part_file.suffix + ".tmp")
        batch[[id_column, "description", "mapped_ingredient"]].to_csv(tmp, index=False)
        tmp.replace(part_file)

        # Advance checkpoint atomically
        set_last_id(int(batch[id_column].max()), CKPT)

        # Optional: log progress
        print(f"Wrote {part_file.name} ({start}:{stop}) — checkpoint={get_last_id(CKPT)}")

    print("Done.")


def assemble_food_kg_df():
    # Later, to assemble everything:
    OUT_DIR = Path("outputs/ingredients_dataset")  # directory of many part files
    dfs = [pd.read_csv(p) for p in sorted(OUT_DIR.glob("part_*.csv"))]
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


def assemble_branded_food_experimental_df():
    # Later, to assemble everything:
    OUT_DIR = Path("outputs/food_branded_experimental")  # directory of many part files
    dfs = [pd.read_csv(p) for p in sorted(OUT_DIR.glob("part_*.csv"))]
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


# Load System Prompts
with open("prompts/system_message_ingredients.txt", "r", encoding="utf-8") as f:
    SYSTEM_MSG_INGREDIENTS = f.read()
with open("prompts/system_message_products.txt", "r", encoding="utf-8") as f:
    SYSTEM_MSG_PRODUCTS = f.read()


def _deduplicate_preserve_order(items):
    # Post processing step to remove duplicate entries
    # ["salt", "salt", "SALT"] => ["salt"]
    seen, out = set(), []
    for x in items:
        x = (x or "").strip().lower()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def map_to_ingredient(
        description,
        model="gpt-4o-mini",
        client=None,
):
    if description is None or client is None: return []  # guard for NaN
    text = str(description).strip()
    if not text: return []  # guard for empty
    max_tokens = 800
    # reinforce JSON-only on the user message
    user_msg = f"""Input product: {text}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG_PRODUCTS},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    content = response.choices[0].message.content
    # Strip whitespace, punctuation, quotes, and code fences
    parsed = (
        content.strip()
        .strip('`')
        .strip('"')
        .strip("'")
        .splitlines()[0]  # if it accidentally includes multiple lines
        .strip()
    )
    # Basic validation: must be a non-empty string of text
    if not parsed or parsed.lower() in {"unknown", "none"}:
        # Try a repair prompt once
        repair_msg = f"""Your previous response was invalid or empty.

    Return ONLY one lowercase ingredient label as plain text.
    No JSON, no quotes, no extra words.

    Input product: {text}"""

        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG_INGREDIENTS},
                {"role": "user", "content": repair_msg},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = response2.choices[0].message.content or ""
        parsed = (
            content.strip()
            .strip('`')
            .strip('"')
            .strip("'")
            .splitlines()[0]
            .strip()
        )

    # Return plain string (never NaN)
    return parsed if parsed else ""

    # try to parse; try to repair with one retry if not valid JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        repair_msg = f"""Your previous response was not valid JSON.

        Return ONLY one lowercase ingredient label as a valid JSON string.
        If none of the allowed labels clearly fit, CREATE a concise, realistic new one.
        No arrays, no explanations, no code fences.

        Input product: {text}"""
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG_INGREDIENTS},
                {"role": "user", "content": repair_msg},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = response2.choices[0].message.content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ""  # give up gracefully if still invalid

    # Normalize shape
    if isinstance(parsed, list):
        return _deduplicate_preserve_order([str(x) for x in parsed])
    # # Flatten if model returned an object (shouldn’t for single row)
    if isinstance(parsed, dict):
        flat = []
        for v in parsed.values():
            if isinstance(v, list):
                flat.extend([str(x) for x in v])
        return _deduplicate_preserve_order(flat)
    return ""


def extract_ingredients(
        description,
        model="gpt-4o-mini",
        client=None,
):
    if description is None or client is None: return []  # guard for NaN
    text = str(description).strip()
    if not text: return []  # guard for empty
    max_tokens = 800
    # reinforce JSON-only on the user message
    user_msg = f"""Input: {text}
Return ONLY a valid JSON array of lowercase ingredient names, no explanations, no code fences.
Example: ["chicken","butter"]"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG_INGREDIENTS},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    content = response.choices[0].message.content
    # try to parse; try to repair with one retry if not valid JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        repair_msg = f"""Your previous response was not valid JSON.
Return ONLY a valid JSON array of strings, no explanations, no code fences.
Example: ["chicken","butter"]

Input: {text}"""
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG_INGREDIENTS},
                {"role": "user", "content": repair_msg},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = response2.choices[0].message.content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return []  # give up gracefully if still invalid

    # Normalize shape
    if isinstance(parsed, list):
        return _deduplicate_preserve_order([str(x) for x in parsed])
    # # Flatten if model returned an object (shouldn’t for single row)
    if isinstance(parsed, dict):
        flat = []
        for v in parsed.values():
            if isinstance(v, list):
                flat.extend([str(x) for x in v])
        return _deduplicate_preserve_order(flat)
    return []

# def get_food_kg_from_engine(engine, batch_size=500, process_chunk=lambda df: print("No Processing Function Provided")):
#     SQL = text("""
#                SELECT *
#                FROM "FoodKG"
#                WHERE (:last_id IS NULL OR id > :last_id)
#                ORDER BY id LIMIT :lim
#                """)
#
#     last_id = get_last_id()
#     with engine.connect().execution_options(stream_results=True) as conn:
#         try:
#             while True:
#                 df = pd.read_sql_query(SQL, conn, params={"last_id": last_id, "lim": batch_size})
#                 if df.empty:
#                     break
#                 process_chunk(df)  # may raise
#                 last_id = int(df["id"].iloc[-1])
#                 set_last_id(last_id)
#         except Exception as e:
#             print(f"Error after id {last_id}: {e}")
#             raise  # or log and continue
