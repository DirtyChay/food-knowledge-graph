import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()  # Take environment variables from .env
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

client = OpenAI(
    api_key=api_key
)

CKPT = Path("checkpoints/.foodkg_checkpoint.json")

# Load System Prompt
with open("../prompts/system_message_ingredients.txt", "r") as f:
    SYSTEM_MSG_INGREDIENTS = f.read()


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


def extract_ingredients(description):
    if description is None: return []  # guard for NaN
    text = str(description).strip()
    if not text: return []  # guard for empty
    # reinforce JSON-only on the user message
    user_msg = f"""Input: {text}
Return ONLY a valid JSON array of lowercase ingredient names, no explanations, no code fences.
Example: ["chicken","butter"]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_MSG_INGREDIENTS},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=600,
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MSG_INGREDIENTS},
                {"role": "user", "content": repair_msg},
            ],
            max_tokens=600,
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


def get_last_id():
    if CKPT.exists():
        with CKPT.open() as f:
            try:
                return json.load(f).get("last_id")
            except json.JSONDecodeError:
                return None
    return None


def set_last_id(v):
    # ensure directory exists
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    tmp = CKPT.with_suffix(CKPT.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump({"last_id": v}, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(CKPT)


SQL = text("""
           SELECT *
           FROM "FoodKG"
           WHERE (:last_id IS NULL OR id > :last_id)
           ORDER BY id LIMIT :lim
           """)


def get_food_kg(engine, batch_size=500, process_chunk=lambda df: print("No Processing Function Provided")):
    last_id = get_last_id()
    with engine.connect().execution_options(stream_results=True) as conn:
        try:
            while True:
                df = pd.read_sql_query(SQL, conn, params={"last_id": last_id, "lim": batch_size})
                if df.empty:
                    break
                process_chunk(df)  # may raise
                last_id = int(df["id"].iloc[-1])
                set_last_id(last_id)
        except Exception as e:
            print(f"Error after id {last_id}: {e}")
            raise  # or log and continue
