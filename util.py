import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

CKPT = Path("checkpoints/.foodkg_checkpoint.json")


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
