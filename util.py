import json
import os

import pandas as pd
from sqlalchemy import text

TABLE = '"FoodKG"'  # quoted because of uppercase name
ID_COL = "id"
CHECKPOINT_PATH = "checkpoints/.foodkg_ckpt.json"


def _load_checkpoint(path=CHECKPOINT_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"last_id": None, "watermark": None}


def _save_checkpoint(state, path=CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)


def process_foodkg_in_batches(
        engine,
        batch_size=500,
        columns=("id", "title", "ingredients", "directions", "link", "source", "ner"),
        process_fn=lambda df: None,
        CUTOFF=None  # testing purposes only, stop when reach this point
):
    """
    Streams rows from FoodKG in ascending id order up to a snapshot watermark.
    Persists checkpoint to resume after interruptions.
    """
    state = _load_checkpoint()
    last_id = state.get("last_id")
    watermark = state.get("watermark")

    # Take a snapshot watermark for consistency across the whole run
    if watermark is None:
        with engine.begin() as conn:
            watermark = conn.execute(
                text(f"SELECT max({ID_COL}) FROM {TABLE}")
            ).scalar()
        _save_checkpoint({"last_id": last_id, "watermark": int(watermark) if watermark is not None else None})

    cols_sql = ", ".join(columns)
    base_sql = f"""
        SELECT {cols_sql}
        FROM {TABLE}
        WHERE (:last_id IS NULL OR {ID_COL} > :last_id)
          AND (:watermark IS NULL OR {ID_COL} <= :watermark)
        ORDER BY {ID_COL} ASC
    """

    total_processed = 0
    with engine.begin() as conn:
        for chunk in pd.read_sql_query(
                text(base_sql),
                conn,
                params={"last_id": last_id, "watermark": watermark},
                chunksize=batch_size,
        ):
            if chunk.empty:
                break
            # process batch
            process_fn(chunk)
            # advance checkpoint
            last_id = int(chunk[ID_COL].iloc[-1])
            _save_checkpoint({"last_id": last_id, "watermark": int(watermark) if watermark is not None else None})

            total_processed += len(chunk)
            if CUTOFF is not None and total_processed >= CUTOFF:
                print("Processed {} rows. Cutting off".format(total_processed))
                break

    # Clear the watermark when done so next run snapshots fresh rows
    _save_checkpoint({"last_id": last_id, "watermark": None})
    print("✅ Done up to id:", last_id)


def my_processing(df: pd.DataFrame):
    print("processing hoho")
