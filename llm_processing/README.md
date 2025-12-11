### Prerequisites

Before running these notebooks, ensure that the following CSV exists. It can be obtained from the Google Drive
submission folder or by completing the spaCy steps.

- data/output/foodkg_spacy_processed.csv

Run `retrieve_food_kg.ipynb` and `retrieve_usda_branded_experimental.ipynb` to download the necessary CSVs.

- data/raw/nourish_public_FoodKG.csv
- data/raw/usda_2022_food_branded_experimental_DESCRIPTION_ONLY

### LLM Processing

All prompts used by these notebooks and scripts are stored in the **prompts** folder.

This folder contains the code used to perform LLM-based processing on the data.  
These notebooks typically rely on local CSVs retrieved from the data server.

If you prefer not to use local files, refer to the `spacy_processor` script in the **SpacyProcessing** section for an
example of retrieving data in batches directly from the server.

---

### Retrieval

#### `retrieve_food_kg`

Retrieves `nourish_public_FoodKG` from the SQL server and downloads it locally as a CSV for convenient reuse.

#### `retrieve_usda_branded_experimental`

Retrieves `usda_2022_food_branded_experimental` from the SQL server and downloads it locally as a CSV.  
Only the necessary columns are retained for future processing.

---

### Preprocessing

#### `spacy_unique_ingredients`

Identifies unique ingredients from spaCy-processed data, counts occurrences, and determines a cutoff that filters the
dataset to roughly 100 ingredients.  
These most common ingredients are saved to `spacy_unique_ingredients.txt` and appended to the LLM system prompt in
`process_products.py`.

---

### Processing

#### `process_products.py`

A Python script that processes dataset rows asynchronously using the OpenAI GPT model with full resumability.

Features include:

- Chunked, concurrent asynchronous API requests
- Exponential backoff on retries
- Concurrency limits
- Automatic persistence of completed rows to disk
- Resume capability to avoid reprocessing already-completed rows

---

### Testing

#### `test_products`

Evaluates the model’s performance using accuracy metrics and Levenshtein similarity against the test dataset.

#### `test_food_kg`

A legacy notebook for testing LLM-based FoodKG results when spaCy is not used.  
Not actively supported due to resource constraints but included for completeness.
