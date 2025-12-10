### Batch Processing Code

This folder contains the code required to send and retrieve batch files using the OpenAI API.

### `create_batches`

This script sends batches of rows to OpenAI.  
You may enqueue up to **40 million tokens at a time**, which is roughly **50,000 rows per 24 hours**.  
Batch IDs are stored in `batch_tracker.json`.

Use this script if you want to reduce cost at the expense of processing time.  
You are responsible for merging the batch output with the results of the asynchronous calls using the standard OpenAI
API.

### `read_batches`

This script retrieves the batches listed in `batch_tracker.json` and aggregates them into a single CSV file.
