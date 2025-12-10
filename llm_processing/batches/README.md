### Batch Processing Code

This folder contains the code required to send and retrieve batch files using the OpenAI API.

Only use these scripts if you want to reduce cost at the expense of processing time.  
You are responsible for merging the batch output with the results of the asynchronous calls that use the standard OpenAI
API.

### `create_batches`

This script sends batches of rows to OpenAI.  
You may enqueue up to **40 million tokens at a time**, which is roughly **50,000 rows per 24 hours**.  
Batch IDs are stored in `batch_tracker.json`.

### `read_batches`

This script retrieves the batches listed in `batch_tracker.json` and aggregates them into a single CSV file.
