# NEO4J Graph Creation

Files included
-Indecies.cypher => contains cypher queries to create indicies in our graph db
-NEO4Jwriter.ipynb => Notebook file which contains code to clean output CSVs from spacy/llm preprocessing step and create the graph
-Neo4Jwriter.py => End 2 end script to create the entire graph assuming cleaned ouput CSV files are already obtained from NEO4Jwriter.ipynb. Provided just for convenience. Same code can be executed in the notebook.
-requirements.txt => required libs to install

Other Required files (large files saved in persistant storage in NDP)
-"foodkg_spacy_processed.csv" => output file from spacy preprocessing. If cleaned csv (see below) is already available, this is not required. Only needed if running from scratch.
-"foodkg_spacy_processed_cleaned.csv" => generated in NEO4Jwriter.ipynb from "foodkg_spacy_processed.csv"
-"mapped_responses.csv" => output file from llm preprocessing. If cleaned csv (see below) is already available, this is not required. Only needed if running from scratch.
-"brian_results_async_3.csv" => generated in NEO4Jwriter.ipynb from "mapped_responses.csv"


Instructions:
1) Run the cypher queries in Indecies.cypher to create all the indicies for our graph. **This is already done so this can be skipped. Only need to do this if starting from a blank graph**

- Neo4J can be accessed through web using http://awesome-compute.sdsc.edu:7474/browser/
- Connection info:

| Deployment                | Username | Password | Bolt URI                    |
|---------------------------|----------|----------|-----------------------------|
| neo4j-db-group-1-fall-2025 | neo4j    | x7t4p2ks | bolt://67.58.49.84:7687     |


- After connecting using above credentials run each of the Create idx statements in the GUI

2) Make sure to install all the dependencies found in requirements.txt if running locally
3) Copy required output files to same path as NEO4Jwriter.ipynb ("brian_results_async_3.csv", "mapped_responses.csv", "foodkg_spacy_processed_cleaned.csv", "foodkg_spacy_processed.csv") These files are submitted to google drive folder - link: https://drive.google.com/drive/folders/1PjYl_mEVLIxk82QhK-Zt0PHOHFaXvZBc?usp=share_link 
4) If cleaned csvs ( "foodkg_spacy_processed_cleaned.csv" and "brian_results_async_3.csv") are already available move to step 5. Otherwise, if running from scratch, scroll to the bottom of the notebook and run the code under sections "Process the Recipe Ingredients CSV" and "Process the products-ingredients mapping csv to rename cols and drop null vals" to generate the final cleaned csvs
5) Run the cells from top to bottom to create the graph. Alternatively use the Neo4Jwriter.py script from the same location as the generated csvs mentioned in step 4.
