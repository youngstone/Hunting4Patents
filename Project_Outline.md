# Models

1. Search Engine
	- Natural Language Processing
	- TF-IDF
	- Text similarity

2. Recommender
	- Ranking by PageRank
		* all-time citation 

3. Extract feature to predict the longevity of a patent (life expectancy)
	- Features: semantic analysis, owner, law firm, post issuance cost (citation)  category, litigation, transaction records
		* Random Forrest
		* Logistic regression
		* SVM, SVD


# Workflow

Phase 1: get data
=======================================

1) filename:  get_data_core.py
```
	INPUT: None
	OUTPUT: Patent database file -> ./my_database/database_core
	POINTS TO: combine_my_data.py
```
Purpose: download all patent data from patent topics "Drugs / Vasodialators / Gene Therapy / Other Drug Related" from webstie: 'freepatentsonline'
How: using bs4 + requests, or import.io
Go to the pages that have all of the industry patents, get all the patent numbers.
then go to all the individual patent pages. from that page scrape the 'filling date', 'primary classes', 'other classes', 'US patent references', 'Attorney, Agent or Firm', 'link', 'title', 'abstract', 'claim', 'description'
then store all that iinformation into database 'database_core'

*2) filename:  get_data_maintenance.py
```
INPUT: maintenancefee.txt
	OUTPUT: Patent maintenance database file -> ./my_database/database_maintenance
	POINTS TO: combine_my_data.py
```
Purpose: parse maintenancefee.txt, get maintenance action records for each patent
then store all that information into database 'database_maintenance'

*3) filename:  get_data_assignment.py
```
	INPUT: None
	OUTPUT: Patent assignment database file -> ./my_database/database_assignment
	POINTS TO: cobine_my_data.py
```
Purpose: download all patent assignment data of individual patent from webstie: 'http://assignment.uspto.gov/'
then store all that iinformation into database 'database_assignment'


Phase 2: combine my database
=======================================
filename: combine_my_database.py
```
	INPUT: None
	OUTPUT: Patent assignment database file -> ./my_database/full_database
	POINTS TO: get_query.py, [*populate_features.py, *calc_life_and_cost.py]
```
output file: _database.py
type: sql or *mongo
why:  store patent info given from get_data_core.py + get_data_assignment + get_data_maintenance.py
how: use psycopg to create new database called full_database.db.
	 take data from database_core + database_assignment + database_maintenance, and insert into full_database.db.


Phase 3: populate data
=======================================
1) filename: get_reference_data.py
```
	INPUT: full_database
	OUTPUT: reference relation database -> ./my_database/citation_database
	POINTS TO: get_reference_relations.py
```
why: store patent citation data
how: use pandas to create new database called citation_database, which includes patent# and patent citations.

*2) filename: populate_features.py
```
	INPUT: full_database
	OUTPUT: reference relation database -> ./my_database/features_database
	POINTS TO: build_model.py
```

*3) filename: calc_life_and_cost.py
```
	INPUT: full_database
	OUTPUT: reference relation database -> ./my_database/life_cost_database
	POINTS TO: build_regression_model.py
```


Phase 4: get working data
=======================================
1) filename: get_reference_relations.py
```
	INPUT: citation_database
	OUTPUT: citation relation file -> ./my_data/citation.csv
	POINTS TO: calc_pagerank.py
```
why: get rows of one-to-one citation relation
how: use pandas pivot_table 


Phase 5: build model
=======================================
1) filename: calc_pagerank.py
```
	INPUT: citation_database.csv
	OUTPUT: table_pagerank
	POINTS TO: viz_pagerank.py
```

why: calculate pagerank
how: use graphlab to calculate

*2) filename: patent_life_predictor.py
```
	INPUT: 
	OUTPUT: 
	POINTS TO: 
```

*3) filename: patent_matcher.py
```
	INPUT: 
	OUTPUT: 
	POINTS TO: 
```

Phase 6: visualization
=======================================
```
	INPUT: table_pagerank, citation.csv
	OUTPUT: table, chart
	POINTS TO: web app
```

What: graph, table, map
How: use d3.js, plot.ly


Phase7: web app
=======================================
```
	INPUT: 
	OUTPUT: 
```

# Code Structure

```
-- CAPSTONE PROJECT/
|	|-- CODE/
|	|	|-- get_data_core.py => to scrape data and store (main data, title, text, etc)
|	|	|-- get_data_maintenance.py => to import data (maintenance fee)
|	|	|-- get_data_transaction.py => to import transaction records
|	|	|-- -- feature_extraction.py
|	|	|-- -- calculate_expiration_date.py
|	|	|-- -- -- life_events.py
|	|	|-- -- -- citation_flow.py
|	|	|-- -- -- citation_graph.gephi
|	|	|-- -- -- similarity.graphlab
|	|	|-- -- -- -- data_viz.py
|	|	|-- -- -- -- web_app.py
|	|-- DATA/
|	|	|-- maintenance.txt --> maintenance.db
|	|	|-- scape --> patent_info.db
|	|	|-- scape --> patent_info.db
|	|	|-- scape --> patent_info.db
```









