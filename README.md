# Hunting4Patents

Hunting4Patents is a tool that finds patents that are valuable but likely to expire early.

I built a custom web scraper to get patent data from several patent websites and build a clean database. To begin with the data, I built a patent search engine based on content similarity comparison to response user's query. I calculated the pagerank for each patent based on all-time citations. I built a Random Forest model to predict early expiration by utilizing patent features and early life events. 

[Live Web App](http://ec2-52-10-83-141.us-west-2.compute.amazonaws.com/) (Now available for pharmaceutical patents search)

[Project Proposal](Preliminary_Project_Proposal.md)

<a target="_blank"><img src="/results/citation_network.png" 
alt="Citation network of a collection of patents" align='middle' width="400" border="10" /></a>

![png](results/citation_network.png)
(citation network of a collection of patents)

# Incentive

We are in a world of inventions. Making inventions and making use of the inventions are key to a company's success.

US patent law grants the patent owner exclusive rights of the invention for 20 years. By granting the right to produce a new product without fear of competition, patents provide incentive for companies or individuals to continue developing innovative new products or services. 

One example is that pharmaceutical companies spend large sums on research and development and patents are essential to earning a profit.

On ther other hand, if you own a business and you want to make use of others' patents for your business, you may want to find a way to identify those patents that (1) are related to your business, (2) valuable, for example in terms of popularity in the field, and (3) are likely to expire soon.

Therefore, the goal of this project to build a tool that calculates the metrics for these 3 needs and makes the best recommendations.


# Data

I scraped and downloaded patent data from the following website:

* [Google Patent Search](http://www.google.com/patents)
* [Freepatentonline](http://www.freepatentsonline.com)
* [USPTO Patent Assignment Search](http://assignment.uspto.gov) 
* [USPTO Patent Maintenance Fee Events](https://eipweb.uspto.gov/MaintFeeEvents/)
* [Harvard Patent Network Dataverse](https://thedata.harvard.edu/dvn/dv/patent/faces/study/StudyPage.xhtml?globalId=hdl:1902.1/12367&studyListingIndex=0_b547d55c3b44eda0c6f7707020be)

The initial implementation collected 2465 patents in the field of pharmaceutical industry. 

# Features

Features are the useful properties underlying the raw data. I extracted the following features to build my models

* Patent text content
	* --> to be converted to tf-idf vectors

* Patent citation
	* --> to be modeled as connections between patents

* Bibliographical information, maintenance events, etc.
	* --> to be utilized for feature engineering


# Models

The goal is to find 3 metrics of RELEVANCY, VALUE, WHEN TO EXPIRE, so I built a model for each aspect.

1. Search Engine
	- Tool: Natural Language Processing
	- Features: tf-idf vectors
	- How: calculate similarity score weighted by title, abstract, and claims, and return the patents with highest similarity

2. Ranking
	- Tool: network and PageRank
	- Features: all-time citations
	- How: get 1 level depth forward citation for each patent, then calculate the PageRank by either using graphlab package or solving eigen-problem of the transition matrix

3. Predictor of Early Expiration
	- Tool: feature engineering
	- Features: backward patent citations, backward non-patent citaitons, ratio of backward citations made by inventor to made by patent examiner, semantic analysis, post issuance records
	- How: convert features into numerics and build a Random Forrest Classifier with sklearn. Train the model with already expired patent data (early expiration and natural expiration). Use GridSearch to find the best estimator. Then make predictions for current live patents.


# Product

[Live Web App](http://ec2-52-10-83-141.us-west-2.compute.amazonaws.com/) (Now available for pharmaceutical patents search)


# Workflow

Phase 1: get data
=======================================

1) filename:  get_data_patent_content.py
```
	INPUT: None
	OUTPUT: MongoDB database file -> ./database/patent_database.patent_fields
	POINTS TO: combine_my_data.py
```
Purpose: download all patent data from patent topics "Drugs / Vasodialators / Gene Therapy / Other Drug Related" from webstie: 'freepatentsonline'
How: using bs4 + requests
Go to the pages that have all of the industry patents, get all the patent numbers.
then go to all the individual patent pages. from that page scrape the 'filling date', 'primary classes', 'other classes', 'US patent references', 'Attorney, Agent or Firm', 'link', 'title', 'abstract', 'claims', 'description'
then store all that iinformation into database 'patent_database.patent_fields'

2) filename:  get_data_maintenance.py
```
	INPUT: maintenancefee.txt
	OUTPUT: Patent maintenance database file -> ./my_database/database_maintenance.sqlite3
	POINTS TO: combine_my_data.py
```
Purpose: parse maintenancefee.txt, get maintenance action records for each patent
then store all that information into database 'database_maintenance.sqlite3'

3) filename:  get_data_assignment.py
```
	INPUT: None
	OUTPUT: Patent assignment database file -> ./my_database/database_assignment
	POINTS TO: combine_my_data.py
```
Purpose: download patent assignment data of individual patent from webstie: 'http://assignment.uspto.gov/'
then store all that iinformation into database 'database_assignment'


Phase 2: combine my database
=======================================
filename: combine_my_database.py
```
	INPUT: None
	OUTPUT: Patent assignment database file -> ./my_database/full_database
	POINTS TO: get_query.py, [*populate_features.py, *calc_life_and_cost.py]
```
output file: patent_database
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

2) filename: populate_features.py
```
	INPUT: full_database
	OUTPUT: reference relation database -> ./my_database/features_database
	POINTS TO: build_model.py
```

3) filename: calc_life_and_cost.py
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
Model 1: pagerank
---------------------------------------

1) filename: calc_pagerank.py
```
	INPUT: citation_database.csv
	OUTPUT: table_pagerank
	POINTS TO: viz_pagerank.py
```

why: calculate pagerank
how: use graphlab to calculate


Model 2: early expiration predictor
---------------------------------------
2) filename: patent_life_predictor.py
```
	INPUT: 
	OUTPUT: 
	POINTS TO: 
```

Model 3: similarity
---------------------------------------
3) filename: patent_matcher.py
```
	INPUT: 
	OUTPUT: 
	POINTS TO: 
```

Phase 6: visualization
=======================================
```
	INPUT: patent_dataframe.pkl
	OUTPUT: table, chart
	POINTS TO: web app
```

What: graph, table
How: use plot.ly


Phase7: web app
=======================================
```
	INPUT: 
	OUTPUT: 
```

# Code Structure

```
-- HUNTING4PATENTS/
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
|	|-- APP/
|	|	|-- app.py
|	|	|-- my_plot_plotly.py
|	|	|-- patent_matcher.py (copied from ../CODE/)
|	|	|-- patent_tokenizer.py (copied from ../CODE/)
|	|-- DATA/
|	|	|-- maintenance.txt
|	|	|-- 
|	|	|-- 
|	|	|-- 
```

# Result

![png](results/chart_selected_patents.png)





