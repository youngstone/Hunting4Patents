# Capstone Project Proposal

Zidong Yang (Stone Young)

02/23/2014

## Hunting for Patents

### Background
Business success of companies relies on their intellectual properties. There are three domains of intellectual properties in general: 1) protected intellectual properties, including patents, trade mark, and copyrights, 2) public knowledge, including open source, 3) trade secret.

Since the public knowledge is not protectable and the trade secret is not known to public. Protection of patents, trade mark and copyrights are critical. 

Article One, section 8, clause 8 of the U.S. Constitution states:

```
The Congress shall have power ... To promote the progress of science and useful arts, by securing for limited times to authors and inventors the exclusive right to their respective writings and discoveries;
```

There are two aspects of the US patent law:

1) Exclusive protection of patents are given to the patent owner, thus innovations are fostered.

2) In exchange to the protect, patents are disclosed and the protection is only for a limited time. Therefore, other people are allowed to practice the patent once it becomes expired.

### Idea
Due to the unique role of patent law, it may interest companies to know the ecosystem of the patents that are related to their business and to include this knowledge when they make strategic decisions.

In this project, it is proposed to build an automated system that shows the landscape of the patents of interest and gives search and recommendation service based on user queries. The project will mainly use the data of full-text, transaction record, and maintenance fee record, which are all publicly available. 

In addition, the project attempts to gather information about the licensing (authorizing other parties to use) record, the amount of money involved in the patent transaction and litigation, which are usually unavailable.

### Goals
* To answer exploratory questions: overview of a patent and the landscape of patents
	* What are the events during the life of a particular patent?
		* Maintenance fee payment
		* Infringe/litigation
		* Transaction/assignment
		* Cited by other patents
	* What are the average stats of patents?
		* How long is the lifetime (from the grant day to either the expiration day or no more payment of maintenace fee)
		* How much maintenance fee are paid
	* How is the citation flow around some topic/patent?
		* Number of citations
		* PageRank citation ranking

* Interesting questions: potentially actonable decisions
	* How does a patent impact on other patents and the patent landscape? 
		* Is our model able to identify a keystone patent?
		* If this keystone patent becomes expired, what can we gain from it?
	* How does patents landscape change over the years? What's the current trend?
	* Recommender: If we (big company) want to acquire patents/small companies that own the patent, what patents have most (potential) business value? 
	* Search engine: if we (inventor) have prepared a new patent application, what are the similar patents? What prior arts are there? 

* Work flow
	1. Gather information
	2. Clean data
	3. Aggregate data
	4. Exploratory data analysis
	5. Build a classification model to represent patents, validate the model by looking into the sensitivity, roc curve
	6. Build other functions that user may be interested in


### Deliverable
* Web App (An interactive search engine for exploring relevant patents and recommending strategic actions)
* Slides/Presentation (An overall presentation of the value, achievement of the project)
* Technical Report (A more detailed report about the methodology)

### Proposed techniques
* Web scraping: to bulk download patent data
* Data cleaning: to preprocess and transform the raw data
* Database: to aggregate various datasets
* Natural Language Processing: to convert full-text into numerics and build models
* Classification: to demonstrate the capability of our model to identify the similarity and dissimilarity of patents
* Recommender system: to search for target patents according to user's interests
* Data visualization: to demonstrate the results
* Spark: for scalable data analysis

### Dataset
* Features: 
	* Patent granted date
	* Patent expiring date
	* Patent owner/assignee
	* Patent category
	* Patent content
	* Patent claims: 1) content, 2) category (method/matter)
	* Patent litigation record (if found)
* Source: 
	* [Google Patent - USPTO Bulk Downloads: PAIR Data](http://www.google.com/googlebooks/uspto-patents-pair.html)
	* [USPTO Bulk Downloads: Patent Grant Full Text](http://www.google.com/googlebooks/uspto-patents-grants-text.html)
	* [USPTO Patent Assignment Search](http://assignment.uspto.gov) 
	* [USPTO Patent Maintenance Fee Events](https://eipweb.uspto.gov/MaintFeeEvents/)
	* [USPTO General Patent Statistics Reports](http://www.uspto.gov/web/offices/ac/ido/oeip/taf/reports.htm)
	* [Freepatentonline](http://www.freepatentsonline.com)
	* [Google Patent Search API (Deprecated)](https://developers.google.com/patent-search/)

### Reference
* [USPTO](http://www.uspto.gov/patent)
* [Wikipedia: U.S. Patent Law](http://en.wikipedia.org/wiki/United_States_patent_law)
