# python code

# filename:  get_data_core.py
# ''' INPUT: None
# 	OUTPUT: Patent database file -> ./my_database/database_core
# 	POINTS TO: combine_my_data.py
# '''
# Purpose: download all patent data from patent topics "Drugs / Vasodialators / Gene Therapy / Other Drug Related" from webstie: 'freepatentsonline'
# How: using bs4 + requests, or import.io
# Go to the pages that have all of the industry patents, get all the patent numbers.
# then go to all the individual patent pages. from that page scrape the 'filling date', 'primary classes', 'other classes', 'US patent references', 'Attorney, Agent or Firm', 'link', 'title', 'abstract', 'claim', 'description'
# then store all that iinformation into database 'database_core'

import requests
import bs4
import json
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
from bs4 import BeautifulSoup
import re
from dateutil.parser import parse
from bson.son import SON

def main():

	patent_numbers, patent_links = get_patent_number()

	all_patents = get_patent_content(patent_numbers, patent_links)

	# json.dumps(all_patents, ensure_ascii=False)


	# fp = '../data/database.json'
	# with open(fp, 'w') as f:
	# 	json.dump(json.dumps(all_patents, ensure_ascii=False), f)

def get_all_field():
	url_main = 'http://www.freepatentsonline.com'

def get_patent_number():
	site_urls = []
	num_page = 50
	# num_page = 1
	for i in xrange(1, num_page + 1):
		url = 'http://www.freepatentsonline.com/CCL-424-185.1-p%s.html' % str(i)
		site_urls.append(url)

	patent_numbers = []
	patent_links = []

	for url in site_urls:
		response = requests.get(url)
		if response.status_code == 200:
			html = response.content
			soup = BeautifulSoup(html, 'html.parser')
			content = soup.select('div table.listing_table tr td a')
			pat_links = [item.get('href') for item in content]
			pat_numbers = [item[1:-5] for item in pat_links]

			for i in xrange(len(pat_numbers)):
				link = pat_links[i]
				number = pat_numbers[i]
				patent_links.append(link)
				patent_numbers.append(number)

	print '\nTotal number of collected patents:', len(patent_numbers)
	print '\nTotal number of links to them:', len(patent_links)
	print ''

	return (patent_numbers, patent_links)

def get_patent_content(patent_numbers, patent_links):
	
	client = MongoClient('mongodb://localhost:27017/')
	db = client.patent_database
	collection = db.main_content

	all_patent = {}

	for num, link in enumerate(patent_links):

		url = 'http://www.freepatentsonline.com' + link
		# print url

		response = requests.get(url)
		if response.status_code == 200:
			html = response.content
			soup = BeautifulSoup(html, 'html.parser')
			content = soup.select('div table tr td a')
			cite_links = [item.get('href') for item in content]

			keys = []
			vals = []

			for elm_title in soup.find_all('div', class_='disp_elm_title'):
				keys.append( elm_title.get_text()[:-1] )

			for elm_text in soup.find_all('div', class_='disp_elm_text'):
				vals.append( elm_text.get_text().strip() )

			raw_dict = {k: v for k, v in zip(keys, vals)}

			
			'''
			Title
			Abstract
			Filing Date
			Primary Class
			Other Classes
			US Patent References
			Claims
			Description
			'''
			patent_content = {}

			patent_content['Number'] = patent_numbers[num]
			patent_content['Title'] = raw_dict.get('Title', None)
			patent_content['Abstract'] = raw_dict.get('Abstract', None)

			# date = raw_dict.get('Filing Date', None)
			# date = parse(date)
			# print date
			# date = date.strftime('%m/%d/%Y')
			# print date
			# patent_content['Filing Date'] = date

			# patent_content['Filing Date'] = raw_dict.get('Filing Date', None)
			patent_content['Primary Class'] = raw_dict.get('Primary Class', None)
			patent_content['Other Classes'] = raw_dict.get('Other Classes', None)
			patent_content['US Patent References'] = cite_links
			patent_content['Claims'] = raw_dict.get('Claims', None)
			patent_content['Description'] = raw_dict.get('Description', None)

			# for k, v in patent_content.iteritems():
			# 	print k
			# 	print "---"
			# 	print v
			# 	print "==="
			
			all_patent[patent_numbers[num]] = patent_content
			try:
				collection.insert(patent_content)
			except DuplicateKeyError:
				pass

	print collection.find().count()
	print collection.find_one()
	# print len(all_patent)
	# print type(all_patent[patent_numbers[0]])
	return all_patent


if __name__ == '__main__':
	main()