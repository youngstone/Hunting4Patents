# python code

# filename:  get_data_patent_content.py
# '''
#     INPUT: None
#     OUTPUT: Patent database file -> ./my_database/patent_database
#     POINTS TO: clean_my_data.py
# '''
# Purpose: download all patent data from patent topics "Drugs"
#          from webstie: 'freepatentsonline'
# How:     using bs4 + requests
#          Go to the pages that have all of the industry patents,
#          get all the patent numbers.
#          then go to all the individual patent pages.
#          From that page scrape the 'filling date',
#                                     'primary classes',
#                                     'other classes',
#                                     'US patent references',
#                                     'Attorney, Agent or Firm',
#                                     'link',
#                                     'title',
#                                     'abstract',
#                                     'claim',
#                                     'description'
#          Then store all that iinformation into database 'database_core'

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
import pandas as pd
import pickle


def main():
    # scrape patent numbers and links
    patent_numbers, patent_links = get_patent_number()

    '''
    # scrape patent html content
    get_patent_html_content(patent_numbers, patent_links)
    '''

    # scrape patent content and store in fields
    get_patent_content(patent_numbers, patent_links)


def get_patent_category():
    '''
    reserved for future use
    '''
    url_main = 'http://www.freepatentsonline.com'


def get_patent_number():
    '''
    INPUT: None
    OUTPUT: list: patent_numbers, list: patent_links
    '''
    site_urls = []  # to store list of urls of pages of search result
    num_page = 50  # depends on the number of patents in this category
    for i in xrange(1, num_page + 1):
        url = 'http://www.freepatentsonline.com/CCL-424-185.1-p%s.html' \
              % str(i)
        site_urls.append(url)

    patent_numbers = []  # to store patent numbers
    patent_links = []  # to store patent links

    # for each page of search result, get all the patents
    for url in site_urls:
        response = requests.get(url)
        if response.status_code == 200:
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.select('div table.listing_table tr td a')

            # get all patents in the page
            pat_links = [item.get('href') for item in content]
            pat_numbers = [item[1: -5] for item in pat_links]

            # for each patent, store its number and link
            for i in xrange(len(pat_numbers)):
                link = pat_links[i]
                number = pat_numbers[i]
                patent_links.append(link)
                patent_numbers.append(number)

    print '\nTotal number of collected patents:', len(patent_numbers)
    print '\nTotal number of links to them:', len(patent_links)
    print ''

    '''
    # to store patent numbers in txt file

    with open('../data/patent_numbers.txt', 'w') as f:
        for line in patent_numbers:
            f.write(line + '\n')
    '''

    return (patent_numbers, patent_links)


def get_patent_html_content(patent_numbers, patent_links):
    '''
    INPUT: patent_numbers, patent_links
    OUTPUT: mongodb database: patent_database, collection: patent_html
    '''

    # launch mongodb
    client = MongoClient('mongodb://localhost:27017/')
    db = client.patent_database
    collection = db.patent_html

    patent_records = []

    for num, link in enumerate(patent_links):

        url = 'http://www.freepatentsonline.com' + link

        response = requests.get(url)
        if response.status_code == 200:
            html = response.content

            pat_num = patent_numbers[num]
            row = [{"num": pat_num,
                    "url": url,
                    "html": html}]
            try:
                collection.insert(row)
            except DuplicateKeyError:
                pass

            patent_records.append([pat_num, url, html])

    print "Total number of patents added to database"
    print collection.find().count()
    print "One example"
    print collection.find_one()
    print len(patent_records)


def get_patent_content(patent_numbers, patent_links):
    '''
    INPUT: patent_numbers, patent_links
    OUTPUT: mongodb database: patent_database, collection: patent_fields
    '''

    # connect to mongoclient
    client = MongoClient('mongodb://localhost:27017/')
    db = client.patent_database
    collection = db.patent_fields

    all_patent = {}

    for num, link in enumerate(patent_links):
        print num
        url = 'http://www.freepatentsonline.com' + link
        # print url

        raw_dict = {}
        response = requests.get(url)
        if response.status_code == 200:
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.select('div table tr td a')
            cite_links = [item.get('href') for item in content]
            keys = []
            vals = []
            for element in soup.find_all('div', class_='disp_doc2'):
                elm_title_tag = element.find('div', class_='disp_elm_title')
                if elm_title_tag:
                    elm_title = elm_title_tag.get_text().strip().strip(':')
                    elm_text_tag = element.find('div', class_='disp_elm_text')
                if elm_text_tag:
                    elm_text = elm_text_tag.get_text().strip()
                if elm_title and elm_text:
                    raw_dict[elm_title] = raw_dict.get(elm_title, elm_text)

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

            patent_content[u'Number'] = patent_numbers[num]
            patent_content[u'Title'] = raw_dict.get(u'Title', None)
            patent_content[u'Abstract'] = raw_dict.get(u'Abstract', None)
            patent_content[u'Filing Date'] = \
                raw_dict.get(u'Filing Date', None)
            patent_content[u'Primary Class'] = \
                raw_dict.get(u'Primary Class', None)
            patent_content[u'Other Classes'] = \
                raw_dict.get(u'Other Classes', None)
            patent_content[u'US Patent References'] = cite_links
            patent_content[u'Claims'] = raw_dict.get(u'Claims', None)
            # patent_content['Description'] = raw_dict.get('Description', None)

            all_patent[patent_numbers[num]] = patent_content
            try:
                collection.insert(patent_content)
            except DuplicateKeyError:
                pass
            # break

    print collection.find().count()
    print collection.find_one()

    return all_patent


if __name__ == '__main__':
    main()
