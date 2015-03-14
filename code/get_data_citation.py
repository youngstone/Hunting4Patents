# python code

# filename:  get_data_citation.py
# '''
#     INPUT: None
#     OUTPUT: Patent citation database file -> ./my_database/patent_database
#     POINTS TO: clean_my_data.py
# '''

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

    # scrape patent citation and store in database_citation
    get_patent_citation(patent_numbers, patent_links)


def get_patent_number():
    '''
    INPUT: patent_numbers.txt
    OUTPUT: list: patent_numbers, list: patent_links
    '''
    url = 'https://www.google.com/patents/'
    with open('../data/patent_numbers.txt') as f:
        content = f.readlines()
        patent_numbers = [x.strip() for x in content]
        patent_links = [url + 'US' + x for x in patent_numbers]
    return (patent_numbers, patent_links)


def get_patent_citation(patent_numbers, patent_links):
    '''
    INPUT: patent_numbers, patent_links
    OUTPUT: mongodb database: patent_database, collection: patent_citation
    '''

    # launch mongodb
    client = MongoClient('mongodb://localhost:27017/')
    db = client.patent_database
    collection = db.patent_citation

    patent_records = []

    tab_features = ['patent-number',
                    'backward-citations',
                    'npl-citations',
                    'forward-citations',
                    'classifications',
                    'legal-events']

    a = 0

    for num, link in enumerate(patent_links):

        url = link

        patent_tab_features = {x: {} for x in tab_features}

        response = requests.get(url)
        if response.status_code == 200:
            if num % 100 ==0:
                print num
            # print url

            html = response.content
            soup = BeautifulSoup(html, 'html.parser')

            patent_tab_features['patent-number'] = patent_numbers[num]

            table = soup.find('table', class_='patent-bibdata')
            tr = table.find_all('tr')
            for td in tr:
                head = td.find('td', class_='patent-bibdata-heading')
                body = td.find('td', class_='single-patent-bibdata')

                if head is not None and body is not None:
                    print head.get_text(), body.get_text()


            tab_content = soup.find_all('div',
                                class_='patent-section patent-tabular-section')

            for con in tab_content:
                subject = con.find('a')['id']
                if subject == 'backward-citations':
                    rows = con.find_all('tr') 
                    patent_tab_features['backward-citations'] = []
                    for row in rows[1:]:
                        patent_tab_features['backward-citations'].append(
                            {x.text : y.text for x, y in zip(rows[0], row)} )
                        
                elif subject == 'npl-citations':
                    rows = con.find_all('tr') 
                    patent_tab_features['npl-citations'] = len(rows) - 1

                elif subject == 'forward-citations':
                    rows = con.find_all('tr') 
                    patent_tab_features['forward-citations'] = []
                    for row in rows[1:]:
                        patent_tab_features['forward-citations'].append(
                            {x.text : y.text for x, y in zip(rows[0], row)} )

                elif subject == 'classifications':
                    pass
                    # reserved for future development

                elif subject == 'legal-events':
                    rows = con.find_all('tr')
                    patent_tab_features['legal-events'] = []
                    for row in rows[1:]:
                        patent_tab_features['legal-events'].append(
                            {x.text: y.text for x, y in zip(rows[0], row)} )

        try:
            collection.insert(patent_tab_features)
        except DuplicateKeyError:
            pass

    print collection.find().count()
    print collection.find_one()


if __name__ == '__main__':
    main()
