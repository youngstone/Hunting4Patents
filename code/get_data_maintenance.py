# python code

# filename:  get_data_maintenance.py
# ''' INPUT: maintenancefee.txt
# 	OUTPUT: Patent maintenance database file -> ./my_database/database_maintenance
# 	POINTS TO: combine_my_data.py
# '''
# Purpose: parse maintenancefee.txt, get maintenance action records for each patent
# then store all that information into database 'database_maintenance'

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

def load_data():
	filename = '../data/MaintFeeEvents_20150223.txt'
	f = open(filename)
	for line in f:
		print line
		items = line.split()
		print len(items)
		for item in items:
			print item, len(item)
		break

def main():
	load_data()

if __name__ == '__main__':
	main()