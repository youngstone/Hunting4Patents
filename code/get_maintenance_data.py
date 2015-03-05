# python code

# filename: get_maintenance_data.py
# ''' INPUT: maintenance_fee_event.txt
# 	OUTPUT: maintenance database -> ./my_database/maintenance.sqlite3
# 	POINTS TO: get_maintenance_events.py
# '''
# why: get patent maintenance data
# how: create a python pipepline to convert .txt file to sqlite3 database


import pandas as pd
import sqlite3

def main():
	conn = sqlite3.connect('../database/maintenance.sqlite3')
	c = conn.cursor()

	c.execute(''' DROP TABLE IF EXISTS maintenance''')

	c.execute('''CREATE TABLE maintenance (
	                patent varchar(8),
	                small_entity varchar(1),
	                maintenance_day varchar(8),
	                maintenance_event varchar(5)
	                )''')
	conn.commit()

	filename = '../data/MaintFeeEvents_20150223.txt'
	f = open(filename)
	i = 0
	for line in f:
	    i += 1
	    if i % 100000 == 0:
	        print i
	    items = line.split()
	    if len(items) == 7:
	        cmd = '''INSERT INTO maintenance VALUES('%s', '%s', '%s', '%s')''' \
	                % (items[0], items[2], items[5], items[6])
	        c.execute(cmd)
	        conn.commit()

if __name__ == '__main__':
	main()