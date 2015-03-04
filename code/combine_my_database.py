# python code

# filename: combine_my_database.py
# ''' INPUT: None
# 	OUTPUT: Patent assignment database file -> ./my_database/full_database
# 	POINTS TO: get_query.py, [*populate_features.py, *calc_life_and_cost.py]
# '''
# output file: _database.py
# type: sql or *mongo
# why:  store patent info given from get_data_core.py + get_data_assignment + get_data_maintenance.py
# how: use psycopg to create new database called full_database.db.
# 	 take data from database_core + database_assignment + database_maintenance, and insert into full_database.db.



from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
import pandas as pd
import pickle
import re
import numpy as np

def main():
	create_dataframe()
	

def create_dataframe():

    client = MongoClient('mongodb://localhost:27017/')
    db = client.patent_database
    collection = db.patent_fields

    data = collection.find({},{"_id":0})

    df_raw = pd.DataFrame(columns=[u'Number', 
    							   u'Title', 
    							   u'Filing Date', 
    							   u'US Patent References', 
    							   u'Abstract', 
    							   u'Other Classes', 
    							   u'Primary Class', 
    							   u'Claims'])

    for i in data:
        df_raw = df_raw.append([i])     

    df = df_raw

    df['US Patent References'] = \
    				df['US Patent References'].apply(func_clean_reference)

    df['Filing Date'] = pd.to_datetime(df['Filing Date'])

    df['Other Classes'] = df['Other Classes'].apply(func_clean_class)

    df['Primary Class'] = df['Primary Class'].apply(func_clean_class)

    df['Claims'] = df['Claims'].apply(func_clean_claim)

    df = df.set_index(np.arange(df.shape[0]))

    print df.head()

def func_clean_reference(citations):
    if citations:
        result = []
        for cite in citations:
            result.append("".join(re.findall('\d+', cite)))
        return result

def func_clean_class(classes):
    if classes:
        result = [cls.strip() for cls in classes.split(',')]
        return result

def func_clean_claim(claims):
    if claims:
        result = [clm.strip() for clm in claims.split('\n')][1:]
        return result

if __name__ == '__main__':
    main()




