# python code

# filename: clean_my_database.py
# '''
#     INPUT: None
#     OUTPUT: clean patent content dataframe -> ./my_database/dataframe.pkl
#     POINTS TO: calc_pagerank.py
# '''
# output file: dataframe.pkl
# type: pickle file
# why:  store patent info in form of dataframe
# how: take data from database_fields, store in dataframe and clean the format


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

    # launch mongodb
    client = MongoClient('mongodb://localhost:27017/')
    db = client.patent_database
    collection = db.patent_fields

    # use all data
    data = collection.find({}, {"_id": 0})

    # set up empty dataframe
    df = pd.DataFrame(columns=[u'Number',
                               u'Title',
                               u'Filing Date',
                               u'US Patent References',
                               u'Abstract',
                               u'Other Classes',
                               u'Primary Class',
                               u'Claims'])

    # populate the dataframe with raw data
    for i in data:
        df = df.append([i])

    # clean each column
    df['US Patent References'] = \
        df['US Patent References'].apply(func_clean_reference)
    df['Filing Date'] = pd.to_datetime(df['Filing Date'])
    df['Other Classes'] = df['Other Classes'].apply(func_clean_class)
    df['Primary Class'] = df['Primary Class'].apply(func_clean_class)
    df['Claims'] = df['Claims'].apply(func_clean_claim)
    df = df.set_index(np.arange(df.shape[0]))

    # store dataframe in pickle file
    df.to_pickle('../data/dataframe.pkl')


def func_clean_reference(citations):
    '''
    INPUT: an entry
    OUTPUT: an entry
    '''
    if citations:
        result = []
        for cite in citations:
            result.append("".join(re.findall('\d+', cite)))
        return result


def func_clean_class(classes):
    '''
    INPUT: an entry
    OUTPUT: an entry
    '''
    if classes:
        result = [cls.strip() for cls in classes.split(',')]
        return result


def func_clean_claim(claims):
    '''
    INPUT: an entry
    OUTPUT: an entry
    '''
    if claims:
        result = [clm.strip() for clm in claims.split('\n')][1:]
        return result


if __name__ == '__main__':
    main()
