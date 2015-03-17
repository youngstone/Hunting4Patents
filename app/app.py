from flask import Flask
from flask import request
from flask import render_template
from patent_matcher import PatentMatcher
import pandas as pd
import re
import cPickle as pkl
import json
import numpy as np
from random import shuffle
import plotly.plotly as py
from plotly.graph_objs import *
from my_plot_plotly import *

app = Flask(__name__)


def run_on_start_1():
    print "loading patent_matcher..."
    matcher = PatentMatcher()
    matcher.calc_tokenizer_and_vectors()
    return matcher


def run_on_start_2():
    print "loading dataframe..."
    with open('datafile/patent_dataframe.pkl', 'rb') as handle:
        df_1 = pkl.load(handle)
    with open('datafile/patent_prediction.pkl', 'rb') as handle:
        df_2 = pkl.load(handle)
    df = pd.merge(df_1, df_2, on='Patent Number', how='inner')
    print "dateframe loaded."
    return df


# THE HOME PAGE - Query
# ============================================
@app.route('/')
def welcome():
    return render_template('query.html')


# THE RESPONSE PAGE - Query Result
# ============================================
@app.route('/table', methods=['POST'])
def index_hover_table():
    print 'hi'
    title = request.form.get('title', None)
    abstract = request.form.get('abstract', None)
    claims = request.form.get('claims', None)

    # Fit the query
    app.matcher.fit(title, abstract, claims)
    patents = app.matcher.recommendations
    scores = app.matcher.similarity_scores
    df_filtered = app.df.iloc[patents]

    if df_filtered.shape[0] == 0:
        return render_template('my_table.html')

    citation_data = df_filtered['forward-citations_count']
    citation_plot_url = []

    plot_url = plot(df_filtered, scores)
    print plot_url

    pat = df_filtered['Patent Number'].values
    sim = scores
    pr = df_filtered['PageRank'].values
    expire_day = df_filtered['Default Expire Day'].values
    time_to_expire = df_filtered['Time to Expire'].values
    pred = df_filtered['Expiration Prediction'].values

    url = 'https://www.google.com/patents/'
    link = df_filtered['Patent Number'].apply(lambda x: url + 'US' + x)

    x = zip(pat, sim, pr, expire_day, time_to_expire, link, pred)

    return render_template('my_table.html', data=x, plot_url=plot_url)


if __name__ == '__main__':
    app.matcher = run_on_start_1()
    app.df = run_on_start_2()
    print "Preloading completed."
    print "Enjoy."
    app.run(host='0.0.0.0', port=80, debug=True)
