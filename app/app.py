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

    # # load from pickled file
    # with open('../data/patent_matcher.pkl', 'rb') as handle:
    #     matcher = pkl.load(handle)

    return matcher

def run_on_start_2():
    print "loading dataframe..."
    with open('datafile/patent_dataframe.pkl', 'rb') as handle:
        df = pkl.load(handle)
    print "dateframe loaded."
    return df



# OUR HOME PAGE
# ============================================
@app.route('/')
def welcome():
    return render_template('query2.html')


# @app.route('/blog')
# def blog():
#     return render_template('blog.html')


@app.route('/table', methods=['POST'])
def index_hover_table():
    title = request.form.get('title', None)
    abstract = request.form.get('abstract', None)
    claims = request.form.get('claims', None)

    # Fit the query
    app.matcher.fit(title, abstract, claims)
    patents, scores = app.matcher.recommendations, app.matcher.similarity_scores

    df = pd.read_csv('datafile/selected_patent.csv', index_col=None)

    df_filtered = df.iloc[patents]

    pat = df_filtered['Patent Number'].values
    sim = scores
    pr = df_filtered['PageRank'].values
    expire_day = df_filtered['Default Expire Day'].values
    time_to_expire = df_filtered['Time to Expire'].values
    # url = 'http://www.freepatentsonline.com/'
    # link = df['Patent Number'].apply(lambda x: url + x +'.html')
    url = 'https://www.google.com/patents/'
    link = df_filtered['Patent Number'].apply(lambda x: url + 'US' + x)
    x = zip(pat, sim, pr, expire_day, time_to_expire, link)
    return render_template('my_table.html', data=x)


@app.route('/table2', methods=['POST'])
def index_hover_table2():
    print 'hi'
    title = request.form.get('title', None)
    abstract = request.form.get('abstract', None)
    claims = request.form.get('claims', None)

    # Fit the query
    app.matcher.fit(title, abstract, claims)
    patents, scores = app.matcher.recommendations, app.matcher.similarity_scores

    df = pd.read_pickle('datafile/patent_dataframe.pkl')

    df_filtered = df.iloc[patents]

    citation_data = df_filtered['forward-citations_count']
    citation_plot_url = []
    # for d in citation_data:
    #     url = plot_count(d)
    #     citation_plot_url.append(url)
    print 'now', df_filtered.shape
    plot_url = plot(df_filtered, scores)
    print plot_url

    pat_id = df_filtered['Patent Number'].values

    tit = df_filtered['Title'].values

    cls = [c[0] for c in df_filtered['Primary Class'].values]

    text = ['US' + i + '<br>' + t + '<br>' + c for i, t, c in zip(pat_id, tit, cls)]

    gyear = np.array([int(str(x)[:4]) for x in df_filtered['Filing Date'].values])

    pr = df_filtered['PageRank'].values
    pr = pr / pr.max() 

    sim = scores


    pat = df_filtered['Patent Number'].values
    sim = scores
    pr = df_filtered['PageRank'].values
    expire_day = df_filtered['Default Expire Day'].values
    time_to_expire = df_filtered['Time to Expire'].values

    url = 'https://www.google.com/patents/'
    link = df_filtered['Patent Number'].apply(lambda x: url + 'US' + x)

    x = zip(pat_id, sim, pr, expire_day, time_to_expire, link)

    return render_template('my_table2.html', data=x, plot_url=plot_url)



if __name__ == '__main__':
    app.matcher = run_on_start_1()
    app.df = run_on_start_2()
    print "Preloading completed."
    print "Enjoy."
    app.run(host='0.0.0.0', port=80, debug=True)
