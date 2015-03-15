from flask import Flask
from flask import request
from flask import render_template
# from patent_matcher import PatentMatcher
import pandas as pd
import re
import cPickle as pkl
import json
import numpy as np
from random import shuffle
import plotly.plotly as py
from plotly.graph_objs import *

app = Flask(__name__)

def run_on_start_1():
    print "loading patent_matcher..."
    with open('../data/patent_matcher.pkl', 'rb') as handle:
        matcher = pkl.load(handle)
    return matcher

def run_on_start_2():
    print "loading dataframe..."
    with open('../data/patent_dataframe.pkl', 'rb') as handle:
        df = pkl.load(handle)
    return df

def color_generator(color_pool):
    for i in color_pool:
        yield i

def plot(df_filtered, scores):
    pat_id = df_filtered['Patent Number'].values

    tit = df_filtered['Title'].values

    cls = [c[0] for c in df_filtered['Primary Class'].values]

    text = ['US' + i + '<br>' + t + '<br>' + c for i, t, c in zip(pat_id, tit, cls)]

    gyear = np.array([int(str(x)[:4]) for x in df_filtered['Filing Date'].values])

    pr = df_filtered['PageRank'].values
    pr = pr / pr.max() 

    sim = scores

    color_pool = ["#A2692C","#E147E7","#4FADE4","#3EC126","#E43077",
              "#448962","#CB91DC","#E54523","#B5AF1B","#716BEC",
              "#A6677D","#ED961A","#537C8C","#45BF6E","#787D2E",
              "#BC4D9B","#42C4D2","#6872B9","#DA8B70","#8CB952",
              "#E6384C","#358C2A","#DC31B6","#3CBCA7","#D06626",
              "#D673EB","#CBA156","#527DE4","#B69ECA","#E6787F",
              "#E16452","#E496AE","#8FA7BE","#D4A42F","#679BEA",
              "#AC625E","#EB67A1","#417FB4","#9DAF65","#B187EE",
              "#8A7643","#6EB989","#74759F","#D54764","#45867E",
              "#E28F3F","#E03392","#93BA2D","#47C14E","#B342BC",
              "#A760B0","#BB557A","#B35B3F","#A39DE3","#A27D22",
              "#E682D3","#73B9AF","#AC5BE4","#D183B4","#8D9427",
              "#8762CE","#E861CB","#65B022","#648055","#36BF86",
              "#E08E5B","#59ACC4","#946E8F","#478E50","#5F8E33",
              "#84A9D5","#816CC1","#B6B245","#41A0A3","#6FBD64"]

    shuffle(color_pool)

    a = color_generator(color_pool)

    cls_dict = {}
    for c in cls:
        cls_dict[c] = cls_dict.get(c, a.next())
        
    colors = [cls_dict[c] for c in cls]

    trace1 = Scatter(
            x = gyear,
            y = sim.tolist(),
            mode ='markers',
            text = text,
            marker = Marker(
            size=(20 + pr * 40).tolist(),
            color = colors
            )
            )

    layout = Layout(
            title="Selected Patents",
            showlegend=False,
            autosize=False,
            width=650,
            height=500,
            xaxis=XAxis(
                title='Publication Year',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=1.5,
                gridcolor='#FFFFFF'
            ),
            yaxis=YAxis(
                title='Similarity Score',
                zeroline=True,
                ticks='outside',
                ticklen=8,
                tickwidth=1.5,
                gridcolor='#FFFFFF'
            ),
            
            plot_bgcolor='#EFECEA',
            hovermode='closest'
            )

    data = Data([trace1])


    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='Selected Patents', auto_open=False)

    return plot_url

    # fig = Figure(data=data, layout=layout)
    # # plot_url = py.iplot(fig)

    # chart = py.iplot(data, filename = 'basic-line')
    # return chart.embed_code


def plot_count(d):
    x = []
    y = []
    for k, v in d.iteritems():
        x.append(int(k))
        y.append(int(v))
        
    data = Data([
        Bar(
            x=x,
            y=y
        )
    ])
    plot_url = py.plot(data, auto_open=False)
    return plot_url


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

    # Initialize PatentMatcher
    # with open('../data/patent_matcher.pkl', 'rb') as handle:
    #         matcher = pkl.load(handle)

    # Fit the query
    app.matcher.fit(title, abstract, claims)
    patents, scores = app.matcher.recommendations, app.matcher.similarity_scores

    # print 'title:', '==' + title + '=='
    # print 'title:', '==' + abstract + '=='
    # print 'matched?', bool(re.match(r'^\s*$', title))
    # print 'matched?', bool(re.match(r'^\s*$', abstract))

    df = pd.read_csv('../data/selected_patent.csv', index_col=None)

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
    title = request.form.get('title', None)
    abstract = request.form.get('abstract', None)
    claims = request.form.get('claims', None)

    # Fit the query
    app.matcher.fit(title, abstract, claims)
    patents, scores = app.matcher.recommendations, app.matcher.similarity_scores

    df = pd.read_pickle('../data/patent_dataframe.pkl')

    df_filtered = df.iloc[patents]

    citation_data = df_filtered['forward-citations_count']
    citation_plot_url = []
    # for d in citation_data:
    #     url = plot_count(d)
    #     citation_plot_url.append(url)

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
    # return render_template('my_table2.html')
    # return render_template('plot.html', data=urll)


if __name__ == '__main__':
    app.matcher = run_on_start_1()
    app.df = run_on_start_2()
    app.run(host='0.0.0.0', port=7878, debug=True)
