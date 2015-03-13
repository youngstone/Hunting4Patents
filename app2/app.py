from flask import Flask
from flask import request
from flask import render_template
# from patent_matcher import PatentMatcher
import pandas as pd
import re
import cPickle as pkl

app = Flask(__name__)

def run_on_start():
    print "loading..."
    with open('../data/patent_matcher.pkl', 'rb') as handle:
        matcher = pkl.load(handle)
    return matcher


# OUR HOME PAGE
# ============================================
@app.route('/')
def welcome():
    return render_template('query.html')


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


if __name__ == '__main__':
    app.matcher = run_on_start()
    app.run(host='0.0.0.0', port=7878, debug=True)
