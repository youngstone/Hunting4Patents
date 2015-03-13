from flask import Flask
from flask import request
from flask import render_template
from patent_matcher import PatentMatcher
import pandas as pd
import re
import cPickle as pkl
# from werkzeug.contrib.cache import SimpleCache

import time
from flask.ext.cache import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@cache.memoize(timeout=60)
def query_db():
    time.sleep(5)
    with open('../data/patent_matcher.pkl', 'rb') as handle:
        matcher = pkl.load(handle)    
    print 'loading completed'
    return matcher

@app.route('/')
def index():
    return query_db()

app.run(debug=True)
