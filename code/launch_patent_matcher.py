import cPickle as pkl
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from patent_tokenizer import PatentTokenizer
from textblob import TextBlob
import graphlab as gl
from patent_matcher import PatentMatcher



matcher = PatentMatcher()
matcher.load_tokenizer_and_database()

# save the matcher
with open('../data/patent_matcher.pkl', 'wb') as handle:
    pkl.dump(matcher, handle)