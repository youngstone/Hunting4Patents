from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import cPickle as pkl
from nltk.stem.snowball import SnowballStemmer
import re


class PatentTokenizer(object):
    def __init__(self, ngram_range=(1, 1), use_stem=False):
        self.ngram_range = ngram_range
        self.use_stem = use_stem
        self.max_features = 2000
        self.title_vectorizer = TfidfVectorizer(stop_words='english',
                  ngram_range=self.ngram_range, max_features=self.max_features)
        self.abstract_vectorizer = TfidfVectorizer(stop_words='english',
                  ngram_range=self.ngram_range, max_features=self.max_features)
        self.claims_vectorizer = TfidfVectorizer(stop_words='english',
                  ngram_range=self.ngram_range, max_features=self.max_features)
        self.title_vectors = None
        self.abstract_vectors = None
        self.claims_vectors = None
        self.df = None

    def set_df(self, file_name):
        # load df from pickle file
        df_input = pd.read_pickle(file_name)
        self.df = df_input[['Number', 'Title', 'Abstract', 'Claims']]

    def stem_text(self, text):
        # stem each word in the text
        snowball = SnowballStemmer('english')
        sub1 = re.sub(r'[^\x00-\x7F]+', ' ', text)
        sub2 = re.sub(r'[)(!?}{:;,\.\[\]]', '', sub1)
        sub3 = re.sub(r'\b\d+\b', ' ', sub2)
        stemmed = lambda doc: \
            ' '.join(snowball.stem(word) for word in doc.split())
        return stemmed(sub3)

    def set_vectors(self):
        # transform patent title
        title = self.get_title()
        if self.use_stem:
            title = self.stem_text(title)
        self.title_vectors = \
            self.title_vectorizer.fit_transform(title).toarray()
        # transform patent abstract
        abstract = self.get_abstract()
        if self.use_stem:
            abstract = self.stem_text(abstract)
        self.abstract_vectors = \
            self.abstract_vectorizer.fit_transform(abstract).toarray()
        # transform patent claims
        claims = self.get_claims()
        if self.use_stem:
            claims = self.stem_text(claims)
        self.claims_vectors = \
            self.claims_vectorizer.fit_transform(claims).toarray()

    def get_df(self):
        return self.df

    def get_title_vectorizer(self):
        return self.title_vectorizer

    def get_abstract_vectorizer(self):
        return self.abstract_vectorizer

    def get_claims_vectorizer(self):
        return self.claims_vectorizer

    # For output tf-idf vectors:

    def get_title_vectors(self):
        return self.title_vectors

    def get_abstract_vectors(self):
        return self.abstract_vectors

    def get_claims_vectors(self):
        return self.claims_vectors

    # For output patent text content:

    def get_patent_number(self):
        return self.df['Number']

    def get_title(self):
        return self.df['Title']

    def get_abstract(self):
        return self.df['Abstract']

    def get_claims(self):
        return self.df['Claims']
