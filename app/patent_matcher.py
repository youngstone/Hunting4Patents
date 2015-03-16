import cPickle as pkl
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from patent_tokenizer import PatentTokenizer
import graphlab as gl


class PatentMatcher(object):
    def __init__(self, ngram_range=(1, 1), use_stem=False, use_tagger=False):
        self.title = None
        self.abstract = None
        self.claims = None

        self.title_vectors = None
        self.abstract_vectors = None
        self.claims_vectors = None

        self.database_title_vectors = None
        self.database_abstract_vectors = None
        self.database_claims_vectors = None

        self.tokenizer = None

        self.title_vectorizer = None
        self.abstract_vectorizer = None
        self.claims_vectorizer = None

        self.similarity_title = None
        self.similarity_abstract = None
        self.similarity_claims = None


        self.ngram_range = ngram_range
        self.use_tagger = use_tagger
        self.use_stem = use_stem

        self.recommendations = []
        self.similarity_scores = []

    def fit(self, title, abstract, claims):
        '''
        fit the input data with the model
        '''
        print 'fitting the model'
        self.initializer(title, abstract, claims)

        self.vectorize_title()
        self.vectorize_abstract()
        self.vectorize_claims()

        self.find_similarity()
        self.recommend()

    def not_empty_entry(self, entry):
        '''
        Check if it is an empty entry
        INPUT: STRING
        OUTPUT: BOOLEAN
        '''
        return False if entry == '' or re.match(r'^\s*$', entry) else True

    def load_tokenizer_and_database(self):
        print 'loading tokenizer'
        # load the tokenizer
        with open('../data/patent_tokenizer.pkl', 'rb') as handle:
            patent_tokenizer = pkl.load(handle)

        self.title_vectorizer = patent_tokenizer.get_title_vectorizer()
        self.abstract_vectorizer = patent_tokenizer.get_abstract_vectorizer()
        self.claims_vectorizer = patent_tokenizer.get_claims_vectorizer()

        print 'loading pre-calculated vectors'
        # load the vectors
        with open('../data/patent_title_vectors.pkl', 'rb') as handle:
            self.database_title_vectors = pkl.load(handle)
        with open('../data/patent_abstract_vectors.pkl', 'rb') as handle:
            self.database_abstract_vectors = pkl.load(handle)
        with open('../data/patent_claims_vectors.pkl', 'rb') as handle:
            self.database_claims_vectors = pkl.load(handle)

    def initializer(self, title, abstract, claims):
        print 'initializing'
        self.title = [title]
        self.abstract = [abstract]
        self.claims = [claims]

    def stem_text(self, text):
        # stem each word in the text
        snowball = SnowballStemmer('english')
        sub1 = re.sub(r'[^\x00-\x7F]+', ' ', text)
        sub2 = re.sub(r'[)(!?}{:;,\.\[\]]', '', sub1)
        sub3 = re.sub(r'\b\d+\b', ' ', sub2)
        stemmed = lambda doc: \
                        ' '.join(snowball.stem(word) for word in doc.split())
        return stemmed(sub3)

    def vectorize_title(self):
        title = self.title
        if self.use_stem:
            title = self.stem_text(title)
        self.title_vector = self.title_vectorizer.transform(title)

    def vectorize_abstract(self):
        abstract = self.abstract
        if self.use_stem:
            abstract = self.stem_text(abstract)
        self.abstract_vector = self.abstract_vectorizer.transform(abstract)

    def vectorize_claims(self):
        claims = self.claims
        if self.use_stem:
            claims = self.stem_text(claims)
        self.claims_vector = self.claims_vectorizer.transform(claims)

    def find_similarity(self):
        self.similarity_title = linear_kernel(self.title_vector,
                self.database_title_vectors).flatten()
        self.similarity_abstract = linear_kernel(self.abstract_vector,
                self.database_abstract_vectors).flatten()
        self.similarity_claims = linear_kernel(self.claims_vector,
                self.database_claims_vectors).flatten()

    def recommend(self, k=20):
        '''
        INPUT: None
        OUTPUT: Patent numbers of the top k best matched patents
        '''
        tag_title = 1 if self.not_empty_entry(self.title[0]) else 0
        tag_abstract = 1 if self.not_empty_entry(self.abstract[0]) else 0
        tag_claims = 1 if self.not_empty_entry(self.claims[0]) else 0

        if (tag_title + tag_abstract + tag_claims) > 0:
            total_similarity_score = (self.similarity_title * 1.0 + \
                self.similarity_abstract * 1.0 + \
                self.similarity_claims * 1.0) / \
                (tag_title + tag_abstract + tag_claims)
            top_match = np.argsort(total_similarity_score)[:-k:-1]
            self.recommendations = top_match
            self.similarity_scores = total_similarity_score[top_match]

        return (self.recommendations, self.similarity_scores)
