from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import cPickle as pkl
from nltk.stem.snowball import SnowballStemmer
import re
from patent_tokenizer import PatentTokenizer


patent_tokenizer = PatentTokenizer()
patent_tokenizer.set_df('../data/patent_text_df.pkl')
patent_tokenizer.set_vectors()

patent_title_vectorizer = patent_tokenizer.get_title_vectorizer()
patent_abstract_vectorizer = patent_tokenizer.get_abstract_vectorizer()
patent_claims_vectorizer = patent_tokenizer.get_claims_vectorizer()

patent_title_vectors = patent_tokenizer.get_title_vectors()
patent_abstract_vectors = patent_tokenizer.get_abstract_vectors()
patent_claims_vectors = patent_tokenizer.get_claims_vectors()

# patent_numbers = patent_tokenizer.get_patent_number()
# patent_title_vectors = patent_tokenizer.get_title()
# patent_abstract_vectors = patent_tokenizer.get_abstract()
# patent_claims_vectors = patent_tokenizer.get_claims()

# info = zip(patent_numbers,
#            patent_title_vectors,
#            patent_abstract_vectors,
#            patent_claims_vectors)
# print info[0]

# save the tokenizer
with open('../data/patent_tokenizer.pkl', 'wb') as handle:
    pkl.dump(patent_tokenizer, handle)

# save the vectors
with open('../data/patent_title_vectors.pkl', 'wb') as handle:
    pkl.dump(patent_title_vectors, handle)
with open('../data/patent_abstract_vectors.pkl', 'wb') as handle:
    pkl.dump(patent_abstract_vectors, handle)
with open('../data/patent_claims_vectors.pkl', 'wb') as handle:
    pkl.dump(patent_claims_vectors, handle)