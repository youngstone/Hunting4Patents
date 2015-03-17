import cPickle as pkl
from patent_tokenizer import PatentTokenizer


def main():
    
    print "Launching PatentTokenizer..."
    patent_tokenizer = PatentTokenizer()
    patent_tokenizer.set_df('../data/patent_text_df.pkl')

    print "Setting dataframe..."
    patent_tokenizer.set_vectors()

    print "Saving result..."
    patent_title_vectorizer = patent_tokenizer.get_title_vectorizer()
    patent_abstract_vectorizer = patent_tokenizer.get_abstract_vectorizer()
    patent_claims_vectorizer = patent_tokenizer.get_claims_vectorizer()

    patent_title_vectors = patent_tokenizer.get_title_vectors()
    patent_abstract_vectors = patent_tokenizer.get_abstract_vectors()
    patent_claims_vectors = patent_tokenizer.get_claims_vectors()

    '''
    print "Saving to pickle file..."
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
    '''


if __name__ == '__main__':
    main()
