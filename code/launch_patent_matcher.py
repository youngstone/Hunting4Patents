import cPickle as pkl
from patent_tokenizer import PatentTokenizer
from patent_matcher import PatentMatcher


def main():
    print "Launching PatentMatcher..."
    matcher = PatentMatcher()
    matcher.load_tokenizer_and_database()

    # save the matcher
    print "Saving to pickle file..."
    with open('../data/patent_matcher.pkl', 'wb') as handle:
        pkl.dump(matcher, handle)

if __name__ == '__main__':
    main()
