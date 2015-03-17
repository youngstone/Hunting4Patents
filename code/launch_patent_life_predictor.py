import cPickle as pkl
from patent_life_predictor import *


def main():

    print "Launching PatentLongevityPredictor..."

    with open('../data/patent_dataframe.pkl', 'rb') as handle:
        df_input = pkl.load(handle)

    plr = PatentLongevityPredictor()
    plr.initialization(df_input)
    plr.print_result()
    plr.predict()


if __name__ == '__main__':
	main()
