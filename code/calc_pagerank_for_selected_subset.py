# python code

# filename: calc_pagerank_for_selected_subset.py
# '''
#     INPUT: None
#     OUTPUT: csv file -> ./data/selected_patent.csv
#     POINTS TO: app/app.py
# '''
# output file: selected_patent.csv
# type: csv file
# why:  store patent info in csv file
# how: take data from dataframe.pkl, calculate pagerank, expiration date


import pandas as pd
import numpy as np
from datetime import date
import graphlab as gl
from collections import Counter, defaultdict


def main():
    # read pickled Pandas DataFrame
    df = pd.read_pickle('../data/dataframe.pkl')

    # get today and the screening day for expired patents
    today = date.today()
    # screening_day = date(today.year - 20, today.month, today.day)
    # df_select = df[df['Filing Date'] > screening_day]

    # create citation Pandas DataFrame
    df_citation = pd.DataFrame(columns=['patent', 'cited'])

    for ix in xrange(len(df)):
        pat = df['Number'].iloc[ix]
        cites = df['US Patent References'].iloc[ix]
        if cites:
            for cit in cites:
                df_citation = df_citation.append({'patent': pat, 'cited': cit},
                                                 ignore_index=True)

    # create Graphlab SFrame
    sf = gl.SFrame(data=df_citation)

    # create Graphlab Graph
    gr = gl.SGraph(sf, vid_field='patent', src_field='cited',
                   dst_field='patent')
    gr = gr.add_edges(sf, src_field='cited', dst_field='patent')

    # calculate pagerank
    pr = gl.pagerank.create(gr)

    # create new Pandas DataFrame to store results
    columns = ['Patent Number',
               'PageRank',
               'Default Expire Day',
               'Time to Expire']
    df_result = pd.DataFrame(columns=columns)

    # put calculated pagerank into Pandas DataFrame
    df_pagerank = pr['pagerank'].to_dataframe()

    for pat in df['Number'].values:
        day = df['Filing Date'][df['Number'] == pat].iloc[0]
        expire_day = date(day.year + 20, day.month, day.day)
        time_to_expire = expire_day - today
        if np.sum(df_pagerank['__id'] == pat) == 1:
            pr = df_pagerank[df_pagerank['__id'] == pat]['pagerank'].values[0]
        else:
            pr = 0
        row = {'Patent Number': pat,
               'PageRank': pr,
               'Default Expire Day': expire_day,
               'Time to Expire': time_to_expire}
        df_result = df_result.append(row, ignore_index=True)

    print df_result.head()

    df_result.to_csv('../data/selected_patent.csv', index=False)


if __name__ == '__main__':
    main()
