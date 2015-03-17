# python code

# filename: calc_pagerank_for_whole.py
# '''
#     INPUT: full_citation.csv
#     OUTPUT: csv file -> ./data/full_pagerank.csv
# '''
# output file: full_pagerank.csv
# type: csv file
# why:  store patent pagerank in csv file
# how: take data from full_citation.csv, calculate pagerank


import graphlab as gl


def calc_pagerank():

    sf = gl.SFrame.read_csv('../data/full_citation.csv',
                            delimiter=',', error_bad_lines=True)
    gr = gl.SGraph(sf, vid_field='Patent', src_field='Citation',
                   dst_field='Patent')
    gr = gr.add_edges(sf, src_field='Citation', dst_field='Patent')
    pr = gl.pagerank.create(gr)
    pr_out = pr['pagerank']
    pr_out = pr_out.rename({'__id': 'Patent'})
    pr_out = pr_out[['Patent', 'pagerank']]
    pr_out.save('../data/full_pagerank.csv', format='csv')


if __name__ == '__main__':
    calc_pagerank()
