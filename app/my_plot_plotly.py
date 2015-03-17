# plot_plotly.py
# from patent_matcher import PatentMatcher
import pandas as pd
import re
import cPickle as pkl
import json
import numpy as np
from random import shuffle
import plotly.plotly as py
from plotly.graph_objs import *


def color_generator(color_pool):
    for i in color_pool:
        yield i

def plot(df_filtered, scores):
    pat_id = df_filtered['Patent Number'].values

    tit = df_filtered['Title'].values

    cls = [c[0] for c in df_filtered['Primary Class'].values]

    text = ['US' + i + '<br>' + t + '<br>' + c for i, t, c in zip(pat_id, tit, cls)]

    gyear = np.array([int(str(x)[:4]) for x in df_filtered['Filing Date'].values])

    pr = df_filtered['PageRank'].values
    pr = pr / pr.max() 

    sim = scores

    color_pool = ["#A2692C","#E147E7","#4FADE4","#3EC126","#E43077",
              "#448962","#CB91DC","#E54523","#B5AF1B","#716BEC",
              "#A6677D","#ED961A","#537C8C","#45BF6E","#787D2E",
              "#BC4D9B","#42C4D2","#6872B9","#DA8B70","#8CB952",
              "#E6384C","#358C2A","#DC31B6","#3CBCA7","#D06626",
              "#D673EB","#CBA156","#527DE4","#B69ECA","#E6787F",
              "#E16452","#E496AE","#8FA7BE","#D4A42F","#679BEA",
              "#AC625E","#EB67A1","#417FB4","#9DAF65","#B187EE",
              "#8A7643","#6EB989","#74759F","#D54764","#45867E",
              "#E28F3F","#E03392","#93BA2D","#47C14E","#B342BC",
              "#A760B0","#BB557A","#B35B3F","#A39DE3","#A27D22",
              "#E682D3","#73B9AF","#AC5BE4","#D183B4","#8D9427",
              "#8762CE","#E861CB","#65B022","#648055","#36BF86",
              "#E08E5B","#59ACC4","#946E8F","#478E50","#5F8E33",
              "#84A9D5","#816CC1","#B6B245","#41A0A3","#6FBD64"]

    shuffle(color_pool)

    a = color_generator(color_pool)

    cls_dict = {}
    for c in cls:
        cls_dict[c] = cls_dict.get(c, a.next())
        
    colors = [cls_dict[c] for c in cls]

    trace1 = Scatter(
            x = gyear,
            y = sim.tolist(),
            mode ='markers',
            text = text,
            marker = Marker(
            size=(20 + pr * 40).tolist(),
            color = colors
            )
            )

    layout = Layout(
            title="Selected Patents",
            showlegend=False,
            autosize=False,
            width=650,
            height=500,
            xaxis=XAxis(
                title='Publication Year',
                zeroline=False,
                ticks='outside',
                ticklen=8,
                tickwidth=1.5,
                gridcolor='#FFFFFF'
            ),
            yaxis=YAxis(
                title='Similarity Score',
                zeroline=True,
                ticks='outside',
                ticklen=8,
                tickwidth=1.5,
                gridcolor='#FFFFFF'
            ),
            
            plot_bgcolor='#EFECEA',
            hovermode='closest'
            )

    data = Data([trace1])


    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='Selected Patents', auto_open=False)

    return plot_url

    # fig = Figure(data=data, layout=layout)
    # # plot_url = py.iplot(fig)

    # chart = py.iplot(data, filename = 'basic-line')
    # return chart.embed_code


def plot_count(d):
    x = []
    y = []
    for k, v in d.iteritems():
        x.append(int(k))
        y.append(int(v))
        
    data = Data([
        Bar(
            x=x,
            y=y
        )
    ])
    plot_url = py.plot(data, auto_open=False)
    return plot_url
