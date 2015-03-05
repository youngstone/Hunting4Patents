from flask import Flask
from flask import request
from flask import render_template
import pandas as pd

app = Flask(__name__)


# OUR HOME PAGE
#============================================
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/table')
def index_hover_table():
    df = pd.read_csv('../data/citation.csv', index_col=None)
    df = df[['patent','cited']]
    px = df['patent'].values
    years = df['cited'].values
    x = zip(px,years)
    return render_template('my_table.html', data = x)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
