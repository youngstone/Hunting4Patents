import pandas as pd
import sqlite3


def export_full_citation_with_category():
    con = sqlite3.connect("../database/citation.sqlite3")
    df = pd.read_sql("SELECT patent, citation, category FROM citation", con)
    df.to_csv('../data/full_citation_with_category.csv', index=False)


def export_full_citation():
    con = sqlite3.connect("../database/citation.sqlite3")
    df = pd.read_sql("SELECT patent, citation FROM citation", con)
    df.to_csv('../data/full_citation.csv', index=False)


if __name__ == '__main__':

    # export_full_citation()

    export_full_citation_with_category()
