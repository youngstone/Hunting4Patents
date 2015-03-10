

import pandas as pd
import sqlite3


def load_citation_database():
	con = sqlite3.connect("../database/citation.sqlite3")
	df = pd.read_sql("SELECT patent, citation, strftime('%Y', Cit_Date) AS year FROM citation", con)
	df.to_csv('../data/full_citation_with_year.csv', index=False)


if __name__ == '__main__':
	load_citation_database()
