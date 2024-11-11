import pandas as pd
import sqlite3

# conn = sqlite3.connect("data/db.sqlite")

def find_movie_id(title: str, conn):
    query = f'SELECT * from movie where original_title like "%{title}%"'
    search = pd.read_sql_query(query, conn)
    print(search.head())
    conn.close()