```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Get data files
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# Remove users with less than 200 ratings and books with less than 100 ratings
user_rating_counts = df_ratings['user'].value_counts()
book_rating_counts = df_ratings['isbn'].value_counts()

df_ratings = df_ratings[df_ratings['user'].isin(user_rating_counts[user_rating_counts >= 200].index)]
df_ratings = df_ratings[df_ratings['isbn'].isin(book_rating_counts[book_rating_counts >= 100].index)]

# Merge books and ratings data
df = pd.merge(df_ratings, df_books, on='isbn')

# Remove duplicates
df = df.drop_duplicates(['user', 'title'])

# Create pivot table
df_pivot = df.pivot(index='title', columns='user', values='rating').fillna(0)
df_matrix = csr_matrix(df_pivot.values)

# Build KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(df_matrix)

# Function to get recommendations
def get_recommends(book = ""):
    try:
        query_index = df_pivot.index.get_loc(book)
    except KeyError:
        print(f"Book '{book}' not found in database.")
        return None
    
    distances, indices = model_knn.kneighbors(
        df_pivot.iloc[query_index, :].values.reshape(1, -1),
        n_neighbors=6
    )
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append([
            df_pivot.index[indices.flatten()[i]], 
            distances.flatten()[i]
        ])
    
    recommended_books = sorted(recommended_books, key=lambda x: x[1], reverse=True)
    
    return [book, recommended_books]

# Test the function
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```
