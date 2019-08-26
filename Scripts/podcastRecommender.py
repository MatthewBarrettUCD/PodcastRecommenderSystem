import numpy as np
import pandas as pd 
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
print(os.listdir("./input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

podcasts = pd.read_csv('./input/podcasts.csv')
podcasts.info()

# Cleaning the data

podcasts = podcasts[podcasts.language == 'English']

podcasts = podcasts.dropna(subset=['description'])

podcasts = podcasts.drop_duplicates('itunes_id')

# Checking the length of the descriptions of each podcast
podcasts['description_length'] = [len(x.description.split()) for _, x in podcasts.iterrows()]
print(podcasts['description_length'].describe(include = 'all'))

podcasts['categories_length'] = [len(x.categories.split('|')) for _, x in podcasts.iterrows()]
print(podcasts['categories_length'].describe(include = 'all'))

podcasts = podcasts[podcasts.description_length >= 20]

# Creating a dataset of my personal favorite podcasts

favorite_podcasts = ['The Joe Rogan Experience', "Your Mom's House with Christina P. and Tom Segura", "VIEWS with David Dobrik and Jason Nash"]
favorites = podcasts[podcasts.title.isin(favorite_podcasts)]

podcasts = podcasts[~podcasts.isin(favorites)].sample(5000)
data = pd.concat([podcasts,favorites], sort = True).reset_index(drop = True)

data = data.astype({"description": str})

tf = TfidfVectorizer(analyzer = 'word', ngram_range= (1, 2), min_df = 0, stop_words = 'english')
tf_idf = tf.fit_transform(data['description'])

#print(len(tf_idf))
#print(tf.get_feature_names())
print(tf_idf.shape)

similarity = linear_kernel(tf_idf,tf_idf)

x = data[data.title == "Your Mom's House with Christina P. and Tom Segura"].index[0]
similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
#print((similar_idx))
for i in similar_idx:
    print(i)


for i in similar_idx:
    print(similarity[x][i], '-', data.title[i], '-', data.description[i], '\n - ', data.categories[i], ' - \n')
print('Original - ' + data.description[x] + '\n - ' + data.categories[x] + ' - \n')