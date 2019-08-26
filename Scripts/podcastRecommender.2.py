import numpy as np
import pandas as pd 
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
#print(os.listdir("./input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

podcasts = pd.read_csv('./input/podcasts.csv')

#Cleaning the data
podcasts = podcasts[podcasts.language == 'English']

podcasts = podcasts.dropna(subset=['description'])

podcasts = podcasts.drop_duplicates('itunes_id')

#Definining my favorite podcasts to ensure they are included in the sample
favorite_podcasts = ['The Joe Rogan Experience', "Your Mom's House with Christina P. and Tom Segura", "VIEWS with David Dobrik and Jason Nash"]
favorites = podcasts[podcasts.title.isin(favorite_podcasts)]

#Taking a sample of 5000 podcasts from the dataset
podcasts = podcasts[~podcasts.isin(favorites)].sample(5000)
data = pd.concat([podcasts,favorites], sort = True).reset_index(drop = True)

#Casting the descriptions to strings to remove any issues with word analysis
data = data.astype({"description": str})

#Building a sparse matrix of occurences of bigram words
tf = TfidfVectorizer(analyzer = 'word', ngram_range= (1, 2), min_df = 0, stop_words = 'english')
tf_idf = tf.fit_transform(data['description'])

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(tf_idf)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=tf.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
#print(df_idf.sort_values(by=['idf_weights']))

#Comparing all podcasts to each other for similarity
similarity = linear_kernel(tf_idf,tf_idf)

x = data[data.title == "Your Mom's House with Christina P. and Tom Segura"].index[0]
x1 = data[data.title == "Your Mom's House with Christina P. and Tom Segura"]


feature_names = tf.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=tf_idf[len(similarity[x])-1]
print("HERE FIRST THO:")
print(data.tail(1)['description'])
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
print(df.sort_values(by=["tfidf"],ascending=False))


similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
for i in similar_idx:
    print(similarity[x][i], '-', data.title[i], '-', data.description[i], '\n - ', data.categories[i], ' - \n')
print('Original - ' + data.description[x] + '\n - ' + data.categories[x] + ' - \n')