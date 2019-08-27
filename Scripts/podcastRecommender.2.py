import numpy as np
import pandas as pd 
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
import requests
from pprint import pprint
#print(os.listdir("./input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

podcasts = pd.read_csv('./input/podcasts.csv')

podcasts = podcasts.append({'uuid' : '3e95f8d1c93542609fada7afed2a1e34', 'title' : 'Howya Now with Matt and Pat' , 'image' : 'https://cdn-images-1.listennotes.com/podcasts/howya-now-with-matt-and-pat-matthew-barrett-9btYx07fmiS.300x300.jpg', 'description' : "Matt Barrett and Patrick Hession are the hosts of 'Howya Now with Matt and Pat', a podcast in which they deep dive into an array of topics that they find interesting or are passionate about. This can range from the meaning of life to diet and health, and everything in between with hopefully a bit of craic along the way. If you're enjoying the podcast follow our social media @howyanowpodcast on Instagram and Twitter, for updates about new episodes.", 'language' : 'English', 'categories' : 'Comedy | Society & Culture | Fitness | Health & Fitness', 'website' : 'https://howya-now.pinecast.co', 'author' : 'Matt and Pat', 'itunes_id' : '1460740322'}, ignore_index=True)

#Cleaning the data
podcasts = podcasts[podcasts.language == 'English']

podcasts = podcasts.dropna(subset=['description'])

podcasts = podcasts.drop_duplicates('itunes_id')

# Checking the length of the descriptions of each podcast
podcasts['description_length'] = [len(x.description.split()) for _, x in podcasts.iterrows()]

podcasts = podcasts[podcasts.description_length >= 20]

#Definining my favorite podcasts to ensure they are included in the sample
favorite_podcasts = ["Your Mom's House with Christina P. and Tom Segura", 'Howya Now with Matt and Pat', "VIEWS with David Dobrik and Jason Nash"]
favorites = podcasts[podcasts.title.isin(favorite_podcasts)]

#Taking a sample of 5000 podcasts from the dataset
podcasts = podcasts[~podcasts.isin(favorites)].sample(5000)
data = pd.concat([podcasts,favorites], sort = True).reset_index(drop = True)

#Casting the descriptions to strings to remove any issues with word analysis
data = data.astype({"description": str})

#Building a sparse matrix of occurences of bigram words
tf = TfidfVectorizer(analyzer = 'word', ngram_range= (1, 1), min_df = 0, stop_words = 'english')
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

feature_names = tf.get_feature_names()
 
#get tfidf vector for the podcast in question
first_document_vector=tf_idf[len(similarity[x])-2]
 
#take the top 3 scoring words and put them in an array
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
top3 = (df.sort_values(by=["tfidf"],ascending=False)).head(3)
top3array = []
top3array.append(top3.index[0])
top3array.append(top3.index[1])
top3array.append(top3.index[2])

for a in top3array:
    print(a)

top3UrlForm = str(top3array[0] + "%20" + top3array[1] + "%20" + top3array[2])
apiRequestUrl = str("https://listen-api.listennotes.com/api/v2/search?q=" + top3UrlForm + "&sort_by_date=0&type=podcast&offset=0&len_min=0&published_after=0&only_in=description&language=English&safe_mode=0")


response = requests.get(apiRequestUrl,
  headers={
    "X-ListenAPI-Key": "d1932d3198964e2891821c9013c9fba0",
  },
)

response.json()
pprint(response.json())


similar_idx = similarity[x].argsort(axis = 0)[-4:-1]
for i in similar_idx:
    print(similarity[x][i], '-', data.title[i], '-', data.description[i], '\n - ', data.categories[i], ' - \n')
print('Original - ' + data.description[x] + '\n - ' + data.categories[x] + ' - \n')