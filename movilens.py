import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import json



import Image
import urllib, cStringIO
import matplotlib.pyplot as plt


response = requests.get('http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)')
print response.url.split('/')[-2]


#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
data = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['user_id' , 'movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')

"""
print(users.head())
print(users.shape)

print(data.head())
print(data.shape)

print(items.head())
print(items.shape)


print(users.info())
print(data.info())
print(items.info())

"""

"""
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

"""


nusers = users.shape[0]
nitems = items.shape[0]

"""
print(nusers)
print(nitems)

"""

#To create the user-item matrix
#Values in the user-item matrix are the ratings rating[i,j] is user i's rating of movie j

ratings = np.zeros((nusers, nitems))
for row in data.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]

"""
print(ratings)

"""

#To calculate sparsity of the user-item matrix
"""
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print 'Sparsity: {:4.2f}%'.format(sparsity)
"""


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test

train,test = train_test_split(ratings)
"""
print(train.shape)
print(test.shape)
"""

print(train[:10,:10])
print(test[:10,:10])

#Similarity between users by using cosine distance
def fast_similarity(ratings, kind='user', epsilon=1e-9):
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    print(norms.shape)
    return (sim /(norms.dot(norms.T)))


user_similarity = fast_similarity(train, kind='user')
print(user_similarity)

item_similarity = fast_similarity(train , kind='item')
print(item_similarity.shape)


def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

user_prediction = predict_fast_simple(train, user_similarity, kind='user')
item_prediction = predict_fast_simple(train , item_similarity, kind='item')

print 'User-based CF MSE: ' + str(get_mse(user_prediction, test))
print 'Item-based CF MSE: ' + str(get_mse(item_prediction, test))


#Only using top-k similar users to predict
def predict_topk(ratings , similarity , kind , k = 40):
    pred = np.zeros(ratings.shape)
    if kind=='user':
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]

            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))

    if kind == 'item':
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

    return pred




pred = predict_topk(train, user_similarity, kind='user', k=40)
print 'Top-k User-based CF MSE: ' + str(get_mse(pred, test))

pred = predict_topk(train, item_similarity, kind='item', k=40)
print 'Top-k Item-based CF MSE: ' + str(get_mse(pred, test))

#Using Pearson coefficient to measure similarity and then calculating among top k users
def predict_topk_nobias(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]
    if kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
        pred += item_bias[np.newaxis, :]

    return pred

user_pred = predict_topk_nobias(ratings , user_similarity)

print 'User-based CF MSE for Pearson: ' + str(get_mse(user_pred, test))

#Extracting movie poster from movie id using the tmdb API


def get_poster(imdb_url , base_url):
    response = requests.get(imdb_url)
    print(response.url)
    movie_id = response.url.split('/')[-2]
    print(movie_id)

    IMG_PATTERN = 'http://api.themoviedb.org/3/movie/{imdbid}/images?api_key={key}'
    r = requests.get(IMG_PATTERN.format(key=KEY,imdbid=movie_id))
    api_response = r.json()

    print("API response")
    print(api_response)

    file_path = api_response['posters'][0]['file_path']
    return base_url + file_path


CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
KEY = '519cfd04db19e108fba5cab32cca5238'
url = CONFIG_PATTERN.format(key=KEY)
r = requests.get(url)
config = r.json()

print(config)

base_url = config['images']['base_url']
sizes = config['images']['poster_sizes']

size = sizes[2]
print(size)

base_url = base_url + size

#Example to show the poster for the toy story movie
toy_story = 'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)'
url = get_poster(toy_story , base_url)
file = cStringIO.StringIO(urllib.urlopen(url).read())
img = Image.open(file)

plt.imshow(img)
plt.show()

idx_to_movie = {}
with open('u.item', 'r') as f:
    for line in f.readlines():
        info = line.split('|')
        idx_to_movie[int(info[0])-1] = info[4]

def top_k_movies(similarity, mapper, movie_idx, k=6):
    return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

idx = 0 # Toy Story
movies = top_k_movies(item_similarity, idx_to_movie, idx)
posters = tuple(Image(url=get_poster(movie, base_url)) for movie in movies)

display(*posters)
