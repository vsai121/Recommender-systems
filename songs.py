import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

user_data = pd.read_table('./lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])
user_profiles = pd.read_table('./lastfm-dataset-360K/usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])


print(user_data.head())
print(user_profiles.head())

#Drops a row in the data in which artist name is NA
if user_data['artist-name'].isnull().sum()>0:
    user_data = user_data.dropna(axis=0 , subset=['artist-name'])

#To find plays of each artist
artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']]
    )


print(artist_plays.head())

#Merge artist play data with user data
user_data_with_plays = user_data.merge(artist_plays , left_on = 'artist-name' , right_on = 'artist-name' , how='left')
print(user_data_with_plays.head())

#Choosing only popular artists to reduce dataset size and noise in data
popularity_threshold = 40000
user_data_popular_artists = user_data_with_plays.query('total_artist_plays >= @popularity_threshold')
print(user_data_popular_artists.head())

#To remove duplicate entries where user and artist names are same
print(user_data_popular_artists.shape)

datset = user_data_popular_artists.drop_duplicates(['users' , 'artist-name'])

print(datset.shape)

combined = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
usa_data = combined.query('country == \'United States\'')
usa_data.head()

print(usa_data.shape)
wide_artist_data = usa_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)


print('Constructed sparse matrix')

#Nearest neighbors using cosine metric
modelKNN = NearestNeighbors(metric='cosine' , algorithm='brute')
modelKNN.fit(sparse_matrix)

#Picking a random row to test nearest neighbors
query_index = np.random.choice(wide_artist_data.shape[0])
distances, indices = model_knn.kneighbors(wide_artist_data.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print 'Recommendations for {0}:\n'.format(wide_artist_data.index[query_index])
    else:
        print '{0}: {1}, with distance of {2}:'.format(i, wide_artist_data.index[indices.flatten()[i]], distances.flatten()[i])
