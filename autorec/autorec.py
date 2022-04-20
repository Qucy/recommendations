import os
import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import tensorflow_datasets as tfds

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

# rating file
filename = 'rating.csv'
matrix_filename = 'score.npy'


# if rate is already exist then skip loading data from tfds
if not os.path.isfile(filename):
    # Ratings data.
    ratings = tfds.load("movielens/100k-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/100k-movies", split="train")

    # check how data looks like
    rating = ratings.take(1).as_numpy_iterator().next()
    pprint.pprint(rating)

    movie = movies.take(1).as_numpy_iterator().next()
    pprint.pprint(movie)

    # create dataframe to store user & movie ratings
    rating_df = pd.DataFrame()

    # let's take 1000 rates as example to generate a score vector for item
    for r in ratings.take(1000).as_numpy_iterator():
        user_id = [int(r['user_id'])]
        movie_id = [int(r['movie_id'])]
        user_rating = [r['user_rating']]
        d = {'user_id':user_id, 'movie_id':movie_id, 'user_rating':user_rating}
        rating_df = pd.concat([rating_df, pd.DataFrame(data=d)], axis=0)
    # save to current folder
    rating_df.to_csv(filename, index=False)
else:
    rating_df = pd.read_csv(filename)

# calc number users and movies
num_of_users = rating_df['user_id'].nunique()
num_of_movies = rating_df['movie_id'].nunique()

# score matrix is already exist then load it otherwise generate it
if not os.path.isfile(matrix_filename):
    # create a matrix with shape (num_of_users, num_of_movies) to store scores
    score_matrix = np.zeros((num_of_users, num_of_movies))
    # fill score to matrix
    for uidx, user in enumerate(rating_df['user_id'].unique()):
        print(f"start to handle user {user}-{uidx}")
        for midx, movie in enumerate(rating_df['movie_id'].unique()):
            score = rating_df.loc[(rating_df['user_id'] == user) & (rating_df['movie_id'] == movie)]
            if len(score) > 0:
                score_matrix[uidx][midx] = score['user_rating'].values[0]
    # save score matrix
    np.save(matrix_filename, score_matrix)
else:
    score_matrix = np.load(matrix_filename)

print(score_matrix.shape)


class AutoRec(Model):
    
    def __init__(self, hidden_dim, output_dim):
        super(AutoRec, self).__init__()
        self.l2 = tf.keras.regularizers.L2()
        self.hidden_layer = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=self.l2)
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        x = self.output_layer(x)
        return x


# select 1 movie score
x = score_matrix[:, 0]
x = x.reshape((1, 513))
# construct dataset, for auto-encoder input is x and target is also x
ds_train = tf.data.Dataset.from_tensor_slices((x, x))
ds_train = ds_train.cache().batch(1).prefetch(buffer_size=tf.data.AUTOTUNE)
# init model
autoRec = AutoRec(10, 513)
# compile model
autoRec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError()])
# fit model
autoRec.fit(ds_train, epochs=10)









