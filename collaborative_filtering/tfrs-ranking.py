from typing import Dict, Text
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow.keras import Sequential, layers, Model



# init variable
random_seed = 42
# setup random seed
tf.random.set_seed(random_seed)

# loading MovieLens 100K datasets
ratings = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

print(f"Total have {len(ratings)} ratings.")

#================= rating data =================
# {'bucketized_user_age': 45.0,
#  'movie_genres': array([7], dtype=int64),
#  'movie_id': b'357',
#  'movie_title': b"One Flew Over the Cuckoo's Nest (1975)",
#  'raw_user_age': 46.0,
#  'timestamp': 879024327,
#  'user_gender': True,
#  'user_id': b'138',
#  'user_occupation_label': 4,
#  'user_occupation_text': b'doctor',
#  'user_rating': 4.0,
#  'user_zip_code': b'53211'}


# get movie title to make embedding
movies = ratings.map(lambda x: x["movie_title"])
movie_titles = np.unique(np.concatenate(list(movies.batch(100_000))))
# get userID to make user embedding
userIDs = ratings.map(lambda x: x["user_id"])
userIDs = np.unique(np.concatenate(list(userIDs.batch(100_000))))
# print
print(f"Dataset have total unique {len(userIDs)} users and {len(movie_titles)} movies")

# prepare training and testing data
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000).batch(8192).cache()
test = shuffled.skip(80_000).take(20_000).batch(4096).cache()


# model class
class RankingModel(Model):

    def __init__(self, unique_user_ids, unique_movie_titles, embedding_dim):
        super(RankingModel, self).__init__()
        self.userEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=embedding_dim)
        ])
        self.movieEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            layers.Embedding(input_dim=len(unique_movie_titles) + 1, output_dim=embedding_dim)
        ])
        self.MLP = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1),
        ])

    def call(self, inputs):
        userID, movie_title = inputs
        userEmbedding = self.userEmbedding(userID)
        movieEmbedding = self.movieEmbedding(movie_title)
        output = self.MLP(tf.concat([userEmbedding, movieEmbedding], axis=-1))
        return output


class MovielensModel(tfrs.models.Model):

  def __init__(self, unique_user_ids, unique_movie_titles, embedding_dim):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids, unique_movie_titles, embedding_dim)
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model((features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    rating_predictions = self(features)
    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)


model = MovielensModel(userIDs, movie_titles, 32)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train, epochs=3)
model.evaluate(test, return_dict=True)
tf.saved_model.save(model, "./model/tfrs-ranking")
loaded = tf.saved_model.load("./model/tfrs-ranking")

prediction = loaded({"user_id": np.array(["42"]), "movie_title": ["Speed (1994)"]}).numpy()

print(prediction)





