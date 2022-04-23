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
movies = tfds.load("movielens/100k-movies", split="train")

print(f"Total have {len(ratings)} ratings and {len(movies)} movies")

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

#================= movie data =================
# {'movie_genres': array([4], dtype=int64),
#  'movie_id': b'1681',
#  'movie_title': b'You So Crazy (1994)'}

# preprocess data only keep interaction between movie and user
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
# get movie title to make embedding
movies = movies.map(lambda x: x["movie_title"])
movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
# get userID to make user embedding
userIDs = ratings.map(lambda x: x["user_id"])
userIDs = np.unique(np.concatenate(list(userIDs.batch(100_000))))

print(f"Dataset have total unique {len(userIDs)} users and {len(movie_titles)} movies")

# shuffle data
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
# split data
training = shuffled.take(80_000).batch(8192).cache()
testing = shuffled.skip(80_000).take(20_000).batch(4096).cache()


# construct model
class NeuralCF(tfrs.Model):

    def __init__(self, unique_user_ids, unique_movie_titles, embedding_dim):
        """
        init function for NeuralCF
        :param unique_user_ids: unique user ids
        :param unique_movie_titles: unique movie titles
        :param embedding_dim: embedding dimension
        """
        super(NeuralCF, self).__init__()
        self.user_model : Model = Sequential([
            layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=embedding_dim)
        ])
        self.movie_model : Model = Sequential([
            layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            layers.Embedding(input_dim=len(unique_movie_titles) + 1, output_dim=embedding_dim)
        ])
        self.task : layers.Layer  = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(self.movie_model)))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        """
        Compute loss function for NeuralCF
        :param features: training features
        :param training: is training or not
        :return:
        """
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


# init model
neuralCF = NeuralCF(unique_user_ids=userIDs, unique_movie_titles=movie_titles, embedding_dim=32)
neuralCF.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
# fit model
neuralCF.fit(training, epochs=3)
# evaluate model
neuralCF.evaluate(testing, return_dict=True)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(neuralCF.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
    tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(neuralCF.movie_model)))
)
# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = "./model/neuralCF"
  # Save the index.
  tf.saved_model.save(index, path)
  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)
  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])
  print(f"Recommendations: {titles[0][:3]}")

