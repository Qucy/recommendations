import os
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential

os.environ['MIN_TF_LOG_LEVEL'] = '2'

print(f"TF version is {tf.__version__}")

class NCF(Model):
    """
    Model class for NeuralCF
    """
    def __init__(self, unique_user_ids, unique_item_names, embedding_dim, mlp_activation, mlp_dropout):
        """
        init function for NCF model
        :param unique_user_ids: unique user ids in list format
        :param unique_item_names: unique item names in list format
        :param embedding_dim: embedding dimensions
        :param mlp_activation: MLP layer activation function
        :param mlp_dropout: MLP layer dropout
        """
        super(NCF, self).__init__()

        self.userMFEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=embedding_dim)]
        )

        self.userMLPEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=embedding_dim)]
        )

        self.itemMFEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_item_names, mask_token=None),
            layers.Embedding(input_dim=len(unique_item_names) + 1, output_dim=embedding_dim)]
        )

        self.itemMLPEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_item_names, mask_token=None),
            layers.Embedding(input_dim=len(unique_item_names) + 1, output_dim=embedding_dim)]
        )

        self.MLP = Sequential([
            layers.Dense(embedding_dim * 2, activation=mlp_activation),
            layers.Dropout(mlp_dropout),
            layers.Dense(embedding_dim , activation=mlp_activation),
            layers.Dropout(mlp_dropout),
            layers.Dense(embedding_dim // 2, activation=mlp_activation),
            layers.Dropout(mlp_dropout)]
        )

        self.logits_layer = layers.Dense(1)


    def call(self, inputs):
        user_id, item_name = inputs
        userMFEmbedding = self.userMFEmbedding(user_id) # (b, embedding_dim)
        userMLPmbedding = self.userMLPEmbedding(user_id) # (b, embedding_dim)
        itemMFEmbedding = self.itemMFEmbedding(item_name) # (b, embedding_dim)
        itemMLPmbedding = self.itemMLPEmbedding(item_name) # (b, embedding_dim)
        # left tower MF user vector * MF item vector
        left = tf.math.multiply(userMFEmbedding, itemMFEmbedding)  # (b, embedding_dim)
        # right tower MLP user vector concat with MLP item vector
        right = tf.concat([userMLPmbedding, itemMLPmbedding], axis=-1) # (b, 2 * embedding_dim)
        # go through MLP layer
        right = self.MLP(right)  # (b, embedding_dim // 2)
        # concat left and right tower
        left_right = tf.concat([left, right], axis=-1)
        # go through logits layer
        output = self.logits_layer(left_right)

        return output


if __name__ == "__main__":

    # test forward pass
    unique_user_ids = [str(i) for i in range(100)]
    unique_item_names = [str(i) for i in range(100)]
    embedding_dim = 32
    mlp_activation = 'relu'
    mlp_dropout = .2
    # fake input
    user_ids = ['1', '2']
    item_names = ['1', '2']
    # init model
    model = NCF(unique_user_ids, unique_item_names, embedding_dim, mlp_activation, mlp_dropout)
    prediction = model((user_ids, item_names))
    print(prediction)
