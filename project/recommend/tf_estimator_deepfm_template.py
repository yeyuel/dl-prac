from tensorflow.python import data
from tensorflow.python import feature_column
from tensorflow.python import train
from tensorflow.python import logging
from tensorflow.python import gfile
import tensorflow.keras as keras
import tf_slim as slim
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from annoy import AnnoyIndex

from tensorflow_estimator.python.estimator.run_config import RunConfig

print("Tensorflow: {}".format(tf.__version__))

"""
constant definition
"""
DATA_DIR = '../tutorial/data/recommend/ml-latest-small'
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'training_samples/part-00000-cbb6a3ce-9d1f-4946-b71d-633de031b904-c000.csv')
MOVIE_META_FILE = os.path.join(DATA_DIR, 'test_samples/part-00000-d0fab76a-7cae-4607-938f-e7ab8b1453fe-c000.csv')
MODELS_LOCATION = 'models/movie_lens'
MODEL_NAME = 'deep_fm_01'
MODEL_DIR = os.path.join(MODELS_LOCATION, MODEL_NAME)
EXPORT_DIR = os.path.join(MODEL_DIR, 'export')

# minimal csv features
FIELDS = {'movieId': 0,
          'userId': 0,
          'rating': 3.5,
          'timestamp': 0,
          'label': 0,
          'releaseYear': 0,
          'movieGenre1': "NA",
          'movieGenre2': "NA",
          'movieGenre3': "NA",
          'movieRatingCount': 0.0,
          'movieAvgRating': 0.0,
          'movieRatingStddev': 0.0,
          'userRatedMovie1': 0,
          'userRatedMovie2': 0,
          'userRatedMovie3': 0,
          'userRatedMovie4': 0,
          'userRatedMovie5': 0,
          'userRatingCount': 0,
          'userAvgReleaseYear': 0.0,
          'userReleaseYearStddev': 0.0,
          'userAvgRating': 0.0,
          'userRatingStddev': 0.0,
          'userGenre1': 'NA',
          'userGenre2': 'NA',
          'userGenre3': 'NA',
          'userGenre4': 'NA',
          'userGenre5': 'NA'}
HEADER = FIELDS.keys()
HEADER_DEFAULTS = FIELDS.values()

GENRE_VOCAB = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical', 'NA']

# label
TARGET_NAME = 'label'

# training params
PARAMS = tf.contrib.training.HParams(
    batch_size=265,
    training_steps=100000,
    learning_rate=0.1,
    embedding_size=16,
    eval_throttle_secs=0,
)

RUN_CONFIG: RunConfig = tf.estimator.RunConfig(
    tf_random_seed=19831060,
    save_checkpoints_steps=100000,
    keep_checkpoint_max=3,
    model_dir=MODEL_DIR,
)

"""
data process
"""
movies_data = pd.read_csv(MOVIE_META_FILE)
ratings_data = pd.read_csv(TRAIN_DATA_FILE)
num_users = ratings_data.userId.max()
num_movies = movies_data.movieId.max()
print('user id max: {}, movie id max: {}'.format(num_users, num_movies))


def make_input_fn(file_pattern,
                  batch_size,
                  num_epochs,
                  mode=tf.estimator.ModeKeys.EVAL):  # coverage three modes
    def _input_fn():
        dataset = data.experimental.make_csv_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            column_names=HEADER,
            column_defaults=HEADER_DEFAULTS,
            label_name=TARGET_NAME,
            field_delim=',',
            use_quote_delim=True,
            header=True,
            num_epochs=num_epochs,
            shuffle=(mode == tf.estimator.ModeKeys.TRAIN)
        )
        return dataset

    return _input_fn


"""
input to feature column
"""


def create_feature_columns(features, embedding_size):
    """
    Feature column construction
    :param features:
    :param embedding_size:
    :return: list of feature columns
    """
    embeddings = []
    user_emb = feature_column.embedding_column(
        feature_column.categorical_column_with_identity(
            'userId', num_buckets=num_users + 1
        ),
        embedding_size
    )
    embeddings.append(feature_column.input_layer(features, [user_emb]))
    item_emb = feature_column.embedding_column(
        feature_column.categorical_column_with_identity(
            'movieId', num_buckets=num_movies + 1
        ),
        embedding_size
    )
    embeddings.append(feature_column.input_layer(features, [item_emb]))
    genre_columns = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                     'movieGenre1', 'movieGenre2', 'movieGenre3']
    for genre_column in genre_columns:
        emb = feature_column.embedding_column(
            feature_column.categorical_column_with_vocabulary_list(
                key=genre_column,
                vocabulary_list=GENRE_VOCAB
            ),
            embedding_size
        )
        embeddings.append(feature_column.input_layer(features, [emb]))
    denses = [
        feature_column.numeric_column('releaseYear'),
        feature_column.numeric_column('movieRatingCount'),
        feature_column.numeric_column('movieAvgRating'),
        feature_column.numeric_column('movieRatingStddev'),
        feature_column.numeric_column('userRatingCount'),
        feature_column.numeric_column('userAvgRating'),
        feature_column.numeric_column('userRatingStddev')
    ]
    denses = [feature_column.input_layer(features, [num]) for num in denses]
    return denses, embeddings


"""
model function definition
"""


def combine_embedding_dense(embeddings, denses):
    embeddings = tf.concat(embeddings, 1)
    dense = tf.concat(denses, 1)
    return tf.concat([embeddings, dense], 1)


def fm(embeddings):
    embeddings = tf.concat(embeddings, 1)
    sum_sqrt = tf.square(tf.reduce_sum(embeddings, 1))
    sqrt_sum = tf.reduce_sum(tf.square(embeddings), 1)
    value = 0.5 * tf.subtract(sum_sqrt, sqrt_sum)
    value = tf.expand_dims(value, [1])
    linear = keras.layers.Dense(units=1, name='linear_output')(embeddings)
    # linear = tf.layers.dense(inputs=embeddings, units=1, name='linear_output')
    return tf.concat([value, linear], -1)


def deep_fm(embeddings, denses):
    inputs = combine_embedding_dense(embeddings, denses)
    deep_layer = slim.stack(inputs, slim.fully_connected, [128, 64, 32], scope='fc')
    # deep_layer = tf.layers.stack(input, tf.layers.full_connected, [128, 64, 32], scope='fc')
    fm_layer = fm(embeddings)
    last_layer = tf.concat([deep_layer, fm_layer], 1)
    logits = keras.layers.Dense(units=1, activation='sigmoid', name='logits')(last_layer)
    # logits = tf.layers.dense(inputs=last_layer, units=1, name='logits')
    return logits


def model_fn(features, labels, mode, params):
    """
    model function definition
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return: EstimatorSpec
    """
    denses, embeddings = create_feature_columns(features, params.embedding_size)
    logits = deep_fm(embeddings, denses)
    predictions = None
    export_outputs = None
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits,
                       'labels': labels}
        export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    else:
        loss = tf.losses.sigmoid_cross_entropy(labels, tf.squeeze(logits, axis=1))
        train_op = train.AdamOptimizer(params.learning_rate).minimize(
            loss=loss,
            global_step=train.get_global_step()
        )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs=export_outputs,
        loss=loss,
        train_op=train_op
    )


"""
estimator definition
"""


def create_estimator(params, run_config):
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config
    )
    return estimator


def train_and_evaluate_experiment(params, run_config):
    train_input_fn = make_input_fn(
        TRAIN_DATA_FILE,
        batch_size=params.batch_size,
        num_epochs=None,
        mode=tf.estimator.ModeKeys.TRAIN
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.training_steps
    )

    eval_input_fn = make_input_fn(
        TRAIN_DATA_FILE,
        num_epochs=1,
        batch_size=params.batch_size,
    )

    eval_spec = tf.estimator.EvalSpec(
        name=datetime.utcnow().strftime("%H%M%S"),
        input_fn=eval_input_fn,
        steps=None,
        start_delay_secs=0,
        throttle_secs=params.eval_throttle_secs
    )

    logging.set_verbosity(logging.INFO)

    if gfile.Exists(run_config.model_dir):
        print("Removing previous artefacts...")
        gfile.DeleteRecursively(run_config.model_dir)

    print('')
    estimator = create_estimator(params, run_config)
    print('')

    time_start = datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

    return estimator


"""
training logic
"""
ncf_estimator = train_and_evaluate_experiment(PARAMS, RUN_CONFIG)

"""
model saving
"""


def make_serving_input_receiver_fn():
    """
    Build model input
    :return: model input dict
    """
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        {'userId': tf.placeholder(shape=[None], dtype=tf.int32)}
    )


# if gfile.Exists(EXPORT_DIR):
#     gfile.DeleteRecursively(EXPORT_DIR)
#
# ncf_estimator.export_saved_model(
#     export_dir_base=EXPORT_DIR,
#     serving_input_receiver_fn=make_serving_input_receiver_fn()
# )

"""
embedding inference
"""


# def find_embedding_tensor():
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, 'model.ckpt-100000.meta'))
#         saver.restore(sess, os.path.join(MODEL_DIR, 'model.ckpt-100000'))
#         graph = tf.get_default_graph()
#         trainable_tensors = map(str, graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
#         for tensor in set(trainable_tensors):
#             print(tensor)
#
#
# find_embedding_tensor()
#
#
# def extract_embeddings():
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, 'model.ckpt-100000.meta'))
#         saver.restore(sess, os.path.join(MODEL_DIR, 'model.ckpt-100000'))
#         graph = tf.get_default_graph()
#         weights_tensor = graph.get_tensor_by_name('input_layer_1/movieId_embedding/embedding_weights:0')
#         weights = np.array(sess.run(weights_tensor))
#     embeddings = {}
#     for i in range(weights.shape[0]):
#         embeddings[i] = weights[i]
#     return embeddings
#
#
# _embeddings = extract_embeddings()
#
# """
# Build embedding index
# """
#
#
# def build_embeddings_index(num_trees):
#     total_items = 0
#     annoy_index = AnnoyIndex(PARAMS.embedding_size, metric='angular')
#     for item_id in _embeddings.keys():
#         annoy_index.add_item(item_id, _embeddings[item_id])
#         total_items += 1
#     print("{} items where added to the index".format(total_items))
#     annoy_index.build(n_trees=num_trees)
#     print("Index is built")
#     return annoy_index
#
#
# index = build_embeddings_index(100)
#
# """
# evaluation similar items
# """
#
#
# def get_similar_movies(movie_id, num_matches=5):
#     similar_movie_ids = index.get_nns_by_item(
#         movie_id,
#         num_matches,
#         search_k=1,
#         include_distances=False
#     )
#     similar_movies = movies_data[movies_data['movieId'].isin(similar_movie_ids)].title
#     return similar_movies
#
#
# frequent_movie_ids = list(ratings_data.movieId.value_counts().index[:15])
#
# for _movie_id in frequent_movie_ids:
#     movie_title = movies_data[movies_data['movieId'] == _movie_id].title.values[0]
#     print("Movie: {}".format(movie_title))
#     _similar_movies = get_similar_movies(_movie_id)
#     print("Similar Movies:")
#     print(_similar_movies)
#     print("--------------------------------------")
#
# """
# prediction observation
# """
# saved_model_dir = os.path.join(
#     EXPORT_DIR, [f for f in os.listdir(EXPORT_DIR) if f.isdigit()][0]
# )
# print(saved_model_dir)
# predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir)
# start = time.time()
# output = predictor_fn({"userId": [1]})
# print('Elapse:{}'.format(time.time() - start))
# print(output)
#
#
# def recommend_new_movies(user_id, num_recommendations=5):
#     watched_movie_ids = list(ratings_data[ratings_data['userId'] == user_id]['movieId'])
#     user_embedding = predictor_fn({'userId': [user_id]})['user_embedding'][0]
#     similar_movie_ids = index.get_nns_by_vector(
#         user_embedding, num_recommendations + len(watched_movie_ids), search_k=-1, include_distances=False
#     )
#     recommended_movie_ids = set(similar_movie_ids) - set(watched_movie_ids)
#     recommended_movies = movies_data[movies_data['movieId'].isin(recommended_movie_ids)].title
#     return recommended_movies
#
#
# frequent_user_ids = list((ratings_data.userId.value_counts().index[-350:]))[:5]
# for _user_id in frequent_user_ids:
#     print("User: {}".format(_user_id))
#     recommended = recommend_new_movies(_user_id)
#     print("Recommend movies: {}".format(len(recommended)))
#     print(recommended)
#     print("--------------------------------------")

