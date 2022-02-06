#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[3]:


import math
import os
import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.python import data

print ("Tensorflow: {}".format(tf.__version__))

SEED = 19831060


# In[6]:


DATA_DIR='./data/recommend'
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'ml-latest-small/ratings.csv')


# In[8]:


ratings_data = pd.read_csv(TRAIN_DATA_FILE)
ratings_data.describe()


# In[10]:


ratings_data.head()


# In[11]:


movies_data = pd.read_csv(os.path.join(DATA_DIR, 'ml-latest-small/movies.csv'))
movies_data.head()


# In[12]:


HEADER = ['userId', 'movieId', 'rating', 'timestamp']
HEADER_DEAFAULTS = [0, 0, 0.0, 0]
TARGET_NAME = 'rating'
num_users = ratings_data.userId.max()
num_movies = movies_data.movieId.max()
num_users, num_movies


# In[13]:


def make_input_fn(file_pattern, 
                 batch_size,
                 num_epochs,
                 mode=tf.estimator.ModeKeys.EVAL):
    def _input_fn():
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            column_names=HEADER,
            column_defaults=HEADER_DEAFAULTS,
            label_name=TARGET_NAME,
            field_delim=',',
            use_quote_delim=True,
            header=True,
            num_epochs=num_epochs,
            shuffle=(mode == tf.estimator.ModeKeys.TRAIN)
        )
        return dataset
    return _input_fn


# In[14]:


def create_feature_columns(embedding_size):
    feature_columns = []
    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                'userId', num_buckets=num_users + 1
            ),
            embedding_size
        )
    )
    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                'movieId', num_buckets=num_movies + 1
            ),
            embedding_size
        )
    )
    return feature_columns


# In[67]:


def model_fn(features, labels, mode, params):
    feature_columns = create_feature_columns(params.embedding_size)
    user_layer = tf.feature_column.input_layer(
        features={'userId': features['userId']},
        feature_columns=[feature_columns[0]]
    )     
    predictions = None
    export_outputs = None
    loss = None
    train_op = None
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'user_embedding': user_layer}
        export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    else:
        movie_layer = tf.feature_column.input_layer(
            features={'movieId': features['movieId']},
            feature_columns=[feature_columns[1]]
        )
        dot_product = tf.keras.layers.Dot(axes=1)([user_layer, movie_layer])
        logits = tf.clip_by_value(clip_value_min=0, clip_value_max=5, t=dot_product)
        loss = tf.losses.mean_squared_error(labels, tf.squeeze(logits))
        train_op = tf.train.FtrlOptimizer(params.learning_rate).minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        loss = tf.losses.mean_squared_error(labels, tf.squeeze(logits))
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs=export_outputs,
        loss=loss,
        train_op=train_op
    )


# In[68]:


def create_estimator(params, run_config):
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config
    )
    return estimator


# In[69]:


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
        input_fn = eval_input_fn,
        steps=None,
        start_delay_secs=0,
        throttle_secs=params.eval_throttle_secs
    )
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    if tf.gfile.Exists(run_config.MODEL_DIR):
        print("Removing previous artefacts...")
        tf.gfile.DeleteRecursively(run_config.MODEL_DIR)
    
    print ('')
    estimator = create_estimator(params, run_config)
    print ('')
    
    
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


# In[107]:


MODELS_LOCATION = 'models/movieles'
MODEL_NAME = 'recommender_01'
model_dir = os.path.join(MODELS_LOCATION, MODEL_NAME)

params = tf.contrib.training.HParams(
    batch_size=265,
    training_steps=100000,
    learning_rate=0.1,
    embedding_size=16,
    eval_throttle_secs=0,
)

run_config = tf.estimator.RunConfig(
    tf_random_seed=SEED,
    save_checkpoints_steps=100000,
    keep_checkpoint_max=3,
    model_dir=model_dir,
)

estimator = train_and_evaluate_experiment(params, run_config)


# In[108]:


def find_embedding_tensor():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.ckpt-100000.meta'))
        saver.restore(sess, os.path.join(model_dir, 'model.ckpt-100000'))
        graph = tf.get_default_graph()
        trainable_tensors = map(str, graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        for tensor in set(trainable_tensors):
            print(tensor)
find_embedding_tensor()


# In[110]:


def extract_embeddings():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.ckpt-100000.meta'))
        saver.restore(sess, os.path.join(model_dir, 'model.ckpt-100000'))
        graph = tf.get_default_graph()
        weights_tensor = graph.get_tensor_by_name('input_layer_1/movieId_embedding/embedding_weights:0')
        weights = np.array(sess.run(weights_tensor))
    embeddings = {}
    for i in range(weights.shape[0]):
        embeddings[i] = weights[i]
    return embeddings
embeddings = extract_embeddings()
embeddings[0], len(embeddings)


# In[111]:


from annoy import AnnoyIndex

def build_embeddings_index(num_trees):
    total_items = 0
    annoy_index = AnnoyIndex(params.embedding_size, metric='angular')
    for item_id in embeddings.keys():
        annoy_index.add_item(item_id, embeddings[item_id])
        total_items += 1
    print ("{} items where added to the index".format(total_items))
    annoy_index.build(n_trees=num_trees)
    print("Index is built")
    return annoy_index

index = build_embeddings_index(100)


# In[112]:


frequent_movie_ids = list(ratings_data.movieId.value_counts().index[:15])
frequent_movie_ids


# In[113]:


movies_data[movies_data['movieId'].isin(frequent_movie_ids)]


# In[114]:


def get_similar_movies(movie_id, num_matches=5):
    similar_movie_ids = index.get_nns_by_item(
        movie_id,
        num_matches,
        search_k=1,
        include_distances=False
    )
    similar_movies = movies_data[movies_data['movieId'].isin(similar_movie_ids)].title
    return similar_movies

for movie_id in frequent_movie_ids:
    movie_title = movies_data[movies_data['movieId'] == movie_id].title.values[0]
    print ("Movie: {}".format(movie_title))
    similar_movies = get_similar_movies(movie_id)
    print ("Similar Movies:")
    print (similar_movies)
    print ("--------------------------------------")


# In[115]:


def make_serving_input_receiver_fn():
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        {'userId': tf.placeholder(shape=[None], dtype=tf.int32)}
    )
export_dir = os.path.join(model_dir, 'export')

if tf.gfile.Exists(export_dir):
    tf.gfile.DeleteRecursively(export_dir)

estimator.export_saved_model(
    export_dir_base=export_dir,
    serving_input_receiver_fn=make_serving_input_receiver_fn()
)


# In[119]:


import os
import time

export_dir = os.path.join(model_dir, 'export')
saved_model_dir = os.path.join(
    export_dir, [f for f in os.listdir(export_dir) if f.isdigit()][0]
)
print(saved_model_dir)

predictor_fn = tf.contrib.predictor.from_saved_model(
    export_dir=saved_model_dir
)
start = time.time()
output = predictor_fn({"userId": [1]})
print('Elapse:{}'.format(time.time() - start))
print(output)


# In[117]:


def recommend_new_movies(userId, num_recommendations=5):
    watched_movie_ids = list(ratings_data[ratings_data['userId'] == userId]['movieId'])
    user_embedding = predictor_fn({'userId': [userId]})['user_embedding'][0]
    similar_movie_ids = index.get_nns_by_vector(
        user_embedding, num_recommendations + len(watched_movie_ids), search_k = -1, include_distances=False
    )
    recommended_movie_ids = set(similar_movie_ids) - set(watched_movie_ids)
    similar_movies = movies_data[movies_data['movieId'].isin(recommended_movie_ids)].title
    return similar_movies

frequent_user_ids = list((ratings_data.userId.value_counts().index[-350:]))[:5] 
for user_id in frequent_user_ids:
    print ("User: {}".format(user_id))
    recommended = recommend_new_movies(user_id)
    print ("Recommend movies: {}".format(len(recommended)))
    print (recommended)
    print ("--------------------------------------")

