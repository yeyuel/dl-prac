#!/usr/bin/env python
# coding: utf-8

# In[2]:
import tensorflow as tf
from tensorflow import data
import numpy as np
import shutil
import math
from datetime import datetime
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2

from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn import make_export_strategy

print(tf.__version__)


# In[3]:


MODEL_NAME = 'class-model-02'

TRAIN_DATA_FILES_PATTERN = './data/classification/train-*.csv'
VALID_DATA_FILES_PATTERN = './data/classification/valid-*.csv'
TEST_DATA_FILES_PATTERN = './data/classification/test-*.csv'

RESUME_TRAINING = False
PROCESS_FEATURES = True
EXTEND_FEATURE_COLUMNS = True
MULTI_THREADING = True


# In[16]:


HEADER = ['key','x','y','alpha','beta','target']
HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], ['NA']]

NUMERIC_FEATURE_NAMES = ['x', 'y']  
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha':['ax01', 'ax02'], 'beta':['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

TARGET_NAME = 'target'
TARGET_LABELS = ['positive', 'negative']
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})

print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
print("Target: {} - labels: {}".format(TARGET_NAME, TARGET_LABELS))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))


# In[6]:


def parse_csv_row(csv_row):
    columns = tf.decode_csv(csv_row, record_defaults=HEADER_DEFAULTS)
    features = dict(zip(HEADER, columns))
    
    for column in UNUSED_FEATURE_NAMES:
        features.pop(column)
        
    target = features.pop(TARGET_NAME)
    return features, target

def process_features(features):
    features["x_2"] = tf.square(features["x"])
    features["y_2"] = tf.square(features["y"])
    features["xy"] = tf.multiply(features["x"], features["y"])
    features["dist_xy"] = tf.sqrt(tf.squared_difference(features["x"], features["y"]))
    
    return features


# In[7]:


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(TARGET_LABELS))
    return table.lookup(label_string_tensor)

def csv_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
                skip_header_lines=0,
                num_epochs=None,
                batch_size=200):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")
    
    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TextLineDataset(filenames=file_names)
    
    dataset = dataset.skip(skip_header_lines)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row))
    
    if PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target))
    
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
                              
    features, target = iterator.get_next()
    return features, parse_label_column(target)


# In[8]:


features, target = csv_input_fn(files_name_pattern="")
print("Feature read from CSV: {}".format(list(features.keys())))
print("Target read from CSV: {}".format(target))


# In[9]:


def extend_feature_columns(feature_columns, hparams):
    
    num_buckets = hparams.num_buckets
    embedding_size = hparams.embedding_size

    buckets = np.linspace(-3, 3, num_buckets).tolist()

    alpha_X_beta = tf.feature_column.crossed_column(
            [feature_columns['alpha'], feature_columns['beta']], 4)

    x_bucketized = tf.feature_column.bucketized_column(
            feature_columns['x'], boundaries=buckets)

    y_bucketized = tf.feature_column.bucketized_column(
            feature_columns['y'], boundaries=buckets)

    x_bucketized_X_y_bucketized = tf.feature_column.crossed_column(
           [x_bucketized, y_bucketized], num_buckets**2)

    x_bucketized_X_y_bucketized_embedded = tf.feature_column.embedding_column(
            x_bucketized_X_y_bucketized, dimension=embedding_size)


    feature_columns['alpha_X_beta'] = alpha_X_beta
    feature_columns['x_bucketized_X_y_bucketized'] = x_bucketized_X_y_bucketized
    feature_columns['x_bucketized_X_y_bucketized_embedded'] = x_bucketized_X_y_bucketized_embedded
    
    return feature_columns


def get_feature_columns(hparams):
    
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = ['x_2', 'y_2', 'xy', 'dist_xy']
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES.copy() 
    
    if PROCESS_FEATURES:
        all_numeric_feature_names += CONSTRUCTED_NUMERIC_FEATURES_NAMES

    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in all_numeric_feature_names}

    categorical_column_with_vocabulary =         {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}
        
    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)
    
    if EXTEND_FEATURE_COLUMNS:
        feature_columns = extend_feature_columns(feature_columns, hparams)
        
    return feature_columns

feature_columns = get_feature_columns(tf.contrib.training.HParams(num_buckets=5,embedding_size=3))
print("Feature Columns: {}".format(feature_columns))


# In[25]:


def get_input_layer_feature_columns(hparams):
    
    feature_columns = list(get_feature_columns(hparams).values())
    
    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column_v2.NumericColumn) |
                              isinstance(column, feature_column_v2.EmbeddingColumn),
               feature_columns
        )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column_v2.VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column_v2.BucketizedColumn),
                   feature_columns)
    )
    

    indicator_columns = list(
            map(lambda column: tf.feature_column.indicator_column(column),
                categorical_columns)
    )
    
    return dense_columns+indicator_columns


# In[20]:


def classification_model_fn(features, labels, mode, params):
    hidden_units = params.hidden_units
    output_layer_size = len(TARGET_LABELS)
    
    feature_columns = get_input_layer_feature_columns(hparams)

    input_layer = tf.feature_column.input_layer(features=features,
                                               feature_columns=feature_columns)
    hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                           layer=tf.contrib.layers.fully_connected,
                                           stack_args=hidden_units)
    logits = tf.layers.dense(inputs=hidden_layers,
                            units=output_layer_size)
    
    output = tf.squeeze(logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, -1)
        
        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode,
                                         predictions=predictions,
                                         export_outputs=export_outputs)
    
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
    )
    tf.summary.scalar('loss', loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer()

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss, 
                                          train_op=train_op)
    
    
    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )
        
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities)
        }
        
        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode, 
                                          loss=loss, 
                                          eval_metric_ops=eval_metric_ops)

def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=classification_model_fn,
                                  params=hparams, 
                                  config=run_config)
    
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator


# In[21]:


def generate_experiment_fn(**experiment_args):

    def _experiment_fn(run_config, hparams):

        train_input_fn = lambda: csv_input_fn(
            TRAIN_DATA_FILES_PATTERN,
            mode = tf.estimator.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.batch_size
        )

        eval_input_fn = lambda: csv_input_fn(
            VALID_DATA_FILES_PATTERN,
            mode=tf.estimator.ModeKeys.EVAL,
            num_epochs=1,
            batch_size=hparams.batch_size
        )

        estimator = create_estimator(run_config, hparams)

        return tf.contrib.learn.Experiment(
            estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_steps=None,
            **experiment_args
        )

    return _experiment_fn


# In[22]:


TRAIN_SIZE = 12000
NUM_EPOCHS = 1 #1000
BATCH_SIZE = 64
NUM_EVAL = 1 #10
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))

hparams  = tf.contrib.training.HParams(
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
    hidden_units=[16, 12, 8],
    num_buckets = 6,
    embedding_size = 3,
    dropout_prob = 0.001)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.contrib.learn.RunConfig(
    save_checkpoints_steps=CHECKPOINT_STEPS,
    tf_random_seed=19830610,
    model_dir=model_dir
)

print(hparams)
print("Model Directory:", run_config.MODEL_DIR)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:",TRAIN_SIZE/BATCH_SIZE)
print("Total Steps:", (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)
print("Required Evaluation Steps:", NUM_EVAL) 
print("That is 1 evaluation step after each",NUM_EPOCHS/NUM_EVAL," epochs")
print("Save Checkpoint After",CHECKPOINT_STEPS,"steps")


# In[23]:


def json_serving_input_fn():
    
    receiver_tensor = {}

    for feature_name in FEATURE_NAMES:
        dtype = tf.float32 if feature_name in NUMERIC_FEATURE_NAMES else tf.string
        receiver_tensor[feature_name] = tf.placeholder(shape=[None], dtype=dtype)

    if PROCESS_FEATURES:
        features = process_features(receiver_tensor)

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)


# In[26]:


if not RESUME_TRAINING:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print("Resuming training...") 


tf.logging.set_verbosity(tf.logging.INFO)
 
time_start = datetime.utcnow() 
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................") 

learn_runner.run(
    experiment_fn=generate_experiment_fn(

        export_strategies=[
            make_export_strategy(
            json_serving_input_fn,
            exports_to_keep=1,
            as_text=True
            )
        ]
    ),
    run_config=run_config,
    schedule="train_and_evaluate",
    hparams=hparams
)

time_end = datetime.utcnow() 
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

