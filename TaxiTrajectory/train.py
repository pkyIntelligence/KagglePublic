import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss

EPSILON = 1e-7

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

data_dir = '/kaggle/input/taxitesth5/'

class TaxiModel(Model):

    def __init__(self, emb_szs, n_cont, n_dest_clusters):
        """
        emb_szs a list of tuples (vs, ed), vs is vocab_size, ed is embedding_dimension
        """
        super(TaxiModel, self).__init__()
        # define construction/layers
        self._embeds = [Embedding(vs, ed) for vs, ed in emb_szs]
        self._n_emb = sum(ed for _, ed in emb_szs)
        self._n_cont = n_cont

        self._hidden_lyr = Dense(500, activation='relu')
        self._logit_lyr = Dense(n_dest_clusters, activation='softmax')
        self._n_dest_clusters = n_dest_clusters

    def call(self, inputs):
        x_embeds = inputs['x_cat']
        x_cont = inputs['x_cont']
        dest_clusters = inputs['cc']
        # broadcasted across batch dimension, shape = [batch, n_dest_clusters, (long, lat)]
        if self._n_emb != 0:
            x = [e(x_embeds[:, i]) for i, e in enumerate(self._embeds)]
            x = tf.concat(x, axis=1)
        if self._n_cont != 0:
            x = tf.concat([x, x_cont], axis=1)
        x = self._hidden_lyr(x)
        x = self._logit_lyr(x)  # x.shape = [batch, n_dest_clusters], is p(x) for any one batch example
        # (dest_clusters * x).shape = [batch, n_dest_clusters, (long, lat)]
        x = tf.expand_dims(x, axis=2) * dest_clusters
        x = tf.math.reduce_sum(x, axis=1)
        return x


class EuclDist(Loss):
    def __init__(self):
        super(EuclDist).__init__()
        self.name = 'euclidean_distance'
        self.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    def call(self, y_true, y_pred):
        return tf.math.sqrt(tf.math.square(y_true[:, 0] - y_pred[:, 0]) + tf.math.square(y_true[:, 1] - y_pred[:, 1]))


def taxi_data_gen():
    df = pd.read_csv(data_dir + 'taxi_train.csv')

    with h5py.File(data_dir + 'cluster_centers3.h5', 'r') as f:
        cluster_centers = np.array(f['cc'])

    emb_field_list = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
                      'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Hour', 'Minute', 'Second',
                      'TAXI_ID', 'ORIGIN_CALL', 'CALL_TYPE', 'ORIGIN_STAND']

    cont_field_list = ['Elapsed', 'Origin_Latitude', 'Origin_Longitude', 'first0_long', 'first0_lat', 'first1_long',
                       'first1_lat', 'first2_long', 'first2_lat', 'first3_long', 'first3_lat', 'first4_long',
                       'first4_lat', 'last1_long', 'last1_lat', 'last2_long', 'last2_lat', 'last3_long', 'last3_lat',
                       'last4_long', 'last4_lat', 'last5_long', 'last5_lat']

    emb_token_fields_list = []
    cont_normed_fields_list = []

    for emb_field in emb_field_list:
        emb_token_fields_list.append(emb_field + '_token_id')

    for cont_field in cont_field_list:
        cont_normed_fields_list.append(cont_field + '_normed')

    for index, row in df.iterrows():
        yield {'x_cat': row[emb_token_fields_list].values, 'x_cont': row[cont_normed_fields_list].values,
               'cc': cluster_centers}, \
              row[['target_long', 'target_lat']].values


with h5py.File(data_dir + 'cluster_centers3.h5', 'r') as f:
    cluster_centers = np.array(f['cc'])

with h5py.File(data_dir + 'emb_szs.h5', 'r') as f:
    emb_szs = np.array(f['embedding_sizes'])

ds = tf.data.Dataset.from_generator(taxi_data_gen,
                                    output_types=(
                                    {'x_cat': tf.float64, 'x_cont': tf.float64, 'cc': tf.float64}, tf.float64),
                                    output_shapes=(
                                    {'x_cat': (19,), 'x_cont': (23,), 'cc': cluster_centers.shape}, (2,)))

ds = ds.batch(32)

model = TaxiModel(emb_szs, 23, cluster_centers.shape[0])
model.compile(optimizer='sgd', loss=EuclDist())

model.fit(x=ds, epochs=3)

model.save_weights('model_dir/low_cat_gpu_model_weights.tf', save_format='tf')