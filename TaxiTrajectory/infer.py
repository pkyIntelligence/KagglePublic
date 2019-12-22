import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
import h5py

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)


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


input_dir = '../input/'

with h5py.File(input_dir + 'taxitesth5/emb_szs.h5', 'r') as f:
    emb_szs = np.array(f['embedding_sizes'])

with h5py.File(input_dir + 'taxitesth5/cluster_centers3.h5', 'r') as f:
    cluster_centers = np.array(f['cc'])

cc = np.expand_dims(cluster_centers, axis=0)

model = TaxiModel(emb_szs, 23, cc.shape[1])
model.compile(optimizer='sgd', loss=EuclDist())

X_dummy = {'x_cat': np.ones(shape=(1, 19)), 'x_cont': np.random.normal(size=(1, 23)), 'cc': cc}
y_dummy = np.random.normal(size=(1, 2))

model.fit(x=X_dummy, y=y_dummy)

model.load_weights(input_dir + 'taxi-model/model_dir/low_cat_gpu_model_weights.tf')

test_df = pd.read_csv(input_dir + 'taxitesth5/taxi_test.csv')

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

x_cat = test_df[emb_token_fields_list].values
x_cont = test_df[cont_normed_fields_list].values
num_obs = x_cat.shape[0]
cc = np.repeat(cc, num_obs, axis=0)

y_pred = model.predict(x={'x_cat': x_cat, 'x_cont': x_cont, 'cc': cc})

test_df['LATITUDE'] = y_pred[:, 1]
test_df['LONGITUDE'] = y_pred[:, 0]
sub_df = test_df[['TRIP_ID', 'LATITUDE', 'LONGITUDE']]

sub_df.to_csv('submission.csv', index=False)
