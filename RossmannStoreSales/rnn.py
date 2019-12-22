import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import glob
import os
import xarray as xr

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from typing import Iterable

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

EPSILON = 1e-7

data_dir = 'data/'

def if_none(a, b):
    # `a` if `a` is not None, otherwise `b`
    return b if a is None else a


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    # Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try:
            a = len(p)
        except:
            p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


class BatchDropLin(tf.keras.layers.Layer):
    """
    Convenience class for Batch Normalization -> Dropout -> Linear layers
    """

    def __init__(self, units, bn=True, p=0, act=None):
        super(BatchDropLin, self).__init__()

        self._bn = bn
        self._p = p
        self._act = act

        if bn:
            self._bn_layer = BatchNormalization()
        if p != 0:
            self._do_layer = Dropout(p)
        self._dense_layer = Dense(units)
        if act is not None:
            if act == 'relu':
                self._act_layer = ReLU()
            else:
                self._act = None

    def call(self, inputs, training):
        if self._bn:
            inputs = self._bn_layer(inputs, training)
        if self._p != 0:
            inputs = self._do_layer(inputs, training)
        x = self._dense_layer(inputs)
        if self._act is not None:
            x = self._act_layer(x)
        return x


class TabularModel(tf.keras.Model):

    def __init__(self, emb_szs, n_cont, out_sz, hidden_szs, drop_probs=None,
                 emb_drop=0, y_range=None, use_bn=True, bn_final=False):
        """
        Parameters:
            emb_szs: list of tuples of (vocab_size, embedding_dimension)
            n_cont: number of continuous variables (i.e. features that are numbers)
            out_sz: output size, i.e. number of output neurons
            hidden_szs: list of hidden layer sizes
            drop_probs: a list of probabilities for each dropout layer, default = 0 for all
                        same length as hidden_szs
            emb_drop: drop probability after the embedding layer
            y_range: if specified, an output sigmoid activation to put it in this range
            use_bn: skip BatchNorm layers except the one applied to cont. variables if False
            bn_final:  Use the final batch normalization layer?
        """
        super(TabularModel, self).__init__()

        self._hidden_layer_count = len(hidden_szs) + 1
        drop_probs = if_none(drop_probs, [0] * len(hidden_szs))
        self._embeds = [Embedding(vs, ed) for vs, ed in emb_szs]
        self._emb_drop_layer = Dropout(emb_drop)
        self._bn_cont = BatchNormalization()
        self._n_emb = sum(ed for _, ed in emb_szs)
        self._n_cont, self._y_range = n_cont, y_range

        sizes = [self._n_emb + self._n_cont] + hidden_szs + [out_sz]

        activations = ['relu' for _ in range(len(sizes) - 2)] + [None]

        self._hidden_layers = []
        for i, (n_out, dp, act) in enumerate(zip(sizes[1:], [0.] + drop_probs, activations)):
            self._hidden_layers.append(BatchDropLin(n_out, bn=use_bn and i != 0, p=dp, act=act))
        if bn_final: self._hidden_layers.append(BatchNormalization())

    def call(self, inputs, training):
        """
        Arguments:
            x_cat: tensor of categorical features encoded with their token id's, shape = [batch, num_cat_features]
            x_cont: tensor of continuous features (numbers), shape = [batch, num_cont_features]
            training: to pass along to BatchNorm/Dropout layers
        """
        x_cat = inputs[0]
        x_cont = inputs[1]
        if self._n_emb != 0:  # There is at least one categorical with an embedding dimension
            x = [e(x_cat[:, i]) for i, e in enumerate(self._embeds)]
            # x is a list of embeddings, a list of [batch, num_cat_features, emb_dim] tensors
            x = tf.concat(x, 1)  # concated along features dim, shape = [batch, sum(emb_dim)]
            x = self._emb_drop_layer(x, training)
        if self._n_cont != 0:  # There is at least 1 continuous feature
            x_cont = self._bn_cont(x_cont, training)  # shape = [batch, num_cont_features]
            x = tf.concat([x, x_cont], 1)  # shape = [batch, sum(emb_dim) + num_cont_features]
        for i in range(self._hidden_layer_count):
            x = self._hidden_layers[i](x, training)
        if self._y_range is not None:
            x = (self._y_range[1] - self._y_range[0]) * sigmoid(x) + self._y_range[0]
        return x


def add_datepart(df: pd.DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False):
    """
    Helper function that adds columns relevant to a date in the column `field_name` of `df`.
    fieldname is assumed to be a date or datetime column
    """
    if (prefix is None): prefix = ''
    field = df[field_name]
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower())
    df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


def cat_n_toke(df, cat_list):
    """
    Arguments:
        df: the dataframe to modify
        cat_list: a list of field names to cat & toke

    Helper function to categorify and tokenize categorical variables for embeddings
    returns the list of new fields and optionally drops the original fields
    """
    df[cat_list] = df[cat_list].astype('category')
    cat_tokens_list = []
    for cat_field in cat_list:
        cat_token_field = cat_field + '_token_id'
        df[cat_token_field] = (df[cat_field].cat.codes + 1)
        cat_tokens_list.append(cat_token_field)

    return cat_tokens_list


def norm_cont(df, cont_list, miss_strat='mean', epsilon=EPSILON):
    """
    Arguments:
        df: the dataframe to modify
        cont_list: a list of field names to normalized
        miss_strat: strategy to apply to missing values
        epsilon: to avoid divide by zero cases, a small positive number (~1e-7)

    Helper function to normalize continuous inputs
    returns a list of new fields
    """
    df[cont_list] = df[cont_list].astype('float')
    cont_normed_list = []
    for cont_field in cont_list:
        cont_normed_field = cont_field + '_normed'
        field_mean = df[cont_field].mean()
        df[cont_normed_field] = (df[cont_field] - field_mean) / (df[cont_field].std() + epsilon)
        df[cont_normed_field] = df[cont_normed_field].fillna(0)
        cont_normed_list.append(cont_normed_field)

    return cont_normed_list


def promo2_feat(row):
    """
    feature calculation for if promo2 is active
    """
    if row['Promo2'] == 0: return 0
    promo_start_date = datetime.datetime.strptime(f"{int(row['Promo2SinceYear'])} {int(row['Promo2SinceWeek'])} {1}",
                                            '%Y %U %w')
    promo_start_date = promo_start_date.date()
    datum_date = row['Date'].date()
    if datum_date < promo_start_date: return 0
    months = row['PromoInterval'].split(',')
    if datum_date.strftime('%b') not in months: return 0
    return 1


def states_names_to_abbreviation(state_name):
    d = {}
    d['BadenWuerttemberg'] = 'BW'
    d['Bayern'] = 'BY'
    d['Berlin'] = 'BE'
    d['Brandenburg'] = 'BB'  # do not exist in store_state
    d['Bremen'] = 'HB'  # we use Niedersachsen instead of Bremen
    d['Hamburg'] = 'HH'
    d['Hessen'] = 'HE'
    d['MecklenburgVorpommern'] = 'MV'  # do not exist in store_state
    d['Niedersachsen'] = 'HB,NI'  # we use Niedersachsen instead of Bremen
    d['NordrheinWestfalen'] = 'NW'
    d['RheinlandPfalz'] = 'RP'
    d['Saarland'] = 'SL'
    d['Sachsen'] = 'SN'
    d['SachsenAnhalt'] = 'ST'
    d['SchleswigHolstein'] = 'SH'
    d['Thueringen'] = 'TH'

    return d[state_name]


train_df = pd.read_csv(data_dir + 'train.csv', dtype={'StateHoliday':object})
test_df = pd.read_csv(data_dir + 'test.csv', dtype={'StateHoliday':object})
store_df = pd.read_csv(data_dir + 'store.csv')
store_states_df = pd.read_csv(data_dir + 'store_states.csv')

#Combine google_trends csv's into one dataframe
csv_location = 'googletrend'
google_trend_files = glob.glob(csv_location + '/*.csv')
columns = ['Woche', 'Dez. 2012 - Sep. 2015', 'State Code']
google_trend_df = pd.DataFrame(columns=columns)

for one_state in google_trend_files:
    state = os.path.splitext(os.path.basename(one_state))[0]
    state_code = state[-2:]
    if state_code == 'NI':
        state_code = 'HB,NI'
    gt_one_state_df = pd.read_csv(one_state)
    gt_one_state_df['State Code'] = state_code
    google_trend_df = pd.concat([google_trend_df, gt_one_state_df], axis=0)


def parse_year_week(week_str):
    end_day_of_range = week_str.split(' - ')[1]
    dt = datetime.datetime.strptime(end_day_of_range, '%Y-%m-%d')
    year = dt.year
    week_of_year = dt.isocalendar()[1]
    return year, week_of_year


google_trend_df[['Year', 'Week of Year']] = \
    google_trend_df.apply(lambda row: pd.Series(parse_year_week(row['Woche'])), axis=1)

#preparing weather data
csv_location = 'weather'
german_states_weather = glob.glob(csv_location + '/*.csv')
columns = ['Date', 'CloudCover', 'Events', 'Max_TempC_Normed', 'Mean_TempC_Normed', 'Min_TempC_Normed',
           'Max_Humidity_Normed', 'Mean_Humidity_Normed', 'Min_Humidity_Normed', 'Max_Wind_Normed', 'Mean_Wind_Normed',
           'State Code']
gs_weather_df = pd.DataFrame(columns=columns)

for one_state in german_states_weather:
    state_name = os.path.splitext(os.path.basename(one_state))[0]
    state_code = states_names_to_abbreviation(state_name)
    one_state_weather_df = pd.read_csv(one_state, delimiter=';')
    one_state_weather_df['Max_TempC_Normed'] = (one_state_weather_df.Max_TemperatureC - 10) / 30
    one_state_weather_df['Mean_TempC_Normed'] = (one_state_weather_df.Mean_TemperatureC - 10) / 30
    one_state_weather_df['Min_TempC_Normed'] = (one_state_weather_df.Min_TemperatureC - 10) / 30
    one_state_weather_df['Max_Humidity_Normed'] = (one_state_weather_df.Max_Humidity - 50) / 50
    one_state_weather_df['Mean_Humidity_Normed'] = (one_state_weather_df.Mean_Humidity - 50) / 50
    one_state_weather_df['Min_Humidity_Normed'] = (one_state_weather_df.Min_Humidity - 50) / 50
    one_state_weather_df['Max_Wind_Normed'] = one_state_weather_df.Max_Wind_SpeedKm_h / 50
    one_state_weather_df['Mean_Wind_Normed'] = one_state_weather_df.Mean_Wind_SpeedKm_h / 30
    one_state_weather_df['State Code'] = state_code
    one_state_weather_df = one_state_weather_df[columns]
    gs_weather_df = pd.concat([gs_weather_df, one_state_weather_df], axis=0)

gs_weather_df.CloudCover = gs_weather_df.CloudCover.fillna(0)
gs_weather_df.Date = pd.to_datetime(gs_weather_df.Date)

#target = train_df.Sales

#customers = train_df.Customers

train_df.insert(0, 'Id', None)
test_df.insert(4, 'Customers', None)
test_df.insert(4, 'Sales', None)
#train_df = train_df.drop(['Sales', 'Customers'], axis=1)

train_df.Date = pd.to_datetime(train_df.Date)
test_df.Date = pd.to_datetime(test_df.Date)

#remove unopen days as they are uninformative and we will hardcode 0 to closed days
open_idx = train_df[train_df.Open != 0].index
train_df = train_df.loc[open_idx]
#target = target.loc[open_idx]

train_end_date = '2014-07-31'
valid_end_date = '2015-07-31'

train_idx = train_df[train_df.Date < '2014-08-01'].index
valid_idx = train_df.index.difference(train_idx)

#Split out a validation set 1 year from the end of original train data
valid_df = train_df.loc[valid_idx]
train_df = train_df.loc[train_idx]

#valid_target = target.loc[valid_idx]
#train_target = target.loc[train_idx]

#valid_customers = customers.loc[valid_idx]
#train_customers = customers.loc[train_idx]

train_df['dataset'] = 'train'
valid_df['dataset'] = 'valid'
test_df['dataset'] = 'test'

all_df = pd.concat([train_df, valid_df, test_df])

all_df = all_df.merge(store_df, how='left', left_on='Store', right_on='Store')

all_df['Promo2'] = all_df.apply(lambda row: promo2_feat(row), axis=1)
add_datepart(all_df, 'Date', drop=False)
all_df = all_df.drop(['DayOfWeek', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Elapsed'], axis=1)

#Add in German States
all_df = all_df.merge(store_states_df, how='left', left_on='Store', right_on='Store')
all_df = all_df.merge(google_trend_df, how='left', left_on=['Year', 'Week', 'State'],
                      right_on=['Year', 'Week of Year', 'State Code'])
all_df = all_df.drop(['Woche', 'State Code', 'Week of Year'], axis=1)
all_df = all_df.merge(gs_weather_df, how='left', left_on=['Date', 'State'], right_on=['Date', 'State Code'])
all_df = all_df.drop(['State Code'], axis=1)
all_df = all_df.rename(mapper={'Dez. 2012 - Sep. 2015': 'weekly_google_trend'}, axis=1)

cat_list = ['Store', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
            'State', 'Events']

cont_list = ['CompetitionDistance', 'weekly_google_trend', 'CloudCover', 'Max_TempC_Normed', 'Mean_TempC_Normed',
             'Min_TempC_Normed', 'Max_Humidity_Normed', 'Mean_Humidity_Normed', 'Min_Humidity_Normed',
             'Max_Wind_Normed', 'Mean_Wind_Normed']

cat_token_list = cat_n_toke(all_df, cat_list)
cont_normed_list = norm_cont(all_df, cont_list)

vocab_szs = list(all_df[cat_list].nunique().values)
embedding_size_dict = {
    'Store': 50,
    'Promo': 4,
    'StateHoliday': 3,
    'SchoolHoliday': 1,
    'StoreType': 2,
    'Assortment': 2,
    'CompetitionOpenSinceMonth': 2,
    'CompetitionOpenSinceYear': 2,
    'Promo2': 4,
    'Year': 2,
    'Month': 6,
    'Week': 2,
    'Day': 10,
    'Dayofweek': 6,
    'Dayofyear': 150,
    'Is_month_end': 5,
    'Is_month_start': 5,
    'Is_quarter_end': 5,
    'Is_quarter_start': 5,
    'Is_year_end': 5,
    'Is_year_start': 5,
    'State': 6,
    'Events': 10
}
emb_szs = [(vs+1, embedding_size_dict[es]) for vs, es in zip(vocab_szs, embedding_size_dict)]

all_ds = all_df.to_xarray()
all_ds = all_ds.set_index({'index': ('Store', 'Date')})
all_ds = all_ds.unstack('index')

train_ds = all_ds.loc[{'Date': slice(None, train_end_date)}]

train_df = all_df[all_df['dataset'] == 'train']
valid_df = all_df[all_df['dataset'] == 'valid']


model = TabularModel(emb_szs=emb_szs, n_cont=len(cont_list), out_sz=1, hidden_szs=[500, 1000, 500],
                     drop_probs=[0.2, 0.2, 0.2], emb_drop=0.02, y_range=None, use_bn=True, bn_final=False)

model.compile(optimizer='adam', loss='mse')

#defining training callbacks
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/", histogram_freq=1)

es_callback = tf.keras.callbacks.EarlyStopping(
    # Stop training when `val_loss` is no longer improving
    monitor='val_loss',
    # "no longer improving" being defined as "no better than 1e-2 less"
    min_delta=1e-2,
    # "no longer improving" being further defined as "for at least 2 epochs"
    patience=2,
    verbose=1)

"""
history = model.fit(x=[train_df[cat_token_list].values, train_df[cont_normed_list].values], y=train_target.values,
            batch_size=256, epochs=30, verbose=1,
            validation_data=([valid_df[cat_token_list].values, valid_df[cont_normed_list].values], valid_target.values),
            callbacks=[tb_callback, es_callback])


train_df = pd.concat([train_df, valid_df], axis=0)
train_target = pd.concat([train_target, valid_target], axis=0)

history = model.fit(x=[train_df[cat_token_list].values, train_df[cont_normed_list].values], y=train_target.values,
            batch_size=256, epochs=2, verbose=1, callbacks=[tb_callback])
"""

test_df = all_df[all_df['dataset'] == 'test']
close_test_df = test_df[test_df.Open == 0]
open_test_df = test_df[test_df.Open != 0]

y_pred = model.predict(x=[open_test_df[cat_token_list].values, open_test_df[cont_normed_list].values])
open_test_df.loc[:, 'Sales'] = y_pred
open_test_df.Sales = open_test_df.Sales.apply(lambda x: max(x, 0))
close_test_df.loc[:, 'Sales'] = 0

sub_df = pd.concat([open_test_df, close_test_df], axis=0)
sub_df = sub_df.sort_values('Id')
sub_df = sub_df[['Id', 'Sales']]

sub_df.to_csv('submission.csv', index=False)