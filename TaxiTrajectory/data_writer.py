import numpy as np
import pandas as pd
import ast
import h5py
from sklearn.cluster import MeanShift

EPSILON = 1e-7

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

# Change below for local configuration
data_dir = '/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/'
taxistand_dir = '/kaggle/input/taxistanddata/'
output_dir = ''

train_df = pd.read_csv(data_dir + 'train.csv.zip')
test_df = pd.read_csv(data_dir + 'test.csv.zip')
sample_sub_df = pd.read_csv(data_dir + 'sampleSubmission.csv.zip')
taxistand_df = pd.read_csv(taxistand_dir + 'metaData_taxistandsID_name_GPSlocation.csv')
# ERROR in taxistand data, fixing manually
taxistand_df.loc[40, 'Latitude'] = 41.163066654
taxistand_df.loc[40, 'Longitude'] = -8.67598304213

train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

all_df = pd.concat([train_df, test_df])


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


def target_first5_last5_long_lat(polyline_str, training=True):
    polyline_list = ast.literal_eval(polyline_str)
    n = len(polyline_list)

    if training:
        target_long, target_lat = [None, None]

        if n > 0:
            target_long, target_lat = polyline_list.pop(-1)
            n = n - 1

    if n == 0:
        return [None] * 22 if training else [None] * 20

    first5 = polyline_list[:5]
    last5 = polyline_list[-5:]

    to_pad = max(5 - n, 0)

    if to_pad > 0:
        first5 = first5 + ([first5[-1]] * to_pad)
        last5 = ([last5[0]] * to_pad) + last5

    first0_long, first0_lat = first5[0]
    last1_long, last1_lat = last5[-1]
    first1_long, first1_lat = first5[1]
    last2_long, last2_lat = last5[-2]
    first2_long, first2_lat = first5[2]
    last3_long, last3_lat = last5[-3]
    first3_long, first3_lat = first5[3]
    last4_long, last4_lat = last5[-4]
    first4_long, first4_lat = first5[4]
    last5_long, last5_lat = last5[-5]

    return_values = [first0_long, first0_lat, first1_long, first1_lat, first2_long, first2_lat, first3_long,
                     first3_lat, first4_long, first4_lat, last1_long, last1_lat, last2_long, last2_lat, last3_long,
                     last3_lat, last4_long, last4_lat, last5_long, last5_lat]

    if training:
        return_values.append(target_long)
        return_values.append(target_lat)

    return return_values


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


all_df['DATETIME'] = pd.to_datetime(all_df.TIMESTAMP, unit='s')
add_datepart(all_df, 'DATETIME', drop=False, time=True)
all_df = all_df.merge(taxistand_df, how='left', left_on='ORIGIN_STAND', right_on='ID')  # 1710670
all_df = all_df.drop(['ID', 'Descricao', 'CALL_TYPE', 'DAY_TYPE', 'MISSING_DATA', 'Year', 'Month', 'Day', 'Dayofyear',
                      'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end',
                      'Is_year_start', 'Elapsed', 'Latitude', 'Longitude'], axis=1)
all_df['qtr_hr_of_day'] = all_df.Hour * 4 + (all_df.Minute // 15)
all_df = all_df.drop(['Hour', 'Minute', 'Second'], axis=1)

train_df = all_df[all_df.dataset == 'train']
test_df = all_df[all_df.dataset == 'test']

train_df[['first0_long', 'first0_lat', 'first1_long', 'first1_lat', 'first2_long', 'first2_lat',
          'first3_long', 'first3_lat', 'first4_long', 'first4_lat', 'last1_long', 'last1_lat',
          'last2_long', 'last2_lat', 'last3_long', 'last3_lat', 'last4_long', 'last4_lat',
          'last5_long', 'last5_lat', 'target_long', 'target_lat']] = \
    train_df.apply(lambda row: pd.Series(target_first5_last5_long_lat(row['POLYLINE'])), axis=1)

test_df[['first0_long', 'first0_lat', 'first1_long', 'first1_lat', 'first2_long', 'first2_lat',
         'first3_long', 'first3_lat', 'first4_long', 'first4_lat', 'last1_long', 'last1_lat',
         'last2_long', 'last2_lat', 'last3_long', 'last3_lat', 'last4_long', 'last4_lat',
         'last5_long', 'last5_lat']] = \
    test_df.apply(lambda row: pd.Series(target_first5_last5_long_lat(row['POLYLINE'], False)), axis=1)

test_df['target_long'] = None
test_df['target_lat'] = None

# remove things w/o a target
train_df = train_df[~train_df.target_long.isna()]

# removing outliers
naive_center = (train_df.target_long.mean(), train_df.target_lat.mean())
train_df['naive_dist'] = np.sqrt(np.square(train_df.target_long - naive_center[0])
                                 + np.square(train_df.target_lat - naive_center[1]))
# Naively this restricts the distance from the center of all points to around 150 miles
train_df = train_df[train_df.naive_dist < 3]
train_df = train_df.drop('naive_dist', axis=1)

# calculate cluster centers
coords = np.stack([train_df.target_long.values, train_df.target_lat.values], axis=1)
ms = MeanShift(bandwidth=0.001, bin_seeding=True, min_bin_freq=5)
ms.fit(coords)
cluster_centers = ms.cluster_centers_

hf = h5py.File(output_dir + 'cluster_centers3.h5', 'w')
hf.create_dataset('cc', data=cluster_centers)
hf.close()

all_df = pd.concat([train_df, test_df])

emb_field_list = ['Week', 'Dayofweek', 'qtr_hr_of_day', 'TAXI_ID', 'ORIGIN_CALL', 'ORIGIN_STAND']

cont_field_list = ['first0_long', 'first0_lat', 'first1_long', 'first1_lat', 'first2_long', 'first2_lat', 'first3_long',
                   'first3_lat', 'first4_long', 'first4_lat', 'last1_long', 'last1_lat', 'last2_long', 'last2_lat',
                   'last3_long', 'last3_lat', 'last4_long', 'last4_lat', 'last5_long', 'last5_lat']

emb_token_fields_list = cat_n_toke(all_df, emb_field_list)
cont_normed_fields_list = norm_cont(all_df, cont_field_list, 'mean')

emb_szs = list(all_df[emb_token_fields_list].nunique().values)
emb_szs = [(vocab_size + 1, 10) for vocab_size in emb_szs]

hf = h5py.File(output_dir + 'emb_szs.h5', 'w')
hf.create_dataset('embedding_sizes', data=emb_szs)
hf.close()

target_list = ['target_long', 'target_lat']

# simplifying for export
all_df = all_df.drop(emb_field_list, axis=1)
all_df = all_df.drop(cont_field_list, axis=1)
all_df = all_df.drop(['TIMESTAMP', 'POLYLINE', 'DATETIME'], axis=1)

train_df = all_df[all_df['dataset'] == 'train']
test_df = all_df[all_df['dataset'] == 'test']

train_df.to_csv(output_dir + 'taxi_train.csv', index=False)
test_df.to_csv(output_dir + 'taxi_test.csv', index=False)