import numpy as np
import pandas as pd
import xarray as xr
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Dropout, GRU, TimeDistributed

from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()

# Get Training Data
(market_train_df, news_train_df) = env.get_training_data()


# multiply out the rows by number of assetCodes per item
def normalizeNews(market_df, news_df):
    ac_split_df = news_df['assetCodes'].str.extractall(f"'([\w\./]+)'")
    ac_split_df.reset_index(level='match', drop=True, inplace=True)
    # reduce down to useful assetcodes
    ac_split_df = ac_split_df[ac_split_df[0].isin(market_df.assetCode)]
    ac_split_df.rename(mapper={0: 'assetCode'}, axis=1, inplace=True)
    return news_df.merge(ac_split_df, left_index=True, right_index=True, copy=False)


def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


def make_naive_predictions(predictions_df, news_obs_df):
    predictions_df.confidenceValue = 0  # Ensure 0 = default
    news_norm_df = normalizeNews(news_obs_df)
    news_norm_df = news_norm_df[news_norm_df.assetCode.isin(predictions_df.assetCode)]
    # Only use the latest news
    latest_news_df = news_norm_df.loc[news_norm_df.groupby('assetCode').sourceTimestamp.idxmax()]
    latest_news_df['confidenceValue'] = latest_news_df.sentimentPositive - latest_news_df.sentimentNegative
    merged_df = predictions_df.merge(latest_news_df[['assetCode', 'confidenceValue']], how='left', on='assetCode',
                                     suffixes=('', '_calc'))
    predictions_df.confidenceValue = merged_df.fillna(0).confidenceValue_calc


# Looks at previous prediction and evaluate the subscore x_t
def eval_xt(prev_Pred_df):
    prev_Pred_df['score'] = np.multiply(prev_Pred_df['confidenceValue'], prev_Pred_df['returnsOpenNextMktres10'])
    return prev_Pred_df['score'].sum(axis=0)


def evalScore(subscore_arr):
    return np.mean(subscore_arr) / np.std(subscore_arr)


# gets the equivalent asset time that the news datetime refers to. (accounts for weekends)
def calcAssetTime(newsDT):
    return (newsDT - pd.Timedelta('22 hours')).ceil('d') + 0 * pd.tseries.offsets.BDay() + pd.Timedelta('22 hours')


def newsGroupAgg(x):
    s = {}
    s['timeOfDay'] = np.average(x.timeOfDay, weights=x.relevance)
    s['urgency'] = x.urgency.min()
    s['bodySize'] = x.bodySize.sum()
    s['companyCount'] = x.companyCount.sum()
    s['marketCommentary'] = x.marketCommentary.max()
    s['sentenceCount'] = x.sentenceCount.sum()
    s['wordCount'] = x.wordCount.sum()
    s['firstMentionSentence'] = x.firstMentionSentence.min()
    s['relevance'] = x.relevance.mean()
    s['sentimentNegative'] = np.average(x.sentimentNegative, weights=x.relevance)
    s['sentimentNeutral'] = np.average(x.sentimentNeutral, weights=x.relevance)
    s['sentimentPositive'] = np.average(x.sentimentPositive, weights=x.relevance)
    s['sentimentWordCount'] = x.sentimentWordCount.sum()
    s['noveltyCount12H'] = x.noveltyCount12H.iloc[-1]
    s['noveltyCount24H'] = x.noveltyCount24H.iloc[-1]
    s['noveltyCount3D'] = x.noveltyCount3D.iloc[-1]
    s['noveltyCount5D'] = x.noveltyCount5D.iloc[-1]
    s['noveltyCount7D'] = x.noveltyCount7D.iloc[-1]
    s['volumeCounts12H'] = x.volumeCounts12H.iloc[-1]
    s['volumeCounts24H'] = x.volumeCounts24H.iloc[-1]
    s['volumeCounts3D'] = x.volumeCounts3D.iloc[-1]
    s['volumeCounts5D'] = x.volumeCounts5D.iloc[-1]
    s['volumeCounts7D'] = x.volumeCounts7D.iloc[-1]
    return pd.Series(s)


# reduce the News data by aggregating/downsampling to day-level data in a meaningful manner
# also removes news entries from assets which we don't have training data on
def reduceTrainNews(market_train_df, news_train_df):
    # segment training data into relevant days
    news_train_df = normalizeNews(market_train_df, news_train_df)
    news_train_df['time'] = news_train_df.sourceTimestamp.map(calcAssetTime)
    # add columns for calculating weighted averages
    news_train_df['timeOfDay'] = (
                news_train_df.sourceTimestamp - news_train_df.sourceTimestamp.dt.floor('D')).dt.total_seconds()
    return news_train_df.groupby(['assetCode', 'time']).apply(newsGroupAgg)


# Need different obs reduction to account for reduction in temporal dimension
# Makes handling holiday cases simpler for now
def reduceObsNews(market_obs_df, news_obs_df):
    # segment training data into relevant days
    news_obs_df = normalizeNews(market_obs_df, news_obs_df)
    # add columns for calculating weighted averages
    news_obs_df['timeOfDay'] = (
                news_obs_df.sourceTimestamp - news_obs_df.sourceTimestamp.dt.floor('D')).dt.total_seconds()
    return news_obs_df.groupby('assetCode').apply(newsGroupAgg)


# When there's no news they need a default for a time-independent model
def fillDefaultNews(merged_obs_df):
    timeOfDayMean = merged_obs_df.timeOfDay.mean()
    if not (np.isnan(timeOfDayMean)):
        merged_obs_df.timeOfDay.fillna(merged_obs_df.timeOfDay.mean(), inplace=True)
    else:
        merged_obs_df.timeOfDay.fillna(57600, inplace=True)  # use 4pm
    merged_obs_df.urgency.fillna(3, inplace=True)
    merged_obs_df.bodySize.fillna(0, inplace=True)
    merged_obs_df.companyCount.fillna(0, inplace=True)
    merged_obs_df.marketCommentary.fillna(False, inplace=True)
    merged_obs_df.sentenceCount.fillna(0, inplace=True)
    merged_obs_df.wordCount.fillna(0, inplace=True)
    merged_obs_df.firstMentionSentence.fillna(0, inplace=True)
    merged_obs_df.relevance.fillna(0, inplace=True)
    merged_obs_df.sentimentNegative.fillna(0, inplace=True)
    merged_obs_df.sentimentNeutral.fillna(1, inplace=True)
    merged_obs_df.sentimentPositive.fillna(0, inplace=True)
    merged_obs_df.sentimentWordCount.fillna(0, inplace=True)
    merged_obs_df.noveltyCount12H.fillna(0, inplace=True)
    merged_obs_df.noveltyCount24H.fillna(0, inplace=True)
    merged_obs_df.noveltyCount3D.fillna(0, inplace=True)
    merged_obs_df.noveltyCount5D.fillna(0, inplace=True)
    merged_obs_df.noveltyCount7D.fillna(0, inplace=True)
    merged_obs_df.volumeCounts12H.fillna(0, inplace=True)
    merged_obs_df.volumeCounts24H.fillna(0, inplace=True)
    merged_obs_df.volumeCounts3D.fillna(0, inplace=True)
    merged_obs_df.volumeCounts5D.fillna(0, inplace=True)
    merged_obs_df.volumeCounts7D.fillna(0, inplace=True)


# train the model
def make_model(time_steps=1):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(time_steps, 30)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(GRU(units=128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(TimeDistributed(Dense(1, activation='tanh')))
    model.compile(optimizer='Adam', loss='mse')
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)


def make_simpleNN_predictions(model, predictions_template_df, market_obs_df, news_obs_df):
    x_df, y_df = shapeObsData(market_obs_df, news_obs_df)
    y_df['pred'] = model.predict(x_df.iloc[:, 2:].values)
    merged = predictions_template_df.merge(y_df, how='left', left_on='assetCode', right_on='assetCode')
    predictions_template_df.confidenceValue = merged.pred


def make_RNN_predictions(model, predictions_template_df, market_obs_df, news_obs_df):
    x_df = shapeObsData(market_obs_df, news_obs_df)
    x_df['confidenceValue'] = model.predict(np.expand_dims(x_df.values, 1)).squeeze()
    predictions_template_df.set_index('assetCode', inplace=True)
    predictions_template_df.update(x_df)
    predictions_template_df.reset_index(inplace=True)


# shapes the data in preparation for training or prediction
# returns x,y for training with identifiers, or x,y_pred_holder for prediction with identifiers
# shaping data for a recurrent model
def shapeTrainData(market_train_df, news_train_df):
    # dataprep, reduce to relevant news
    news_train_df = reduceTrainNews(market_train_df, news_train_df)
    # shape market training data to an (assetCode,time) index
    # let's remove some market columns for potential space
    market_train_df.drop(
        columns=['assetName', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevMktres10',
                 'returnsOpenPrevMktres10', 'universe'],
        inplace=True)
    market_ds = xr.Dataset.from_dataframe(market_train_df)
    market_ds.set_index(index=['assetCode', 'time'], inplace=True)
    market_ds = market_ds.unstack('index')
    # shape news data to a dataset
    news_ds = xr.Dataset.from_dataframe(news_train_df)
    news_ds.coords['time'] = pd.to_datetime(news_ds.coords['time'])
    merged_ds = xr.merge([market_ds, news_ds], join='left')
    # fill in data due to alignment
    merged_ds['volume'] = merged_ds['volume'].fillna(0)
    merged_ds['close'] = merged_ds['close'].ffill('time')
    merged_ds['close'] = merged_ds['close'].fillna(0)
    merged_ds['open'] = merged_ds['open'].ffill('time')
    merged_ds['open'] = merged_ds['open'].fillna(0)
    merged_ds['returnsClosePrevRaw1'] = merged_ds['returnsClosePrevRaw1'].fillna(0)
    merged_ds['returnsOpenPrevRaw1'] = merged_ds['returnsOpenPrevRaw1'].fillna(0)
    merged_ds['returnsClosePrevRaw10'] = merged_ds['returnsClosePrevRaw10'].fillna(0)
    merged_ds['returnsOpenPrevRaw10'] = merged_ds['returnsOpenPrevRaw10'].fillna(0)
    merged_ds['timeOfDay'] = merged_ds['timeOfDay'].fillna(merged_ds['timeOfDay'].mean())
    merged_ds['timeOfDay'] = merged_ds['timeOfDay'].fillna(57600)
    merged_ds['urgency'] = merged_ds['urgency'].fillna(3)
    merged_ds['bodySize'] = merged_ds['bodySize'].fillna(0)
    merged_ds['companyCount'] = merged_ds['companyCount'].fillna(0)
    merged_ds['marketCommentary'] = merged_ds['marketCommentary'].fillna(False)
    merged_ds['marketCommentary'] = merged_ds['marketCommentary'].astype(int)
    merged_ds['sentenceCount'] = merged_ds['sentenceCount'].fillna(0)
    merged_ds['wordCount'] = merged_ds['wordCount'].fillna(0)
    merged_ds['firstMentionSentence'] = merged_ds['firstMentionSentence'].fillna(
        merged_ds['firstMentionSentence'].max())
    merged_ds['firstMentionSentence'] = merged_ds['firstMentionSentence'].fillna(1000)
    merged_ds['relevance'] = merged_ds['relevance'].fillna(0)
    merged_ds['sentimentNegative'] = merged_ds['sentimentNegative'].fillna(0)
    merged_ds['sentimentNeutral'] = merged_ds['sentimentNeutral'].fillna(1)
    merged_ds['sentimentPositive'] = merged_ds['sentimentPositive'].fillna(0)
    merged_ds['sentimentWordCount'] = merged_ds['sentimentWordCount'].fillna(0)
    merged_ds['noveltyCount12H'] = merged_ds['noveltyCount12H'].fillna(0)
    merged_ds['noveltyCount24H'] = merged_ds['noveltyCount24H'].fillna(0)
    merged_ds['noveltyCount3D'] = merged_ds['noveltyCount3D'].fillna(0)
    merged_ds['noveltyCount5D'] = merged_ds['noveltyCount5D'].fillna(0)
    merged_ds['noveltyCount7D'] = merged_ds['noveltyCount7D'].fillna(0)
    merged_ds['volumeCounts12H'] = merged_ds['volumeCounts12H'].fillna(0)
    merged_ds['volumeCounts24H'] = merged_ds['volumeCounts24H'].fillna(0)
    merged_ds['volumeCounts3D'] = merged_ds['volumeCounts3D'].fillna(0)
    merged_ds['volumeCounts5D'] = merged_ds['volumeCounts5D'].fillna(0)
    merged_ds['volumeCounts7D'] = merged_ds['volumeCounts7D'].fillna(0)
    merged_ds['returnsOpenNextMktres10'] = merged_ds['returnsOpenNextMktres10'].fillna(0)
    y = merged_ds['returnsOpenNextMktres10']
    merged_ds = merged_ds.drop('returnsOpenNextMktres10')
    y = np.sign(y)
    y = y.rename('optPred')
    y = y.expand_dims('variable')
    return merged_ds, y


# Obs data requires different shaping because of reduction of time dimensionality and doesn't return an optimal prediction array
def shapeObsData(market_obs_df, news_obs_df):
    # dataprep, reduce to relevant news
    news_obs_df = reduceObsNews(market_obs_df, news_obs_df)
    # shape market training data to an (assetCode) index
    # let's remove some market columns for potential space
    market_obs_df.drop(
        columns=['time', 'assetName', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevMktres10',
                 'returnsOpenPrevMktres10'],
        inplace=True)
    market_obs_df.set_index('assetCode', drop=True, inplace=True)
    merged_df = market_obs_df.merge(news_obs_df, how='left', left_index=True, right_index=True)
    # fill in data due to alignment
    fillDefaultNews(merged_df)
    merged_df['marketCommentary'] = merged_df['marketCommentary'].astype(int)
    return merged_df


market_train_df = market_train_df[market_train_df.universe == 1]
x_train_ds, y_train_da = shapeTrainData(market_train_df, news_train_df)
model_t = make_model(x_train_ds.sizes['time'])
model_t.fit(x_train_ds.to_array().transpose('assetCode', 'time', 'variable').values,
            y_train_da.transpose('assetCode', 'time', 'variable').values, verbose=1)
# need seperate model for single time step
model_p = make_model(1)
model_p.set_weights(model_t.get_weights())
# woohoo model is ready

# Get Days, start main loop
days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_RNN_predictions(model_p, predictions_template_df, market_obs_df, news_obs_df)
    env.predict(predictions_template_df)

env.write_submission_file()