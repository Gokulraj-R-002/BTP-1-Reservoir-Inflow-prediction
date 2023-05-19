import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import scipy.cluster.hierarchy as shc
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from numpy import unique
from numpy import where
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from statistics import mean
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df1 = pd.read_csv('./climatic_indices/AO.txt', delimiter=',', header=None)
df2 = pd.read_csv('./climatic_indices/EPO.txt', delimiter=',', header=None)
df3 = pd.read_csv('./climatic_indices/NAO.txt', delimiter=',', header=None)
df4 = pd.read_csv('./climatic_indices/NINO12.txt', delim_whitespace=True, header=None)
df5 = pd.read_csv('./climatic_indices/NINO3.txt', delim_whitespace=True, header=None)
df6 = pd.read_csv('./climatic_indices/NINO34.txt', delim_whitespace=True, header=None)
df7 = pd.read_csv('./climatic_indices/NINO4.txt', delim_whitespace=True, header=None)
df8 = pd.read_csv('./climatic_indices/PNA.txt', delimiter=',', header=None)
df9 = pd.read_csv('./climatic_indices/SOI.txt', delim_whitespace=True)
df10 = pd.read_csv('./climatic_indices/WPO.txt', delimiter=',', header=None)
cle_inf = pd.read_excel('./data/cle/CLE_INFLOW.xlsx')
cle_eva = pd.read_excel('./data/cle/CLE_EVAPORATION.xlsx')
cle_pre = pd.read_excel('./data/cle/CLE_PRECIPITATION.xlsx')

def convert_single_to_datetime(df):
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df['date'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d', errors='coerce')
    df.drop(df.columns[0], axis=1, inplace=True)
    return df

def convert_double_to_datetime(df):
    df['date_str'] = df.iloc[:, 0].astype(str) + df.iloc[:, 1].astype(str).str.zfill(3)
    df['date'] = pd.to_datetime(df['date_str'], format='%Y%j', errors='coerce')
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    df.drop('date_str', axis=1, inplace=True)
    return df

def convert_triple_to_datetime(df):
    df['date_str'] = df.iloc[:, 0].astype(str) + df.iloc[:, 1].astype(str).str.zfill(2) + df.iloc[:, 2].astype(str).str.zfill(2)
    df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce')
    df.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)
    df.drop('date_str', axis=1, inplace=True)
    return df

def convert_column_to_datetime(df, column_name):
    df[column_name] = df[column_name].astype(str)
    df['date'] = pd.to_datetime(df[column_name], format='%Y%m%d', errors='coerce')
    df.drop(columns=[column_name], inplace=True)
    return df

df1 = convert_triple_to_datetime(df1)
df2 = convert_triple_to_datetime(df2)
df3 = convert_triple_to_datetime(df3)
df4 = convert_single_to_datetime(df4)
df5 = convert_single_to_datetime(df5)
df6 = convert_single_to_datetime(df6)
df7 = convert_single_to_datetime(df7)
df8 = convert_triple_to_datetime(df8)
df9 = convert_double_to_datetime(df9)
df10 = convert_triple_to_datetime(df10)
cle_inf = convert_column_to_datetime(cle_inf, 'OBS DATE')
cle_eva = convert_column_to_datetime(cle_eva, 'OBS DATE')
cle_pre = convert_column_to_datetime(cle_pre, 'OBS DATE')

def find_min_max_dates(dataframes):
    min_date = dataframes[0]['date'].min()
    max_date = dataframes[0]['date'].max()

    for df in dataframes:
        if 'date' in df.columns:
            df_min_date = df['date'].min()
            df_max_date = df['date'].max()
            min_date = max(min_date, df_min_date)
            max_date = min(max_date, df_max_date)

    return min_date, max_date

dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, cle_inf, cle_eva, cle_pre]
min_date, max_date = find_min_max_dates(dataframes)

print(min_date)
print(max_date)

def reduce_dataframe_by_date_range(df, min_date, max_date):
    mask = (df['date'] >= min_date) & (df['date'] <= max_date)
    reduced_df = df.loc[mask].copy()
    reduced_df.reset_index(drop=True, inplace=True)
    return reduced_df

ao_df = reduce_dataframe_by_date_range(df1, min_date, max_date)
epo_df = reduce_dataframe_by_date_range(df2, min_date, max_date)
nao_df = reduce_dataframe_by_date_range(df3, min_date, max_date)
nino12_df = reduce_dataframe_by_date_range(df4, min_date, max_date)
nino3_df = reduce_dataframe_by_date_range(df5, min_date, max_date)
nino34_df = reduce_dataframe_by_date_range(df6, min_date, max_date)
nino4_df = reduce_dataframe_by_date_range(df7, min_date, max_date)
pna_df = reduce_dataframe_by_date_range(df8, min_date, max_date)
soi_df = reduce_dataframe_by_date_range(df9, min_date, max_date)
wpo_df = reduce_dataframe_by_date_range(df10, min_date, max_date)
cle_inflow_df = reduce_dataframe_by_date_range(cle_inf, min_date, max_date)
cle_evaporation_df = reduce_dataframe_by_date_range(cle_eva, min_date, max_date)
cle_precipitation_df = reduce_dataframe_by_date_range(cle_pre, min_date, max_date)

ao = pd.DataFrame()
epo = pd.DataFrame()
nao = pd.DataFrame()
nino12 = pd.DataFrame()
nino3 = pd.DataFrame()
nino34 = pd.DataFrame()
nino4 = pd.DataFrame()
pna = pd.DataFrame()
soi = pd.DataFrame()
wpo = pd.DataFrame()
cle_inflow = pd.DataFrame()
cle_evaporation = pd.DataFrame()
cle_precipitation = pd.DataFrame()
seasonality = pd.DataFrame()

ao['VALUE'] = ao_df.iloc[:,0]
epo['VALUE'] = epo_df.iloc[:,0]
nao['VALUE'] = nao_df.iloc[:,0]
nino12['VALUE'] = nino12_df.iloc[:,0]
nino3['VALUE'] = nino3_df.iloc[:,0]
nino34['VALUE'] = nino34_df.iloc[:,0]
nino4['VALUE'] = nino4_df.iloc[:,0]
pna['VALUE'] = pna_df.iloc[:,0]
soi['VALUE'] = soi_df['SOI'].copy()
wpo['VALUE'] = wpo_df.iloc[:,0]
cle_inflow['VALUE'] = cle_inflow_df['VALUE'].copy()
cle_evaporation['VALUE'] = cle_evaporation_df['VALUE'].copy()
cle_precipitation['VALUE'] = cle_precipitation_df['VALUE'].copy()
seasonality['VALUE'] = epo_df['date'].dt.month

def converttofloat(s):
  count = 0
  s = str(s)
  v = float(s.replace(',',''))
  if v < 0:
    count +=1
    return np.nan
  else:
    return v
  
cle_inflow['VALUE'] = cle_inflow['VALUE'].apply(converttofloat)
seasonality['VALUE'] = seasonality['VALUE'].apply(converttofloat)

cle_inflow['VALUE'].interpolate(method='linear', inplace=True)
cle_evaporation['VALUE'].interpolate(method='linear', inplace = True)
cle_precipitation['VALUE'].interpolate(method = 'linear', inplace=True)
ao['VALUE'].interpolate(method = 'linear', inplace=True)

def cfstom3(value):
    value = value*0.0283168466
    return value

def inchestomm(value):
    value = value*25.4
    return value

cle_inflow['VALUE'] = cle_inflow['VALUE'].apply(cfstom3)
cle_evaporation['VALUE'] = cle_evaporation['VALUE'].apply(cfstom3)
cle_precipitation['VALUE'] = cle_precipitation['VALUE'].apply(inchestomm)

df_list = [ao,epo,nao,nino12,nino3,nino34,nino4,pna,soi,wpo,seasonality,cle_evaporation,cle_precipitation,cle_inflow]
X = pd.concat(df_list,axis=1)

inflow = pd.DataFrame()
inflow['VALUE'] = cle_inflow.values[:,0]
y = inflow.copy()

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

pca = PCA().fit(X_scaler)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA for CLE Resercoir')
plt.grid()
plt.show()

pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_scaler)

def correlation(s, o):
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]
    return corr

def NS(s, o):
    return 1 - np.sum((s-o)**2)/np.sum((o-np.mean(o))**2)

def KGE(s, o):
    cc = correlation(s,o)
    alpha = np.std(s)/np.std(o)
    beta = np.sum(s)/np.sum(o)
    kge = 1- np.sqrt( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )
    return kge

from scipy.stats import pearsonr
def CORR(s,o):
  corr, _ = pearsonr(o,s)
  return corr

def Absolute_Percentage_Error(s,o):
  return (np.sum(np.abs(s-o))/np.sum(o))*100

from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split

x_lstm = X_pca.reshape(X_pca.shape[0],1,X_pca.shape[1])

stop_noimprovement = EarlyStopping(monitor='loss', patience=10)
model = Sequential()
model.add(LSTM(units=1024, activation='relu', return_sequences=True, input_shape=(1, x_lstm.shape[2])))
model.add(LSTM(units=1024, activation='relu', return_sequences=True))
model.add(Dense(units=1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

kf = KFold(n_splits=15, shuffle=False)
score_30_rmse = []
score_30_corr = []
score_30_r2 = []
score_30_mae = []
score_30_apb = []
r_lstm_1 = []
p_lstm_1 = []
for train_index, test_index in tqdm(kf.split(x_lstm),total=kf.get_n_splits(),desc="k-fold"):
    X_train, X_test, y_train, y_test = x_lstm[train_index], x_lstm[test_index], y.iloc[train_index], y.iloc[test_index]
    lstm_history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), callbacks=[stop_noimprovement], shuffle=False,verbose=False)
    yht = model.predict(X_test)
    yhat= []
    for i in yht:
      if i[0][0] < 0:
        yhat.append(0)
      else:
        yhat.append(i[0][0])
    r_lstm_1.extend(list(y_test['VALUE']))
    p_lstm_1.extend(yhat)
    yhat = np.array(yhat)
    y_test = np.array(list(y_test['VALUE']))
    score_30_rmse.append(math.sqrt(mean_squared_error(y_test, yhat)))
    score_30_corr.append(correlation(y_test,yhat))
    score_30_r2.append(r2_score(y_test,yhat))
    score_30_mae.append(mean_absolute_error(y_test,yhat))
    score_30_apb.append(Absolute_Percentage_Error(y_test,yhat))

lstm_30 = pd.DataFrame({'actual':r_lstm_1,'predict':p_lstm_1})

print('RMSE ',mean(score_30_rmse))
print('CORR ',mean(score_30_corr))
print('R2 ',mean(score_30_r2))
print('MAE ',mean(score_30_mae))
print('APB ',mean(score_30_apb))