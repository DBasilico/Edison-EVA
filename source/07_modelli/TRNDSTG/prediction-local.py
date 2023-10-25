import shutil
from datetime import datetime
from urllib.parse import quote
import numpy as np
import pandas as pd
from lib import seasonal_decompose, mape
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
import os
import pathlib
from pyspark.sql import functions as f

spark = SparkSession.builder.appName('STAGIONALITA SEMPLICE - PREVISIONE').enableHiveSupport().getOrCreate()

def datetime_to_epoch(ser):
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')
    return res // 10**9


parameters = [pathlib.Path(f).stem for f in os.listdir(f'.\\PARAMETERS')]
parameters = [x.split('_') for x in parameters]
for x in parameters:
    x[0] = x[0].split('#')

for results in parameters:

    print(results)

    DATA_CONSUNTIVO = results[0][0]
    COD_SIMULAZIONE = results[1]
    COD_MODELLO = f'{DATA_CONSUNTIVO}#{int(results[0][1]):02}#v1'
    BASE_PATH = f'.\\BEST_CHOICE_CLUSTER'
    PARAMETERS_FILE = f'.\\PARAMETERS\\{COD_MODELLO}_{COD_SIMULAZIONE}_{results[2]}.parquet'

    GIORNI_PREVISIONE = (30-int(DATA_CONSUNTIVO[-2:])+1)

    df_params = pd.read_parquet(PARAMETERS_FILE)

    CLUSTERS = df_params['CLUSTER'].unique()

    df_params = df_params.set_index('CLUSTER')

    df_out = pd.DataFrame(columns=[])

    for cluster in CLUSTERS:

        df_params_cluster = df_params.loc[[cluster]].iloc[0]

        print(cluster)

        cluster_path = f'.\\BEST_CHOICE_CLUSTER\\CLUSTER={quote(cluster, safe="")}'

        files = [f for f in os.listdir(f'{cluster_path}')]

        df = pd.concat(
            pd.read_parquet(os.path.join(cluster_path, parquet_file))
            for parquet_file in files if pathlib.Path(parquet_file).suffix == '.parquet'
        ).reset_index(drop=True)

        df['CONSUMI_1G'] = df['ENERGIA_1G_BEST'].combine_first(df['ENERGIA_1G_GIORNO']).fillna(0.).replace(np.nan, 0.)
        df['CONSUMI_2G'] = df['ENERGIA_2G_BEST'].fillna(0.).replace(np.nan, 0.)
        df['CONSUMI'] = df['CONSUMI_1G']+df['CONSUMI_2G']

        df = df[df.ORA_GME < 25]

        df['CONSUMI_NORMALIZZATI'] = np.where(df['N_POD'] == df['N_POD_MANCANTI'], df['CONSUMI']/df['N_POD'], df['CONSUMI']/(df['N_POD']-df['N_POD_MANCANTI']))
        #df['CONSUMI_NORMALIZZATI'] = df['CONSUMI'] / df['N_POD']

        df = df[df['UNIX_TIME'] < (datetime.strptime(DATA_CONSUNTIVO, "%Y%m%d") + relativedelta(days=GIORNI_PREVISIONE)).timestamp()]
        df = df[df['UNIX_TIME'] >= datetime(2023, 1, 1).timestamp()].copy()

        df = df[['UNIX_TIME', 'TIMESTAMP', 'CONSUMI_NORMALIZZATI', 'N_POD', 'N_POD_MANCANTI']]
        df = df.rename(columns={'TIMESTAMP': 'TIMESTAMP_STRING', 'CONSUMI_NORMALIZZATI': 'CONSUMI'})

        df = df.astype({'UNIX_TIME': np.int64, 'CONSUMI': np.float64, 'N_POD': np.int32, 'N_POD_MANCANTI': np.int32})
        df['TIMESTAMP'] = pd.to_datetime(df['UNIX_TIME'], unit='s', utc=True).dt.tz_convert('Europe/Rome')
        df = df.set_index('TIMESTAMP').sort_index()

        if df_params_cluster['TIPO_MODELLO'] == 'multiplicative' and (df['CONSUMI'] <= 0).any():
            print('---> ADEGUAMENTO MODELLO - ADDITTIVO <---')
            df_params_cluster['TIPO_MODELLO'] = 'addittive'

        seasonal_dec_result = seasonal_decompose(df['CONSUMI'], model=df_params_cluster['TIPO_MODELLO'], two_sided=False, period=df_params_cluster['PERIODO'])
        trend = seasonal_dec_result.trend
        seasonal = seasonal_dec_result.seasonal

        prediction = trend + seasonal
        prediction.name = 'PREVISIONE'
        prediction = prediction.to_frame()

        prediction = pd.concat([prediction, df['CONSUMI']], axis=1)
        prediction['UNIX_TIME'] = datetime_to_epoch(prediction.index)

        prediction = prediction[prediction.UNIX_TIME > datetime.strptime(DATA_CONSUNTIVO, "%Y%m%d").timestamp()]

        prediction = prediction.merge(df[['N_POD', 'N_POD_MANCANTI']], how='left', left_index=True, right_index=True)

        prediction['PREVISIONE'] = np.where(prediction['N_POD'] == prediction['N_POD_MANCANTI'],
                                            prediction['PREVISIONE']*prediction['N_POD'],
                                            prediction['PREVISIONE']*(prediction['N_POD']-prediction['N_POD_MANCANTI']))
        prediction['CONSUMI'] = np.where(prediction['N_POD'] == prediction['N_POD_MANCANTI'],
                                            prediction['CONSUMI']*prediction['N_POD'],
                                            prediction['CONSUMI']*(prediction['N_POD']-prediction['N_POD_MANCANTI']))

        #prediction['PREVISIONE'] = prediction['PREVISIONE']*prediction['N_POD']
        #prediction['CONSUMI'] = prediction['CONSUMI']*prediction['N_POD']

        prediction = prediction[['PREVISIONE', 'CONSUMI', 'UNIX_TIME']]
        prediction['CLUSTER'] = cluster
        prediction['COD_SIMULAZIONE'] = COD_SIMULAZIONE
        prediction['COD_MODELLO'] = COD_MODELLO
        prediction['GIORNI_PREVISIONE'] = GIORNI_PREVISIONE

        df_out = pd.concat([df_out, prediction])

    current_folder = f'.\\PREVISIONI\\COD_SIMULAZIONE={COD_SIMULAZIONE}\\COD_MODELLO={quote(COD_MODELLO, safe="")}'
    if os.path.isdir(current_folder):
        shutil.rmtree(current_folder)
    df_out = spark.createDataFrame(df_out).repartition(1)
    df_out = df_out.withColumn('GIORNI_PREVISIONE', f.col('GIORNI_PREVISIONE').cast('int'))
    df_out.write.mode('append').partitionBy('COD_SIMULAZIONE', 'COD_MODELLO')\
        .parquet('./PREVISIONI_TEST')

### CALCOLO WMAE ###

df = spark.read.parquet('./PREVISIONI_TEST')
df = df.withColumn('ZONA', f.split(f.col('CLUSTER'), '#')[0])
df = df.filter(f.col('COD_MODELLO') == '20230918#07#v1')
df = df.filter(f.col('ZONA') == 'NORD')
df = df.withColumn('TIMESTAMP', f.from_unixtime('UNIX_TIME'))

df = df.toPandas()[['CONSUMI', 'PREVISIONE', 'TIMESTAMP']].groupby('TIMESTAMP').sum() / 1000

df['NUM'] = np.abs(df['CONSUMI'].values - df['PREVISIONE'].values)

WMAE = np.sum(np.abs(df['CONSUMI'].values - df['PREVISIONE'].values))/np.sum(df['CONSUMI'].values)
