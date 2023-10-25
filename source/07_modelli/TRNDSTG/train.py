# import subprocess
# import sys
#
# subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "s3fs"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "awswrangler"])

from datetime import datetime
from urllib.parse import unquote
import boto3
import numpy as np
import pandas as pd
from lib import seasonal_decompose, mape
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error as mae
import awswrangler as wr
import pickle


def datetime_to_epoch(ser):
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')
    return res // 10**9


def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


def delete_folder_s3(bucket: str, folder_path: str):
    folder_path = check_path_s3(folder_path)
    boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=folder_path).delete()


def get_subfolders_s3(bucket: str, path: str = ''):
    path = check_path_s3(path)
    response = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=path, Delimiter='/')
    if 'CommonPrefixes' not in response.keys():
        return
    else:
        for obj in response["CommonPrefixes"]:
            subfolder = obj['Prefix']
            yield subfolder
            yield from get_subfolders_s3(bucket, path=subfolder)


def list_files(bucket: str, folder_path: str):
    folder_path = check_path_s3(folder_path)
    response = boto3.client('s3').list_objects_v2(Bucket=bucket, Prefix=folder_path, Delimiter='/')
    if "Contents" in response.keys():
        ret = []
        for obj in response["Contents"]:
            obj = obj["Key"]
            if obj[-1] != '/':
                ret.append(obj)
        return ret
    else:
        return list()


RUN_PARAMETERS = [
    ['1000', datetime(2023, 9, 25), 24*1],
    ['1000', datetime(2023, 9, 25), 24*3],
    ['1000', datetime(2023, 9, 25), 24*5],
    ['1000', datetime(2023, 9, 25), 24*7],
    ['1000', datetime(2023, 9, 25), 24*14],
    ['1000', datetime(2023, 9, 18), 24*1],
    ['1000', datetime(2023, 9, 18), 24*3],
    ['1000', datetime(2023, 9, 18), 24*5],
    ['1000', datetime(2023, 9, 18), 24*7],
    ['1000', datetime(2023, 9, 18), 24*14]
]

for COD_SIMULAZIONE, DATA_CONSUNTIVO, DELTA_HOURS in RUN_PARAMETERS:

    tic = datetime.now()
    start_time = tic

    COD_MODELLO = f'{DATA_CONSUNTIVO.strftime("%Y%m%d")}#{int(DELTA_HOURS/24):02}#v1'
    NOME_BUCKET = 'eva-qa-s3-model'
    BASE_PATH = f'datamodel/BEST_CHOICE_CLUSTER/SIMULAZIONE={COD_SIMULAZIONE}/'
    LOG_FILE = f'logs/{COD_MODELLO}_{COD_SIMULAZIONE}_{start_time.strftime("%Y%m%d%H%M%S")}.log'
    PICKLE_FILE = f'modelparameters/{COD_MODELLO}_{COD_SIMULAZIONE}_{start_time.strftime("%Y%m%d%H%M%S")}.obj'

    client = boto3.client('s3')

    log_string = f'PARTENZA - {tic}\n'
    client.put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_FILE)

    df_parameters = pd.DataFrame(columns=['PERIODO', 'TIPO_MODELLO', 'MAPE', 'WMAE', 'CLUSTER'])

    CLUSTERS = []
    for x in get_subfolders_s3(bucket=NOME_BUCKET, path=BASE_PATH):
        ID_CLUSTER = unquote(x).split('/')[3].replace('CLUSTER=', '')
        for zona in ['CALA', 'CNOR', 'CSUD', 'SARD', 'SUD', 'SICI', 'NORD']:
            if ID_CLUSTER.find(zona) != -1:
                CLUSTERS.append(x)

    for cluster_path in CLUSTERS:

        print(cluster_path)

        files = list_files(bucket=NOME_BUCKET, folder_path=cluster_path)

        ID_CLUSTER = unquote(cluster_path).split('/')[3].replace('CLUSTER=', '')

        df = pd.concat(
            wr.s3.read_parquet(f's3://{NOME_BUCKET}/{parquet_file}')
            for parquet_file in files
        ).reset_index(drop=True)

        df['CONSUMI_1G'] = df['ENERGIA_1G_BEST'].combine_first(df['ENERGIA_1G_GIORNO']).fillna(0.).replace(np.nan, 0.)
        df['CONSUMI_2G'] = df['ENERGIA_2G_BEST'].fillna(0.).replace(np.nan, 0.)
        df['CONSUMI'] = df['CONSUMI_1G']+df['CONSUMI_2G']

        df = df[df.ORA_GME < 25]

        df['CONSUMI_NORMALIZZATI'] = np.where(df['N_POD'] == df['N_POD_MANCANTI'], df['CONSUMI']/df['N_POD'], df['CONSUMI']/(df['N_POD']-df['N_POD_MANCANTI']))

        df = df[df['UNIX_TIME'] < DATA_CONSUNTIVO.timestamp()]
        df = df[df['UNIX_TIME'] >= datetime(2023, 1, 1).timestamp()]

        df = df[['UNIX_TIME', 'TIMESTAMP', 'CONSUMI_NORMALIZZATI', 'N_POD', 'N_POD_MANCANTI']]
        df = df.rename(columns={'TIMESTAMP': 'TIMESTAMP_STRING', 'CONSUMI_NORMALIZZATI': 'CONSUMI'})

        df = df.astype({'UNIX_TIME': np.int64, 'CONSUMI': np.float64, 'N_POD': np.int32, 'N_POD_MANCANTI': np.int32})

        df['TIMESTAMP'] = pd.to_datetime(df['UNIX_TIME'], unit='s', utc=True).dt.tz_convert('Europe/Rome')
        df = df.set_index('TIMESTAMP').sort_index()

        if (df['CONSUMI'] > 0).all():
            MODEL_TYPES = ['addictive', 'multiplicative']
        else:
            MODEL_TYPES = ['addictive']

        test_lim = max(df.index).to_pydatetime() - relativedelta(hours=DELTA_HOURS)

        df_train = df[df.index <= test_lim][['CONSUMI', 'N_POD', 'N_POD_MANCANTI']]
        df_test = df[df.index > test_lim][['CONSUMI', 'N_POD', 'N_POD_MANCANTI']]

        best_result = []
        for period in range(2, min(24 * 32, int(np.floor(df_train.shape[0] / 2))), 2):
            for model_type in MODEL_TYPES:
                seasonal_dec_result = seasonal_decompose(df_train['CONSUMI'], model=model_type, two_sided=False, period=period)
                trend = seasonal_dec_result.trend
                seasonal = seasonal_dec_result.seasonal
                prediction_trend = trend[
                    trend.index > max(trend.index).to_pydatetime() - relativedelta(hours=DELTA_HOURS)]
                prediction_trend.index = df_test.index
                prediction_stag = seasonal[
                    seasonal.index > max(seasonal.index).to_pydatetime() - relativedelta(hours=DELTA_HOURS)]
                prediction_stag.index = df_test.index
                prediction = prediction_trend + prediction_stag
                prediction.name = 'PREVISIONE'
                prediction = prediction.to_frame()
                prediction = pd.concat([prediction, df_test['CONSUMI']], axis=1)
                best_result.append(
                    [period, model_type, mape(prediction['CONSUMI'].values, prediction['PREVISIONE'].values),
                     mae(prediction['CONSUMI'].values, prediction['PREVISIONE'].values)]
                )

        best_result = pd.DataFrame(best_result, columns=['PERIODO', 'TIPO_MODELLO', 'MAPE', 'WMAE'])
        best_result = best_result.loc[[best_result['MAPE'].idxmin()]]
        best_result['CLUSTER'] = ID_CLUSTER

        df_parameters = pd.concat([df_parameters, best_result], ignore_index=True)

        toc = datetime.now()
        log_string += f'ELABORATO CLUSTER {ID_CLUSTER} - {toc} - {(toc - tic).seconds} s\n'
        tic = toc
        del toc
        client.put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_FILE)

    df_parameters.reset_index(inplace=True, drop=True)

    pickle_byte_obj = pickle.dumps([df_parameters])
    boto3.resource('s3').Object(NOME_BUCKET, PICKLE_FILE).put(Body=pickle_byte_obj)

    toc = datetime.now()
    log_string += f'PROCESSO COMPLETATO - {toc} - {(toc - tic).seconds} s\n'
    tic = toc
    del toc
    client.put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_FILE)

