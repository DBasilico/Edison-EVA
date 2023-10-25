import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pmdarima"])

from datetime import datetime
from urllib.parse import unquote
import boto3
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error as mae
import pickle
import pathlib
from dateutil import tz
from pmdarima.arima import auto_arima


def datetime_to_epoch(ser):
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')
    return res // 10 ** 9


print('Iniziooooo')

import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from dateutil import tz

ROME_TZ = tz.gettz('Europe/Rome')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--COD_SIMULAZIONE', type=str, default='default_value',
                        help='Description of the test hyperparameter')
    return parser.parse_args()


def datetime_to_epoch(ser):
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')
    return res // 10 ** 9


args = parse_args()

# Leggi gli iperparametri
test_hyperparameter = args.COD_SIMULAZIONE
print(test_hyperparameter)

from datetime import datetime
from urllib.parse import unquote
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error as mae
import os
import pathlib

try:

    def datetime_to_epoch(ser):
        if ser.hasnans:
            res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
        else:
            res = ser.astype('int64')
        return res // 10 ** 9


    RUN_PARAMETERS = [
        ['1000', datetime(2023, 9, 18)]
    ]

    result = dict()

    for COD_SIMULAZIONE, DATA_CONSUNTIVO in RUN_PARAMETERS:
        print(COD_SIMULAZIONE, DATA_CONSUNTIVO)
        tic = datetime.now()
        start_time = tic
        df_parameters = pd.DataFrame(columns=['PERIODO', 'TIPO_MODELLO', 'MAPE', 'WMAE', 'CLUSTER'])
        cluster_dir = '/opt/ml/input/data/train'
        CLUSTERS = []
        for x in [name for name in os.listdir(cluster_dir) if os.path.isdir(os.path.join(cluster_dir, name))]:
            print(x)
            ID_CLUSTER = unquote(x).replace('CLUSTER=', '')
            for zona in ['CALA', 'CNOR', 'CSUD', 'SARD', 'SUD', 'SICI', 'NORD']:
                if ID_CLUSTER.find(zona) != -1:
                    CLUSTERS.append(x)

        cluster_result = dict()
        for cluster_path in CLUSTERS:
            print(cluster_path)
            files = [f for f in os.listdir(os.path.join(cluster_dir, cluster_path))]
            print(files)
            ID_CLUSTER = unquote(cluster_path).replace('CLUSTER=', '')
            df = pd.concat(
                pd.read_parquet(os.path.join(cluster_dir, cluster_path, parquet_file))
                for parquet_file in files if pathlib.Path(parquet_file).suffix == '.parquet'
            ).reset_index(drop=True)
            df['CONSUMI_1G'] = df['ENERGIA_1G_BEST'].combine_first(df['ENERGIA_1G_GIORNO']).fillna(0.).replace(np.nan,
                                                                                                               0.)
            df['CONSUMI_2G'] = df['ENERGIA_2G_BEST'].fillna(0.).replace(np.nan, 0.)
            df['CONSUMI'] = df['CONSUMI_1G'] + df['CONSUMI_2G']
            df = df[df.ORA_GME < 25]
            df['CONSUMI_NORMALIZZATI'] = np.where(df['N_POD'] == df['N_POD_MANCANTI'], df['CONSUMI'] / df['N_POD'],
                                                  df['CONSUMI'] / (df['N_POD'] - df['N_POD_MANCANTI']))
            df = df[df['UNIX_TIME'] < DATA_CONSUNTIVO.timestamp()]
            df = df[df['UNIX_TIME'] >= datetime(2023, 1, 1).timestamp()].copy()
            df = df[['UNIX_TIME', 'TIMESTAMP', 'CONSUMI_NORMALIZZATI', 'N_POD', 'N_POD_MANCANTI']]
            df = df.rename(columns={'TIMESTAMP': 'TIMESTAMP_STRING', 'CONSUMI_NORMALIZZATI': 'CONSUMI'})
            df = df.astype(
                {'UNIX_TIME': np.int64, 'CONSUMI': np.float64, 'N_POD': np.int32, 'N_POD_MANCANTI': np.int32})
            df['TIMESTAMP'] = pd.to_datetime(df['UNIX_TIME'], unit='s', utc=True).dt.tz_convert('Europe/Rome')
            df = df.set_index('TIMESTAMP').sort_index()

            cluster_result[ID_CLUSTER] = auto_arima(df['CONSUMI'], m=52)

        result[f'{COD_SIMULAZIONE}{DATA_CONSUNTIVO.strftime("%Y%m%d")}'] = cluster_result

except Exception as e:
    print(e)
    raise Exception('ERRORE')

# Salva il modello
import joblib

joblib.dump(result, '/opt/ml/model/model_train_eva_arima.joblib')

