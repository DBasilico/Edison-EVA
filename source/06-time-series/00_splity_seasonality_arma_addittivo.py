import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error as mae
import pyspark.pandas as pSpark
from datetime import datetime

COD_SIMULAZIONE = '1000'
COD_MODELLO = '1000' #'TRNDSTG'
NOME_TABELLA = 'FORECAST'
NOME_BUCKET = 'eva-qa-s3-model'

def datetime_to_epoch(ser):
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')
    return res // 10**9


DELTA_HOURS = 24

spark = SparkSession.builder.appName('STAGIONALITA SEMPLICE').enableHiveSupport().getOrCreate()

spark_df = spark.read.parquet('/home/mantioco/BEST_CHOICE.parquet')

spark_df = spark_df.withColumn('CONSUMI_1G', f.coalesce(f.col('ENERGIA_1G_BEST'), f.col('ENERGIA_1G_GIORNO'), f.lit(0.)))
spark_df = spark_df.withColumn('CONSUMI_2G', f.coalesce(f.col('ENERGIA_2G_BEST'), f.lit(0.)))
spark_df = spark_df.withColumn('CONSUMI', f.col('CONSUMI_1G')+f.col('CONSUMI_2G'))
spark_df = spark_df.withColumn('CONSUMI_NORMALIZZATI', f.coalesce(f.col('CONSUMI')/(f.col('N_POD')-f.col('N_POD_MANCANTI')), f.lit(0.)))

spark_df = spark_df.filter(f.split(f.col('CLUSTER'), '#')[0].isin(['CALA', 'CNOR', 'CSUD', 'SARD', 'SUD', 'SICI']))

LISTA_DATE = [
    int(datetime(2023, 9, 26).timestamp()),
    int(datetime(2023, 9, 25).timestamp()),
    int(datetime(2023, 9, 24).timestamp()),
    int(datetime(2023, 9, 23).timestamp()),
    int(datetime(2023, 9, 22).timestamp()),
    int(datetime(2023, 9, 21).timestamp()),
    int(datetime(2023, 9, 20).timestamp()),
    int(datetime(2023, 9, 19).timestamp()),
]

for DATE_CUT in LISTA_DATE:

    spark_df = spark_df.filter(f.col('UNIX_TIME') < DATE_CUT)

    CLUSTERS = [x.CLUSTER for x in spark_df.select('CLUSTER').distinct().collect()]
    N_CLUSTERS = len(CLUSTERS)

    try:
        for k_cluster, ID_CLUSTER in enumerate(CLUSTERS):
            print(DATE_CUT, k_cluster, datetime.now())

            df = spark_df.filter(f.col('CLUSTER') == ID_CLUSTER).select(
                f.col('UNIX_TIME'),
                f.col('TIMESTAMP').alias('TIMESTAMP_STRING'),
                f.col('CONSUMI_NORMALIZZATI').alias('CONSUMI'),
                f.col('N_POD'),
                f.col('N_POD_MANCANTI')
            ).toPandas()
            df['TIMESTAMP'] = pd.to_datetime(df['UNIX_TIME'], unit='s', utc=True).dt.tz_convert('Europe/Rome')
            df = df.set_index('TIMESTAMP').sort_index()

            if (df['CONSUMI'] > 0).all():
                MODEL_TYPES = ['addictive', 'multiplicative']
            else:
                MODEL_TYPES = ['addictive']

            test_lim = max(df.index).to_pydatetime() - relativedelta(hours=DELTA_HOURS)

            df_train = df[df.index <= test_lim][['CONSUMI', 'N_POD', 'N_POD_MANCANTI']]
            df_test = df[df.index > test_lim][['CONSUMI', 'N_POD', 'N_POD_MANCANTI']]

            # Grid search #
            best_result = []
            for period in range(2, min(24*32, int(np.floor(df_train.shape[0]/2))), 2):
                for model_type in MODEL_TYPES:
                    seasonal_dec_result = seasonal_decompose(df_train['CONSUMI'], model=model_type, two_sided=False, period=period)
                    trend = seasonal_dec_result.trend
                    seasonal = seasonal_dec_result.seasonal
                    prediction_trend = trend[trend.index > max(trend.index).to_pydatetime()-relativedelta(hours=DELTA_HOURS)]
                    prediction_trend.index = df_test.index
                    prediction_stag = seasonal[seasonal.index > max(seasonal.index).to_pydatetime()-relativedelta(hours=DELTA_HOURS)]
                    prediction_stag.index = df_test.index
                    prediction = prediction_trend+prediction_stag
                    prediction.name = 'PREVISIONE'
                    prediction = prediction.to_frame()
                    prediction = pd.concat([prediction, df_test['CONSUMI']], axis=1)
                    best_result.append([period, model_type, mape(prediction['CONSUMI'].values, prediction['PREVISIONE'].values), mae(prediction['CONSUMI'].values, prediction['PREVISIONE'].values)])

            best_result = pd.DataFrame(best_result, columns=['PERIODO', 'TIPO_MODELLO', 'MAPE', 'WMAE'])

            best_mape_idx = best_result['MAPE'].idxmin()
            best_period = best_result.loc[best_mape_idx, 'PERIODO']
            best_model = best_result.loc[best_mape_idx, 'TIPO_MODELLO']

            seasonal_dec_result = seasonal_decompose(df_train['CONSUMI'], model=best_model, two_sided=False, period=best_period)

            trend = seasonal_dec_result.trend
            seasonal = seasonal_dec_result.seasonal

            prediction_trend = trend[trend.index > max(trend.index).to_pydatetime()-relativedelta(hours=DELTA_HOURS)]
            prediction_trend.index = df_test.index

            prediction_stag = seasonal[seasonal.index > max(seasonal.index).to_pydatetime()-relativedelta(hours=DELTA_HOURS)]
            prediction_stag.index = df_test.index

            prediction = prediction_trend+prediction_stag
            prediction.name = 'PREVISIONE'

            prediction = prediction.to_frame()
            prediction = pd.concat([prediction, df_test], axis=1)
            prediction['UNIX_TIME'] = datetime_to_epoch(prediction.index)
            prediction['CLUSTER'] = ID_CLUSTER
            prediction['CONSUMI'] = prediction['CONSUMI']*(prediction['N_POD']-prediction['N_POD_MANCANTI'])
            prediction['PREVISIONE'] = prediction['PREVISIONE'] * (prediction['N_POD'] - prediction['N_POD_MANCANTI'])
            prediction['MAPE'] = mape(prediction['CONSUMI'].values, prediction['PREVISIONE'].values)
            prediction['WMAE'] = mae(prediction['CONSUMI'].values, prediction['PREVISIONE'].values)
            prediction.reset_index(drop=True, inplace=True)

            df_out = pSpark.from_pandas(prediction).to_spark()

            df_out = df_out.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))
            df_out = df_out.withColumn('MODELLO', f.lit(COD_MODELLO))
            df_out = df_out.withColumn('DATA_PREVISIONE_UNIX', f.lit(DATE_CUT-86400).cast('bigint'))

            df_out.write.mode('append').partitionBy('SIMULAZIONE', 'MODELLO', 'DATA_PREVISIONE_UNIX').parquet(f'/home/mantioco/{NOME_TABELLA}')
    except Exception as e:
        raise e
