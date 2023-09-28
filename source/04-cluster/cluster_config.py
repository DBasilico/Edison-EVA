import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])

import pyspark.sql.functions as f
from pyspark.sql import SparkSession
import boto3
from datetime import date

COD_SIMULAZIONE = '1000'
NOME_TABELLA = 'CLUSTER'
NOME_BUCKET = 'eva-qa-s3-model'

DATA_VALIDITA_ANAGRAFICA = date(2023, 9, 29)
tab_path = f'datamodel/{NOME_TABELLA}/'

def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


def get_n_partitions(spark: SparkSession, factor: int = 3):
    sc = spark.sparkContext
    return factor * int(sc._conf.get('spark.executor.cores')) * (len(sc._jsc.sc().statusTracker().getExecutorInfos()))


def delete_folder_s3(bucket: str, folder_path: str):
    folder_path = check_path_s3(folder_path)
    boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=folder_path).delete()

spark = SparkSession.builder.appName('CLUSTER DA CONFIGURAZIONE: modello previsivo').enableHiveSupport().getOrCreate()

### Carichiamo la configurazione ###

df_conf = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CONFIGURAZIONE_CLUSTER/')
#df_conf = spark.read.parquet(f'C:/Users/dbasilico/Desktop/Progetti/Edison/EVA/Edison-EVA-sviluppi/CONFIGURAZIONE_CLUSTER')
df_conf = df_conf.filter(f.col('SIMULAZIONE') == COD_SIMULAZIONE).pandas_api().set_index('ORDINAMENTO')
df_conf.sort_index(inplace=True)

if not df_conf.index.is_unique:
    raise AttributeError('Presente ORDINAMENTO con valori multipli')

### Creazione dei cluster ###

df_anagrafica = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/ANAG_SII/')
#df_anagrafica = spark.read.parquet(f'C:/Users/dbasilico/Desktop/Progetti/Edison/EVA/Edison-EVA-sviluppi/ANAG_SII')

df_anagrafica = df_anagrafica.filter(
    (f.col('DT_INI_VALIDITA') <= DATA_VALIDITA_ANAGRAFICA) &
    (f.col('DT_FIN_VALIDITA') >= DATA_VALIDITA_ANAGRAFICA)
)

for k_rule in df_conf.index.to_numpy():
    print(k_rule)
    row = df_conf.loc[int(k_rule)]
    lista_pod = row['LISTA_POD']
    if lista_pod is None:
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(row['REGOLA_SELECT']))
    else:
        if 'CLUSTER' not in df_anagrafica.columns:
            raise AttributeError('ERRORE: manca una definizione di default per la creazione dei cluster')
        concat_syntax = f.concat_ws('#',
            f.coalesce(f.col('ZONA'), f.lit('')),
            f.coalesce(f.col('ID_AREA_GESTIONALE'), f.lit('')),
            f.coalesce(f.col('PROVINCIA_CRM'), f.lit('')),
            f.coalesce(f.col('TRATTAMENTO'), f.lit('')),
            f.coalesce(f.col('POD'), f.lit(''))
        )
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(lista_pod), concat_syntax).otherwise(f.col('CLUSTER')))

df_anagrafica = df_anagrafica.select(
    f.col('POD'),
    f.col('CLUSTER'),
    f.lit(COD_SIMULAZIONE).alias('COD_SIMULAZIONE')
)

if df_anagrafica.groupBy('POD').count().filter(f.col('count') > 1).count() > 0:
    df_anagrafica.join(df_anagrafica.groupBy('POD').count().filter(f.col('count') > 1).select('POD').distinct(), on=['POD'], how='inner').show(100, False)
    raise Exception('ERRORE: sono presenti POD duplicati')

df_anagrafica = df_anagrafica.repartition(get_n_partitions(spark), 'POD')

delete_folder_s3(bucket='eva-qa-s3-model', folder_path=f'datamodel/{NOME_TABELLA}/COD_SIMULAZIONE={COD_SIMULAZIONE}/')

df_anagrafica.write.mode('append').partitionBy(['COD_SIMULAZIONE']).parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}')
