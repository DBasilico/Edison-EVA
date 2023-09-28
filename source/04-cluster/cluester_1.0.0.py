import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import boto3

COD_SIMULAZIONE = '1000'
NOME_TABELLA = 'CLUSTER'
NOME_BUCKET = 'eva-qa-s3-model'

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


spark = SparkSession.builder.appName('CLUSTER v1.0.0').enableHiveSupport().getOrCreate()

df = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/ANAG_SII/')

zona#area#provincia#trattamento#pod

cond_cluster = f.when(
        f.col('ID_AREA_GESTIONALE') == 'RES',
        f.concat_ws('#', f.col('ZONA'), f.col('ID_AREA_GESTIONALE'), f.col('PROVINCIA_CRM'))
    ).otherwise(
        f.concat_ws('#', f.col('ZONA'), f.col('ID_AREA_GESTIONALE'))
)

# TODO: da pensare come calcolare bene i pod perchè al memoento mettiamo un drop_duplicates
df = df.select(
    f.col('POD'),
    cond_cluster.alias('CLUSTER'),
    f.lit(COD_SIMULAZIONE).alias('COD_SIMULAZIONE')
).dropDuplicates(subset=['POD'])

df = df.repartition(get_n_partitions(spark), 'POD')

delete_folder_s3(bucket='eva-qa-s3-model', folder_path=f'datamodel/{NOME_TABELLA}/{COD_SIMULAZIONE}/')

df.write.mode('append').partitionBy(['COD_SIMULAZIONE']).parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}')

