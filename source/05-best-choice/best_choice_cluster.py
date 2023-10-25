import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])

from pyspark.sql import functions as f
from pyspark.sql import SparkSession
import boto3
import sys


def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


def path_exists_s3(bucket: str, path: str, is_file: bool):
    path = check_path_s3(path, is_file)
    files = list(boto3.session.Session().resource('s3').Bucket(bucket).objects.filter(Prefix=path))
    return len(files) > 0


def delete_folder_s3(bucket: str, folder_path: str):
    if path_exists_s3(bucket=bucket, path=folder_path, is_file=False):
        folder_path = check_path_s3(folder_path)
        boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=folder_path).delete()

spark = SparkSession.builder.appName('BEST CHOICE: ripartizione per cluster').enableHiveSupport().getOrCreate()

COD_SIMULAZIONE = sys.argv[1]
NOME_TABELLA = 'BEST_CHOICE_CLUSTER'
NOME_BUCKET = 'eva-qa-s3-model'

df_tempo = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel//CALENDARIO_GME/')
df_tempo = df_tempo.filter(f.col('TIMESTAMP').like(f'{sys.argv[2]} 00:00:00 %')).collect()

UNIX_TIME_INIZIO_TEST = df_tempo[0]['UNIX_TIME']

df_best_choice = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/BEST_CHOICE/')
df_best_choice = df_best_choice.filter(f.col('SIMULAZIONE') == COD_SIMULAZIONE)

df_best_choice = df_best_choice.withColumn('FLAG_TRAIN', f.when(f.col('UNIX_TIME') < UNIX_TIME_INIZIO_TEST, f.lit('TRAIN')).otherwise(f.lit('TEST')))

delete_folder_s3(bucket=NOME_BUCKET, folder_path=f'datamodel/{NOME_TABELLA}/SIMULAZIONE={COD_SIMULAZIONE}/')
df_best_choice.orderBy(f.col('UNIX_TIME').asc()).write.mode('append').partitionBy(['SIMULAZIONE', 'FLAG_TRAIN', 'CLUSTER']).parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}')

